
import os
import shutil
import uuid
import hmac

from dataclasses import dataclass

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    Depends,
    Header,
    Cookie,
    Request,
    Response,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from core.project_registry import ProjectRegistry
from core.project_runtime import ProjectRuntime
from core.project_loader import load_project_settings

from llm import LLMServiceError

from core.security import (
    APIKeyConfigurationError,
    resolve_project_for_api_key,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database.session import get_db, get_session_factory
from database.models import Conversation
from core.auth_service import (
    AuthService,
    UsernameAlreadyExists,
    InvalidCredentials,
    InvalidRefreshToken,
)
from core.auth_dependencies import (
    AuthContext,
    bearer_scheme,
    get_current_user,
    resolve_access_token,
)
from core.auth_cookies import (
    ACCESS_COOKIE_NAME,
    REFRESH_COOKIE_NAME,
    CSRF_COOKIE_NAME,
    set_auth_cookies,
    clear_auth_cookies,
)
from core.auth_schemas import (
    RegisterRequest,
    LoginRequest,
)
from core.conversation_service import (
    ConversationService,
    ConversationNotFound,
)
from core.chat_service import ChatService

# ==================================================
# APPLICATION SETUP
# ==================================================

load_dotenv()

app = FastAPI(
    title="Reusable AI Assistant Core",
    version="1.0.0",
)

origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-CSRF-Token"],
    expose_headers=["X-Conversation-ID"],
)

ALLOWED_FRONTEND_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in origins
    if origin.strip()
}


@app.middleware("http")
async def csrf_protection(request: Request, call_next):
    """
    Protect cookie-authenticated state-changing requests against CSRF.

    Browser clients must echo the CSRF token value in X-CSRF-Token.
    Bearer/API-key integrations without auth cookies remain unchanged.
    """
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") not in ALLOWED_FRONTEND_ORIGINS:
            return JSONResponse(
                status_code=403,
                content={"detail": "Request origin is not allowed."},
            )

        has_auth_cookie = bool(
            request.cookies.get(ACCESS_COOKIE_NAME)
            or request.cookies.get(REFRESH_COOKIE_NAME)
        )

        if has_auth_cookie:
            csrf_cookie = request.cookies.get(CSRF_COOKIE_NAME)
            csrf_header = request.headers.get("X-CSRF-Token")

            if (
                not csrf_cookie
                or not csrf_header
                or not hmac.compare_digest(csrf_cookie, csrf_header)
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF validation failed."},
                )

    return await call_next(request)


STORAGE_PATH = "storage"
PDF_ROOT = "data/projects"


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

DEFAULT_PROJECT_ID, PROJECT_CONFIGS, PROJECT_PROMPTS = (
    load_project_settings()
)

# Optional global fallback prompt from .env.
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")


# ==================================================
# LEGACY STORAGE MIGRATION
# ==================================================

def migrate_legacy_storage() -> None:
    """
    Copy the legacy single-project FAISS index and
    documents into the default project's storage.

    Original legacy files remain untouched.
    """

    legacy_index = os.path.join(
        STORAGE_PATH,
        "index.faiss",
    )

    legacy_documents = os.path.join(
        STORAGE_PATH,
        "documents.pkl",
    )

    project_storage = os.path.join(
        STORAGE_PATH,
        "projects",
        DEFAULT_PROJECT_ID,
    )

    project_snapshot = os.path.join(
        project_storage,
        "vectorstore.snapshot",
    )

    if os.path.isfile(project_snapshot):
        return

    project_index = os.path.join(
        project_storage,
        "index.faiss",
    )

    project_documents = os.path.join(
        project_storage,
        "documents.pkl",
    )

    if not os.path.isfile(legacy_index):
        return

    index_exists = os.path.isfile(project_index)
    documents_exist = os.path.isfile(project_documents)

    # Never overwrite an existing project-specific store.
    if index_exists and documents_exist:
        return

    # Do not silently repair or overwrite a partial store.
    if index_exists or documents_exist:
        raise RuntimeError(
            "The default project's storage is incomplete. "
            "Check index.faiss and documents.pkl before "
            "starting the application."
        )

    if not os.path.isfile(legacy_documents):
        raise RuntimeError(
            "Legacy index.faiss exists, but documents.pkl "
            "is missing. Migration cannot continue safely."
        )

    os.makedirs(
        project_storage,
        exist_ok=True,
    )

    temporary_index = (
        f"{project_index}.migration-{uuid.uuid4().hex}"
    )

    temporary_documents = (
        f"{project_documents}.migration-{uuid.uuid4().hex}"
    )

    try:
        shutil.copy2(
            legacy_index,
            temporary_index,
        )

        shutil.copy2(
            legacy_documents,
            temporary_documents,
        )

        # Install documents before the index.
        os.replace(
            temporary_documents,
            project_documents,
        )

        os.replace(
            temporary_index,
            project_index,
        )

    finally:
        for path in (
            temporary_index,
            temporary_documents,
        ):
            if os.path.exists(path):
                os.remove(path)


# Migrate before initializing the project runtimes.
migrate_legacy_storage()


# ==================================================
# PROJECT REGISTRY AND RUNTIMES
# ==================================================

project_registry = ProjectRegistry(
    projects=PROJECT_CONFIGS,
)


def resolve_system_prompt(
    project_id: str,
) -> str | None:
    """
    Use the project's prompt when configured.
    Otherwise, use the optional global fallback.
    """

    project_prompt = PROJECT_PROMPTS.get(project_id)

    if project_prompt is not None:
        return project_prompt

    return SYSTEM_PROMPT


project_runtimes: dict[str, ProjectRuntime] = {
    config.name: ProjectRuntime(
        config=config,
        storage_root=STORAGE_PATH,
        pdf_root=PDF_ROOT,
        system_prompt=resolve_system_prompt(config.name),
    )
    for config in PROJECT_CONFIGS
}


def authenticate_project(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> str:
    """Authenticate the request and return its project ID."""

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    try:
        project_id = resolve_project_for_api_key(
            x_api_key,
            project_runtimes.keys(),
        )
    except APIKeyConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Project API key configuration is invalid.",
        ) from exc

    if project_id is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return project_id


@dataclass
class RequestPrincipal:
    project_id: str
    auth_context: AuthContext | None = None


def get_request_principal(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    access_token: str | None = Cookie(
        default=None,
        alias=ACCESS_COOKIE_NAME,
    ),
) -> RequestPrincipal:
    """Accept a browser access cookie, a Bearer token, or a legacy API key."""

    # Bearer credentials take precedence for backwards compatibility.
    token = (
        credentials.credentials
        if credentials is not None
        else access_token
    )

    # Open a database connection only for user-token authentication.
    if token is not None:
        session_factory = get_session_factory()
        with session_factory() as auth_db:
            auth_context = resolve_access_token(
                token,
                auth_db,
            )

        return RequestPrincipal(
            project_id=auth_context.project_id,
            auth_context=auth_context,
        )

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer or ApiKey"},
        )

    try:
        project_id = resolve_project_for_api_key(
            x_api_key,
            project_runtimes.keys(),
        )
    except APIKeyConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Project API key configuration is invalid.",
        ) from exc

    if project_id is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return RequestPrincipal(project_id=project_id)


def resolve_auth_project(project_id: str | None) -> str:
    """Resolve an optional auth project to a configured project ID."""

    return get_runtime(project_id).config.name


def get_authorized_runtime(
    authenticated_project_id: str,
    requested_project_id: str | None = None,
) -> ProjectRuntime:
    """Prevent a client from accessing another project."""

    if (
        requested_project_id
        and requested_project_id.strip()
        != authenticated_project_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Access to this project is forbidden.",
        )

    return get_runtime(
    project_id=authenticated_project_id
    )



def get_runtime(
    project_id: str | None = None,
) -> ProjectRuntime:
    """
    Resolve a registered project to its runtime.

    Missing project IDs use the configured default.
    Unknown IDs are rejected and never auto-registered.
    """

    if project_id is None or not project_id.strip():
        project_id = DEFAULT_PROJECT_ID
    else:
        project_id = project_id.strip()

    try:
        config = project_registry.get(project_id)

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown project: {project_id}",
        ) from exc

    runtime = project_runtimes.get(config.name)

    if runtime is None:
        raise HTTPException(
            status_code=500,
            detail="Project runtime is not initialized.",
        )

    return runtime


# ==================================================
# REQUEST MODELS
# ==================================================

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    mode: str = Field(
        default="rag",
    )

    conversation_id: str | None = Field(
        default=None,
        max_length=128,
    )

    project_id: str | None = Field(
        default=None,
        max_length=128,
    )

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot be empty."
            )

        return value

    @field_validator("mode")
    @classmethod
    def validate_mode(
        cls,
        value: str,
    ) -> str:
        value = value.strip().lower()

        if value not in {"chat", "rag", "hybrid"}:
            raise ValueError(
                "Mode must be chat, rag, or hybrid."
            )

        return value

    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None

    @field_validator("project_id")
    @classmethod
    def validate_project_id(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None


# ==================================================
# AUTHENTICATION ENDPOINTS
# ==================================================

@app.post("/auth/register", status_code=201)
def register(
    request: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    # Registration also creates a session, so the browser is logged in
    # immediately after successful sign-up.
    project_id = resolve_auth_project(request.project_id)

    try:
        service = AuthService(db)
        user = service.register(
            username=request.username,
            password=request.password,
        )
        tokens = service.login(
            username=request.username,
            password=request.password,
            project_id=project_id,
        )
    except UsernameAlreadyExists as exc:
        raise HTTPException(
            status_code=409,
            detail="Username is already registered.",
        ) from exc
    except InvalidCredentials as exc:
        raise HTTPException(
            status_code=401,
            detail="Registration succeeded but automatic sign-in failed.",
        ) from exc

    csrf_token = set_auth_cookies(response, tokens)

    return {
        "user": {
            "id": str(user.id),
            "username": user.username,
        },
        "project_id": project_id,
        "expires_in": tokens["expires_in"],
        "csrf_token": csrf_token,
    }


@app.post("/auth/login")
def login(
    request: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    project_id = resolve_auth_project(request.project_id)

    try:
        result = AuthService(db).login(
            username=request.username,
            password=request.password,
            project_id=project_id,
        )
    except InvalidCredentials as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        ) from exc

    user = result["user"]
    csrf_token = set_auth_cookies(response, result)

    # Access and refresh tokens are deliberately not returned in JSON.
    # The CSRF token is safe to expose and is required in a custom header.
    return {
        "user": {
            "id": str(user.id),
            "username": user.username,
        },
        "project_id": project_id,
        "expires_in": result["expires_in"],
        "csrf_token": csrf_token,
    }


@app.post("/auth/refresh")
def refresh_access_token(
    response: Response,
    refresh_token: str | None = Cookie(
        default=None,
        alias=REFRESH_COOKIE_NAME,
    ),
    db: Session = Depends(get_db),
):
    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail="Refresh cookie is missing.",
        )

    try:
        result = AuthService(db).refresh(refresh_token)
    except InvalidRefreshToken as exc:
        raise HTTPException(
            status_code=401,
            detail="Refresh token is invalid, expired, or revoked.",
        ) from exc

    csrf_token = set_auth_cookies(response, result)

    return {
        "status": "refreshed",
        "expires_in": result["expires_in"],
        "csrf_token": csrf_token,
    }


@app.get("/auth/csrf")
def get_csrf_token(
    response: Response,
    csrf_token: str | None = Cookie(
        default=None,
        alias=CSRF_COOKIE_NAME,
    ),
):
    if not csrf_token:
        raise HTTPException(
            status_code=401,
            detail="CSRF cookie is missing.",
        )

    response.headers["Cache-Control"] = "no-store"
    return {"csrf_token": csrf_token}


@app.post("/auth/logout")
def logout(
    response: Response,
    credentials: HTTPAuthorizationCredentials | None = Depends(
        bearer_scheme
    ),
    access_token: str | None = Cookie(
        default=None,
        alias=ACCESS_COOKIE_NAME,
    ),
    refresh_token: str | None = Cookie(
        default=None,
        alias=REFRESH_COOKIE_NAME,
    ),
    db: Session = Depends(get_db),
):
    # Prefer a valid access token; if it has expired, revoke by refresh token.
    token = (
        credentials.credentials
        if credentials is not None
        else access_token
    )

    auth = None
    if token:
        try:
            auth = resolve_access_token(token, db)
        except HTTPException:
            auth = None

    service = AuthService(db)
    if auth is not None:
        service.logout(auth.session_id)
    elif refresh_token:
        service.logout_refresh_token(refresh_token)

    # Always expire browser cookies, even when the access token has expired.
    clear_auth_cookies(response)
    return {"status": "logged_out"}


@app.get("/auth/me")
def get_me(
    auth: AuthContext = Depends(get_current_user),
):
    return {
        "id": str(auth.user.id),
        "username": auth.user.username,
        "project_id": auth.project_id,
    }


# ==================================================
# CONVERSATION HISTORY ENDPOINTS
# ==================================================

@app.get("/conversations")
def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    auth: AuthContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items = ConversationService(db).list_conversations(
        user_id=auth.user.id,
        project_id=auth.project_id,
        limit=limit,
    )
    return {"count": len(items), "conversations": items}


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid conversation ID.",
        ) from exc

    try:
        return ConversationService(db).get_conversation(
            conversation_id=conversation_uuid,
            user_id=auth.user.id,
            project_id=auth.project_id,
        )
    except ConversationNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        ) from exc


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    auth: AuthContext = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        conversation_uuid = uuid.UUID(conversation_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid conversation ID.",
        ) from exc

    deleted = ConversationService(db).delete_conversation(
        conversation_id=conversation_uuid,
        user_id=auth.user.id,
        project_id=auth.project_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    db.commit()
    return {"status": "deleted", "conversation_id": conversation_id}


# ==================================================
# REQUEST VALIDATION
# ==================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request,
    exc: RequestValidationError,
):
    errors = exc.errors()

    if any(
        error.get("loc")
        and error["loc"][-1] == "question"
        for error in errors
    ):
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    "Question must not be empty "
                    "and must be valid."
                )
            },
        )

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request data."
        },
    )


# ==================================================
# LLM ERROR HANDLING
# ==================================================

def to_http_exception(
    exc: LLMServiceError,
) -> HTTPException:
    status_codes = {
        "configuration_error": 500,
        "authentication_error": 502,
        "rate_limit": 503,
        "timeout": 504,
        "connection_error": 502,
        "provider_error": 502,
        "empty_response": 502,
        "llm_error": 502,
    }

    return HTTPException(
        status_code=status_codes.get(
            exc.code,
            502,
        ),
        detail={
            "code": exc.code,
            "message": str(exc),
        },
    )


def safe_stream(generator):
    """
    Convert streaming LLM errors into stream messages.
    """

    try:
        yield from generator

    except LLMServiceError as exc:
        yield (
            f"\n[LLM_ERROR:{exc.code}] "
            f"{str(exc)}"
        )



# ==================================================
# HEALTH ENDPOINT
# ==================================================

@app.get("/health")
def health(
    project_id: str | None = Query(default=None),
    principal: RequestPrincipal = Depends(get_request_principal),
):
    runtime = get_authorized_runtime(
        principal.project_id,
        project_id,
    )

    config = runtime.config

    return {
        "status": "ok",
        "project": config.name,
        "features": {
            "chat": config.chat_enabled,
            "rag": config.rag_enabled,
            "hybrid": (
                config.chat_enabled
                and config.rag_enabled
            ),
        },
        "documents_ready": (
            runtime.rag_service is not None
        ),
    }



# ==================================================
# PDF UPLOAD ENDPOINT
# ==================================================

@app.post("/upload")
def upload_pdf(
    file: UploadFile = File(...),
    project_id: str | None = Query(
        default=None,
    ),
    principal: RequestPrincipal = Depends(get_request_principal),
):
    runtime = get_authorized_runtime(principal.project_id, project_id)

    if not runtime.config.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    # Prevent directory traversal.
    filename = os.path.basename(file.filename)

    if (
        not filename
        or filename in {".", ".."}
        or not filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    os.makedirs(
        runtime.pdf_path,
        exist_ok=True,
    )

    file_path = os.path.join(
        runtime.pdf_path,
        filename,
    )

    # Preserve an existing PDF if indexing fails.
    backup_path = None

    if os.path.isfile(file_path):
        backup_path = (
            f"{file_path}.backup-{uuid.uuid4().hex}"
        )

        shutil.copy2(
            file_path,
            backup_path,
        )

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        chunks_count = runtime.index_pdf(
            file_path,
        )

    except Exception:
        if (
            backup_path
            and os.path.isfile(backup_path)
        ):
            os.replace(
                backup_path,
                file_path,
            )

        elif os.path.isfile(file_path):
            os.remove(file_path)

        raise

    else:
        if (
            backup_path
            and os.path.isfile(backup_path)
        ):
            os.remove(backup_path)

    return {
        "project": runtime.config.name,
        "filename": filename,
        "chunks": chunks_count,
        "status": "ready",
    }



# ==================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ==================================================

@app.get("/documents")
def list_documents(
    project_id: str | None = Query(default=None),
    principal: RequestPrincipal = Depends(get_request_principal),
):
    runtime = get_authorized_runtime(
        principal.project_id,
        project_id,
    )

    if not runtime.config.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    documents = runtime.list_documents()

    return {
        "project": runtime.config.name,
        "count": len(documents),
        "documents": documents,
    }


@app.delete("/documents/{filename}")
def delete_document(
    filename: str,
    project_id: str | None = Query(default=None),
    principal: RequestPrincipal = Depends(get_request_principal),
):
    runtime = get_authorized_runtime(
        principal.project_id,
        project_id,
    )

    if not runtime.config.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    try:
        chunks_deleted = runtime.delete_document(filename)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "project": runtime.config.name,
        "filename": filename,
        "chunks_deleted": chunks_deleted,
        "status": "deleted",
    }



# ==================================================
# CHAT / RAG / HYBRID ENDPOINT
# ==================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    principal: RequestPrincipal = Depends(get_request_principal),
):
    runtime = get_authorized_runtime(
        principal.project_id,
        request.project_id,
    )
    config = runtime.config
    mode = request.mode

    if mode == "chat" and not config.chat_enabled:
        raise HTTPException(
            status_code=403,
            detail="Chat is disabled for this project.",
        )

    if mode == "rag" and not config.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    if mode == "hybrid" and not (
        config.chat_enabled and config.rag_enabled
    ):
        raise HTTPException(
            status_code=403,
            detail=(
                "Hybrid mode requires Chat and RAG "
                "to be enabled."
            ),
        )

    if mode in {"rag", "hybrid"} and runtime.rag_service is None:
        raise HTTPException(
            status_code=409,
            detail="No documents uploaded yet.",
        )

    try:
        # JWT-authenticated requests use durable PostgreSQL history.
        if principal.auth_context is not None:
            auth = principal.auth_context
            session_factory = get_session_factory()

            with session_factory() as db:
                conversations = ConversationService(db)
                conversation = conversations.get_or_create(
                    user_id=auth.user.id,
                    project_id=auth.project_id,
                    mode=mode,
                    question=request.question,
                    conversation_id=request.conversation_id,
                )
                history = conversations.get_history(conversation.id)

                chat_service = None
                if mode in {"chat", "hybrid"}:
                    chat_service = ChatService(
                        system_prompt=runtime.system_prompt,
                        history=history,
                    )

                result = runtime.ask(
                    request.question,
                    mode=mode,
                    conversation_id=str(conversation.id),
                    chat_service=chat_service,
                    history=history,
                )

                conversations.save_turn(
                    conversation=conversation,
                    question=request.question,
                    answer=result["answer"],
                    mode=mode,
                    sources=result.get("sources", []),
                )
                db.commit()
                result["conversation_id"] = str(conversation.id)

        else:
            # Preserve the existing API-key workflow for integrations.
            result = runtime.ask(
                request.question,
                mode=mode,
                conversation_id=request.conversation_id,
            )

        result["project_id"] = config.name
        return result

    except ConversationNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        ) from exc

    except ValueError as exc:
        if "conversation id" in str(exc).lower():
            raise HTTPException(
                status_code=400,
                detail="Invalid conversation ID.",
            ) from exc
        raise

    except LLMServiceError as exc:
        raise to_http_exception(exc) from exc


# ==================================================
# STREAMING RAG ENDPOINT
# ==================================================

@app.post("/chat-stream")
def chat_stream(
    request: ChatRequest,
    principal: RequestPrincipal = Depends(get_request_principal),
):
    runtime = get_authorized_runtime(
        principal.project_id,
        request.project_id,
    )

    if request.mode != "rag":
        raise HTTPException(
            status_code=400,
            detail="Streaming currently supports RAG mode only.",
        )

    if not runtime.config.rag_enabled:
        raise HTTPException(
            status_code=403,
            detail="RAG is disabled for this project.",
        )

    if runtime.rag_service is None:
        raise HTTPException(
            status_code=409,
            detail="No documents uploaded yet.",
        )

    try:
        if principal.auth_context is None:
            # Keep legacy API-key streaming unchanged.
            generator = runtime.stream_answer(request.question)
            return StreamingResponse(
                safe_stream(generator),
                media_type="text/plain",
            )

        auth = principal.auth_context
        session_factory = get_session_factory()
        with session_factory() as db:
            conversations = ConversationService(db)
            conversation = conversations.get_or_create(
                user_id=auth.user.id,
                project_id=auth.project_id,
                mode="rag",
                question=request.question,
                conversation_id=request.conversation_id,
            )
            history = conversations.get_history(conversation.id)
            conversation_id = conversation.id
            db.commit()

        generator = runtime.stream_answer(
            request.question,
            history=history,
        )

    except ConversationNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid conversation ID.",
        ) from exc
    except LLMServiceError as exc:
        raise to_http_exception(exc) from exc

    def persisted_stream():
        answer_parts = []
        completed = True

        try:
            for chunk in safe_stream(generator):
                if "[LLM_ERROR:" in chunk:
                    completed = False
                else:
                    answer_parts.append(chunk)
                yield chunk
        except GeneratorExit:
            completed = False
            raise
        except Exception:
            completed = False
            raise
        finally:
            if completed:
                # Use a fresh session because the HTTP response may outlive
                # the request-scoped dependency session.
                session_factory = get_session_factory()
                with session_factory() as stream_db:
                    stored_conversation = stream_db.get(
                        Conversation,
                        conversation_id,
                    )
                    if stored_conversation is not None:
                        ConversationService(stream_db).save_turn(
                            conversation=stored_conversation,
                            question=request.question,
                            answer="".join(answer_parts),
                            mode="rag",
                            sources=[],
                        )
                        stream_db.commit()

    return StreamingResponse(
        persisted_stream(),
        media_type="text/plain",
        headers={"X-Conversation-ID": str(conversation_id)},
    )


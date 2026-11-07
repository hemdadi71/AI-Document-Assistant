# Authentication and Conversation API

The API supports browser authentication with HttpOnly cookies and keeps the
existing project API-key integration for server-to-server clients. Bearer
access tokens remain accepted for backward compatibility, but browser clients
should use cookies and must never store access or refresh tokens in localStorage.

## Browser authentication cookies

Successful `POST /auth/register` and `POST /auth/login` responses set:

- `ai_access_token`: HttpOnly access cookie, path `/`.
- `ai_refresh_token`: HttpOnly refresh cookie, path `/auth`.
- `ai_csrf_token`: HttpOnly CSRF cookie, path `/`; the CSRF value is also returned in the response JSON.

Access and refresh tokens are not returned in the response JSON. Registration
automatically signs the new user in.

The API uses these environment variables:

```dotenv
AUTH_SECRET_KEY=<random-secret-at-least-32-characters>
AUTH_ACCESS_TOKEN_MINUTES=15
AUTH_REFRESH_TOKEN_DAYS=30
AUTH_COOKIE_SECURE=false
AUTH_COOKIE_SAMESITE=lax
FRONTEND_ORIGINS=http://localhost:3000
```

Generate a secret with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

For production, use HTTPS and set `AUTH_COOKIE_SECURE=true`. `SameSite=None`
requires Secure cookies. Prefer deploying frontend and API on subdomains of the
same site and using `SameSite=Lax`; if they are genuinely cross-site, configure
`SameSite=None; Secure` and verify browser third-party-cookie restrictions.
Set `FRONTEND_ORIGINS` to exact origins, including scheme and port where needed.
Do not commit secrets or `.env` files.

## CSRF protection

For unsafe methods (`POST`, `PUT`, `PATCH`, `DELETE`), when an access or refresh
cookie is present, the browser must send the value of `ai_csrf_token` in the
`X-CSRF-Token` header. The server compares the cookie and header values and
rejects mismatches with HTTP 403. If an Origin header is present, it must match
one of `FRONTEND_ORIGINS`.

The frontend must use `credentials: "include"`. Keep the `csrf_token` value
returned by register/login/refresh in memory and send it as `X-CSRF-Token`.
Because the CSRF cookie is HttpOnly and host-only, a page reload should call
`GET /auth/csrf` with credentials to obtain the current CSRF value again.

```ts
const session = await fetch(`${API_URL}/auth/login`, {
  method: "POST",
  credentials: "include",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    username: "demo_user",
    password: "a-strong-password",
    project_id: "chat_only_demo",
  }),
}).then((response) => response.json());

let csrfToken: string = session.csrf_token;

await fetch(`${API_URL}/auth/logout`, {
  method: "POST",
  credentials: "include",
  headers: { "X-CSRF-Token": csrfToken },
});
```

`GET /auth/csrf` returns the current CSRF value when a CSRF cookie exists.
It is not an authentication endpoint and does not expose access/refresh tokens.

## Authentication endpoints

### Register

`POST /auth/register`

```json
{
  "username": "demo_user",
  "password": "a-strong-password",
  "project_id": "chat_only_demo"
}
```

`project_id` is optional and defaults to the configured default project.
Successful registration creates the user, creates a login session, and sets
the three cookies.

### Login

`POST /auth/login`

```json
{
  "username": "demo_user",
  "password": "a-strong-password",
  "project_id": "chat_only_demo"
}
```

Returns user details and `expires_in`; authentication tokens are set as cookies
and are not included in JSON.

### CSRF bootstrap

`GET /auth/csrf`

Returns the current CSRF token value for a browser session. Call this after a
page reload, using `credentials: "include"`.

### Refresh

`POST /auth/refresh`

No JSON body is required. The browser sends the HttpOnly refresh cookie
automatically. The endpoint rotates the refresh token and updates all cookies.
The CSRF header is required because the refresh cookie is present. The response returns the rotated `csrf_token` value.

### Logout

`POST /auth/logout`

The browser sends the access cookie automatically. The endpoint revokes the
current database session and expires access, refresh, and CSRF cookies using
matching cookie paths and attributes. Send the CSRF header.

### Current user

`GET /auth/me`

Requires a valid access cookie. Bearer access tokens are still accepted for
non-browser clients.

## Conversation endpoints

All conversation endpoints accept the access cookie (or a legacy Bearer token).

- `GET /conversations?limit=50` — list the authenticated user's conversations.
- `GET /conversations/{conversation_id}` — load one conversation and messages.
- `DELETE /conversations/{conversation_id}` — delete one owned conversation.
  Cookie-authenticated DELETE requests require the CSRF header.

Conversation ownership is checked using both `user_id` and `project_id`.

## Chat endpoints

Browser requests use `credentials: "include"`.

- `POST /chat` — accepts cookie authentication; send `X-CSRF-Token`.
- `POST /chat-stream` — accepts cookie authentication; send `X-CSRF-Token`.
- JWT-authenticated user messages and assistant responses are persisted in
  PostgreSQL. Chat and Hybrid restore `ChatService` history. RAG includes recent
  conversation turns in the prompt.
- Legacy API-key clients remain supported and keep the existing in-memory chat
  behavior.

## Document endpoints

`/health`, `/upload`, `/documents`, and `/documents/{filename}` accept either
a valid user access cookie / Bearer token or the existing project API key.
Cookie-authenticated write requests require `X-CSRF-Token`. Documents remain
project-scoped and shared by authenticated users of that project; per-user
document ownership is not part of this version.

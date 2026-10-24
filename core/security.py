
import hmac
import os
import re
from collections.abc import Iterable


class APIKeyConfigurationError(RuntimeError):
    """Raised when project API keys are incorrectly configured."""


def api_key_env_name(project_id: str) -> str:
    """Convert a project ID into its environment variable name."""

    suffix = re.sub(
        r"[^A-Za-z0-9]",
        "_",
        project_id,
    ).strip("_").upper()

    if not suffix:
        raise APIKeyConfigurationError(
            "Invalid project ID for API key configuration."
        )

    return f"PROJECT_API_KEY_{suffix}"


def load_project_api_keys(
    project_ids: Iterable[str],
) -> dict[str, str]:
    """Load and validate API keys for all registered projects."""

    keys: dict[str, str] = {}

    for project_id in project_ids:
        env_name = api_key_env_name(project_id)
        api_key = os.getenv(env_name, "").strip()

        if not api_key:
            raise APIKeyConfigurationError(
                f"API key is not configured for project "
                f"'{project_id}'."
            )

        if len(api_key) < 32:
            raise APIKeyConfigurationError(
                f"API key for project '{project_id}' "
                "must contain at least 32 characters."
            )

        keys[project_id] = api_key

    if len(set(keys.values())) != len(keys):
        raise APIKeyConfigurationError(
            "Every project must have a unique API key."
        )

    return keys


def resolve_project_for_api_key(
    api_key: str | None,
    project_ids: Iterable[str],
) -> str | None:
    """Resolve a valid API key to its owning project."""

    keys = load_project_api_keys(project_ids)

    if not api_key:
        return None

    provided_key = api_key.encode("utf-8")

    for project_id, expected_key in keys.items():
        if hmac.compare_digest(
            provided_key,
            expected_key.encode("utf-8"),
        ):
            return project_id

    return None

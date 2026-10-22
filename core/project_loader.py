
import json
from pathlib import Path

from core.project_config import ProjectConfig


def load_project_settings(
    path: str | Path = "config/projects.json",
) -> tuple[str, list[ProjectConfig], dict[str, str | None]]:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Project configuration not found: {config_path}"
        )

    try:
        data = json.loads(
            config_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid project configuration JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object.")

    projects_data = data.get("projects")
    default_project = data.get("default_project")

    if not isinstance(projects_data, list) or not projects_data:
        raise ValueError(
            "At least one project must be configured."
        )

    if not isinstance(default_project, str) or not default_project.strip():
        raise ValueError(
            "default_project must be a non-empty string."
        )

    configs: list[ProjectConfig] = []
    prompts: dict[str, str | None] = {}
    registered_names: set[str] = set()

    for item in projects_data:
        if not isinstance(item, dict):
            raise ValueError("Each project must be a JSON object.")

        name = item.get("name")

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Every project must have a non-empty name."
            )

        name = name.strip()

        if name in registered_names:
            raise ValueError(
                f"Duplicate project name: {name}"
            )

        chat_enabled = item.get("chat_enabled")
        rag_enabled = item.get("rag_enabled")
        system_prompt = item.get("system_prompt")

        if type(chat_enabled) is not bool:
            raise ValueError(
                f"chat_enabled must be boolean for {name}."
            )

        if type(rag_enabled) is not bool:
            raise ValueError(
                f"rag_enabled must be boolean for {name}."
            )

        if system_prompt is not None and not isinstance(
            system_prompt, str
        ):
            raise ValueError(
                f"system_prompt must be a string for {name}."
            )

        configs.append(
            ProjectConfig(
                name=name,
                chat_enabled=chat_enabled,
                rag_enabled=rag_enabled,
            )
        )

        prompts[name] = system_prompt
        registered_names.add(name)

    if default_project not in registered_names:
        raise ValueError(
            f"Default project is not registered: {default_project}"
        )

    return default_project, configs, prompts

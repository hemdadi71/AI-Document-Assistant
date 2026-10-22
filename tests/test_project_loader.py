
import json

import pytest

from core.project_loader import load_project_settings


def test_load_project_settings(tmp_path):
    config_file = tmp_path / "projects.json"

    config_file.write_text(
        json.dumps({
            "default_project": "assistant",
            "projects": [
                {
                    "name": "assistant",
                    "chat_enabled": True,
                    "rag_enabled": True,
                    "system_prompt": "Be helpful.",
                },
                {
                    "name": "documents",
                    "chat_enabled": False,
                    "rag_enabled": True,
                },
            ],
        }),
        encoding="utf-8",
    )

    default_id, configs, prompts = load_project_settings(
        config_file
    )

    assert default_id == "assistant"
    assert len(configs) == 2

    assert configs[0].chat_enabled is True
    assert configs[0].rag_enabled is True

    assert configs[1].chat_enabled is False
    assert configs[1].rag_enabled is True

    assert prompts["assistant"] == "Be helpful."
    assert prompts["documents"] is None


def test_reject_duplicate_project_names(tmp_path):
    config_file = tmp_path / "projects.json"

    config_file.write_text(
        json.dumps({
            "default_project": "assistant",
            "projects": [
                {
                    "name": "assistant",
                    "chat_enabled": True,
                    "rag_enabled": False,
                },
                {
                    "name": "assistant",
                    "chat_enabled": False,
                    "rag_enabled": True,
                },
            ],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_project_settings(config_file)


def test_reject_unknown_default_project(tmp_path):
    config_file = tmp_path / "projects.json"

    config_file.write_text(
        json.dumps({
            "default_project": "missing",
            "projects": [
                {
                    "name": "assistant",
                    "chat_enabled": True,
                    "rag_enabled": True,
                }
            ],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Default project"):
        load_project_settings(config_file)


from dataclasses import dataclass
from typing import Literal


FeatureName = Literal["chat", "rag"]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    chat_enabled: bool = False
    rag_enabled: bool = False

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("Project name cannot be empty.")

        if not self.chat_enabled and not self.rag_enabled:
            raise ValueError(
                "At least one feature must be enabled."
            )

    def is_enabled(self, feature: FeatureName) -> bool:
        if feature == "chat":
            return self.chat_enabled

        if feature == "rag":
            return self.rag_enabled

        raise ValueError(f"Unknown feature: {feature}")

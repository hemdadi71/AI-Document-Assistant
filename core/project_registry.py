
from collections.abc import Iterable

from core.project_config import ProjectConfig


class ProjectRegistry:

    def __init__(
        self,
        projects: Iterable[ProjectConfig] = (),
    ):
        self._projects: dict[str, ProjectConfig] = {}

        for project in projects:
            self.register(project)

    def register(self, project: ProjectConfig) -> None:
        project_id = project.name

        if project_id in self._projects:
            raise ValueError(
                f"Project already registered: {project_id}"
            )

        self._projects[project_id] = project

    def get(self, project_id: str) -> ProjectConfig:
        try:
            return self._projects[project_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown project: {project_id}"
            ) from exc

    def contains(self, project_id: str) -> bool:
        return project_id in self._projects

    def list_projects(self) -> list[ProjectConfig]:
        return list(self._projects.values())

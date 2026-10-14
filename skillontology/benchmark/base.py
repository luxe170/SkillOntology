from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from skillontology.schema import QueryDecomposition


@dataclass(frozen=True)
class Task:
    instruction: str
    identity: dict[str, Any]


@dataclass(frozen=True)
class Observation:
    text: str
    controls: tuple[str, ...] = ()
    done: bool = False
    success: bool = False
    score: float = 0.0
    ticks: int | None = None


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def parse_task(self, task: Task, observation: Observation, ontology: Any) -> QueryDecomposition:
        raise NotImplementedError

    @abstractmethod
    def binding_roles(self, task: Task, query: QueryDecomposition) -> dict[str, str]:
        raise NotImplementedError


class BenchmarkEnvironment(ABC):
    def __init__(self, config: dict[str, Any], backend: Any = None):
        self.config = config
        self.backend = backend

    @abstractmethod
    def reset_task(self, task: Task) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def step_action(self, action: str) -> Observation:
        raise NotImplementedError

    def close(self) -> None:
        if self.backend is not None:
            for name in ('close', 'shutdown'):
                method = getattr(self.backend, name, None)
                if callable(method):
                    method()
                    break

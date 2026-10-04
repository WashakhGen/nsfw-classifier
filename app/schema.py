import dataclasses
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel

from app.settings import SETTINGS


@dataclasses.dataclass(frozen=True)
class NSFWResult:
    neutral: float
    low: float
    medium: float
    high: float

    def at_least(self, level: Literal["low", "medium", "high"]) -> float:
        "Probability the image is this level or worse (Freepik's cumulative score)."
        match level:
            case "high":
                return self.high
            case "medium":
                return self.medium + self.high
            case "low":
                return self.low + self.medium + self.high

    @property
    def nsfw_score(self) -> float:
        "0->1 Higher is less safe, at the configured level."
        return self.at_least(SETTINGS.NSFW_LEVEL)

    @property
    def is_nsfw(self) -> bool:
        return self.nsfw_score >= SETTINGS.NSFW_THRESHOLD

    def _columns(self) -> Sequence[tuple[str, bool | float]]:
        return (
            ("is_nsfw", self.is_nsfw),
            ("score", self.nsfw_score),
            *dataclasses.asdict(self).items(),
        )

    def tsv(self) -> str:
        columns = self._columns()
        return (
            "\t".join(name for name, _ in columns)
            + "\n"
            + "\t".join(str(round(value, 3)) if isinstance(value, float) else str(value) for _, value in columns)
        )


class Classification(BaseModel):
    nsfw_score: float
    neutral: float
    low: float
    medium: float
    high: float


class NSFWResponse(BaseModel):
    task_id: str
    is_nsfw: bool
    classification: Classification


class ModelStatus(StrEnum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class InferenceError(RuntimeError):
    "Model failed during inference."

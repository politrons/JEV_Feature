from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Intent(StrEnum):
    TRANSLATE_ES_EN = "TRANSLATE_ES_EN"
    ARITHMETIC = "ARITHMETIC"
    WEEKDAY = "WEEKDAY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=4000)


class Decision(BaseModel):
    intent: Intent
    confidence: Probability
    probabilities: dict[Intent, Probability]
    model: str
    latency_ms: float = Field(ge=0)


class AskResponse(BaseModel):
    status: Literal["completed", "needs_clarification", "out_of_scope"]
    decision: Decision
    agent: Intent | None = None
    result: str | float | None = None
    message: str | None = None

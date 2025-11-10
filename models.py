from typing import Any, Optional
from pydantic import BaseModel, Field


class Step(BaseModel):
    description: str = Field(..., min_length=5, description="The step executed.")
    tool: Optional[str] = Field(None, description="The tool used in this step, if any.")
    args: Optional[dict[str, Any]] = None


class StepOutput(BaseModel):
    status: str
    output: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    tool_used: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None

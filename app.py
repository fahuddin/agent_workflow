# app.py
import os, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = FastAPI(title="Build Workflow API", version="1.0.0")

class BuildRequest(BaseModel):
    goal: str = Field(..., min_length=5, description="The build goal to be executed.")

class Step(BaseModel):
    description: str = Field(..., min_length=5, description="The step executed.")
    tool: Optional[str] = Field(None, description="The tool used in this step, if any.")
    args: Optional[dict[str]] = Field(None, description="Arguments for the tool, if any.")



class Task(BaseModel):
    title: str = Field(..., min_length=5, description="The title of the task.")
    steps: list[Step] = Field(..., description="List of steps executed in the task.")

class BuildRequestResponse(BaseModel):
    goal: str
    workflow: list[Task]

class StepOutput(BaseModel):
    status: str 
    output: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    tool_used: Optional[str] = None
    tool_args: Optional[dict[str]] = None

class TaskResult(BaseModel):
    title: str
    step_results: list[StepOutput]


class ExecutionResult(BaseModel):
    goal: str
    results: list[TaskResult]
    status: str  # DONE | FAILED

@app.get("/ping")
async def ping():
    return {"ok": True}

SYSTEM = (
    'You are a workflow planner. '
    'Return STRICT JSON ONLY: {"goal": string, "workflow": '
    '[{"title": string, "steps": [{"description": string}]}]}'
)

@app.post("/build-workflow", response_model=BuildRequestResponse)
async def build_workflow(request: BuildRequest):
    user = f"Build a workflow to achieve the goal: {request.goal}. " \
           f"Create 3-6 tasks; each task has 2-4 steps."

    try:
        # Prefer JSON mode; many providers support this
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user}
            ],
            response_format={"type": "json_object"},
        )
    except Exception as e:
        # Fallback without JSON mode if needed
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM + "\nReturn STRICT JSON only. No extra text."},
                    {"role": "user", "content": user}
                ],
                temperature=0.0
            )
        except Exception as ee:
            raise HTTPException(status_code=500, detail=f"LLM call failed: {type(ee).__name__}: {ee}") from ee

    raw = resp.choices[0].message.content.strip()  # JSON string
    try:
        data = json.loads(raw)             # Python dict
    except Exception:
        raise HTTPException(status_code=500, detail=f"Model did not return JSON. First 200 chars: {raw[:200]}")

    # Validate & return
    try:
        return BuildRequestResponse.model_validate(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}")

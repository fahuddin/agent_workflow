# app.py
import os
import json
import re
import time
import asyncio
import ast
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
import httpx
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from openai import AsyncOpenAI
from db import init_db, start_run, finish_run, create_task_row, insert_step_row

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

app = FastAPI(title="Build Workflow API", version="1.0.0")

init_db()

class BuildRequest(BaseModel):
    goal: str = Field(..., min_length=5, description="The build goal to be executed.")

from models import Step, StepOutput


class Task(BaseModel):
    title: str = Field(..., min_length=5, description="The title of the task.")
    steps: List[Step] = Field(..., description="List of steps executed in the task.")

class BuildRequestResponse(BaseModel):
    goal: str
    workflow: List[Task]

class StepOutput(BaseModel):
    status: str 
    output: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    tool_used: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None

class TaskResult(BaseModel):
    title: str
    step_results: List[StepOutput]


class ExecutionResult(BaseModel):
    goal: str
    results: List[TaskResult]
    status: str  # DONE | FAILED


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4})\b")

def redact_pii(text: str) -> str:
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text

def coerce_to_str(value: Any) -> BuildRequestResponse:
    try:
        data: Any = json.loads(value) if isinstance(value, str) else value
        try:
            return BuildRequestResponse.model_validate(data)
        except ValidationError as ve:
            # Try coercing all fields to strings
            def coerce(value: Any) -> Any:
                if isinstance(value, dict):
                    return {k: coerce(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [coerce(v) for v in value]
                else:
                    return str(value)
            coerced_data = coerce(data)
            return BuildRequestResponse.model_validate(coerced_data)
    except ValidationError as ve:
            raise HTTPException(status_code=500, detail=f"Validation failed: {ve}") from ve
async def tool_http_get(url: str, timeout_s: float = 15.0) -> Dict[str, Any]:
    """Fetch JSON or text from a URL. Keep it minimal & capped."""
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url)
        r.raise_for_status()
        try:
            return {"type": "json", "data": r.json()}
        except Exception:
            return {"type": "text", "data": r.text[:5000]}


# Safe python eval: AST-checked + name whitelist
SAFE_NAMES = {"abs": abs, "min": min, "max": max, "sum": sum, "len": len}


def _is_ast_safe(tree: ast.AST) -> bool:
    """
    Validate AST node types and ensure names are allowed.
    - disallow Attribute (e.g., os.system)
    - ensure ast.Name nodes are whitelisted or are constants (True/False/None)
    """
    allowed_node_types = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Str,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.Constant,
        ast.Compare,
        ast.BoolOp,
        ast.Name,
        ast.Call,
        ast.Load,
        ast.keyword,
        ast.Subscript,
        ast.Slice,
        ast.Index,
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            return False
        if not isinstance(node, allowed_node_types):
            return False
        if isinstance(node, ast.Name):
            # allow literal names + SAFE_NAMES
            if node.id not in SAFE_NAMES and node.id not in ("True", "False", "None"):
                return False
        if isinstance(node, ast.Call):
            # ensure function being called is a Name (not an attribute) and allowed
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in SAFE_NAMES:
                    return False
            else:
                return False
    return True


async def tool_python_eval(expr: str) -> Dict[str, Any]:
    """
    Evaluate a small Python expression safely (very limited).
    Returns {"result": <value>} or raises ValueError.
    """
    tree = ast.parse(expr, mode="eval")
    if not _is_ast_safe(tree):
        raise ValueError("Expression not allowed")

    code_obj = compile(tree, "<expr>", "eval")
    # Execute with NO builtins and a restricted local mapping (SAFE_NAMES)
    result = eval(code_obj, {"__builtins__": {}}, SAFE_NAMES)
    return {"result": result}

@app.get("/ping")
async def ping():
    return {"ok": True}

SYSTEM = (
    'You are a workflow planner. Return STRICT JSON ONLY: '
    '{"goal": string, "workflow": ['
    '{"title": string, '
    '"steps": ['
      '{"description": string, "tool": string (optional), "args": object (optional)}'
    ']}]}'
)


async def run_step(step: Step) -> StepOutput:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tool = step.tool or None
    try:
        if tool == "http_get":
            url = step.args.get("url") if step.args else None
            if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                raise ValueError("Invalid URL provided.")
            result = await tool_http_get(url)
            output = f"Fetched {result['type']} data."
        elif tool  == "python_eval":
            expr = step.args.get("expr") if isinstance(step.args, dict) else step.args
            result = await tool_python_eval(expr)
            output = f"Evaluated expression with result: {result['result']}"
        else:
            output = "No tool executed."
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return StepOutput(
            status="SUCCESS",
            output=output,
            started_at=started_at,
            finished_at=finished_at,
            tool_used=step.tool,
            tool_args=step.args or {}
        )
    except Exception as e:
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return StepOutput(
            status="FAILED",
            error=redact_pii(str(e)),
            started_at=started_at,
            finished_at=finished_at,
            tool_used=step.tool,
            tool_args=step.args
        )
    


@app.post("/step-execute", response_model=StepOutput)
async def step_execute(step: Step):
    try:
        step = Step.model_validate(step)
        return await run_step(step)
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid step data: {ve}") from ve

@app.post("/task-execute", response_model=ExecutionResult)
async def task_execute(plan: BuildRequestResponse):
    run_id = start_run(plan)
    """
    Execute all tasks in the provided workflow plan sequentially.
    Args:
        plan (BuildRequestResponse): The workflow plan containing tasks and steps.
    Returns:
        ExecutionResult: The result of executing all tasks, including step outputs and overall status.
    """
    task_results = []
    for task_idx, task in enumerate(plan.workflow):
        task_id = create_task_row(run_id, task.title, task_idx)
        step_results = []
        for step_idx, step in enumerate(task.steps):
            result = await run_step(step)
            insert_step_row(task_id, step_idx, step, result)
            step_results.append(result)
            if result.status != "SUCCESS":
                break  # Stop on first failure
        task_results.append(TaskResult(title=task.title, step_results=step_results))
    overall_status = "DONE"
    for task_result in task_results:
        for step_result in task_result.step_results:
            if step_result.status != "SUCCESS":
                overall_status = "FAILED"
                break
        if overall_status == "FAILED":
            break
    return ExecutionResult(goal=plan.goal, results=task_results, status=overall_status)



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

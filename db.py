import json, time
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, create_engine, Session, select
from app import Step, StepOutput

DB_URL = "sqlite:///./runs.db"
engine = create_engine(DB_URL, echo=False)

class Run(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    goal: str
    status: str
    started_at: float
    finished_at: float | None = None

class TaskRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    title: str
    order: int

class StepRow(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    task_id: int = Field(index=True)
    idx: int
    description: str
    tool: str | None = None
    args_json: str
    status: str
    output_json: str | None = None
    error: str | None = None
    started_at: float
    finished_at: float

def init_db():
    SQLModel.metadata.create_all(engine)

def _now(): return time.time()

def _parse_ts(ts: str | None) -> float:
    if not ts: return _now()
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except Exception: return _now()

def _json_dumps(obj):
    try: 
        return json.dumps(obj, ensure_ascii=False)
    except: 
        return json.dumps(str(obj))

def _try_load(js: str | None):
    if not js: 
        return None
    try: 
        return json.loads(js)
    except: return js

def start_run(goal: str) -> int:
    with Session(engine) as s:
        r = Run(goal=goal, status="RUNNING", started_at=_now())
        s.add(r); s.commit(); s.refresh(r)
        return r.id

def finish_run(run_id: int, status: str):
    with Session(engine) as s:
        run = s.get(Run, run_id)
        if run:
            run.status = status
            run.finished_at = _now()
            s.add(run); s.commit()

def create_task_row(run_id: int, title: str, order: int) -> int:
    with Session(engine) as s:
        t = TaskRow(run_id=run_id, title=title, order=order)
        s.add(t); s.commit(); s.refresh(t)
        return t.id

def insert_step_row(task_id: int, idx: int, step: Step, result: StepOutput):
    with Session(engine) as s:
        row = StepRow(
            task_id=task_id,
            idx=idx,
            description=step.description,
            tool=step.tool,
            args_json=_json_dumps(step.args),
            status=result.status,
            output_json=_json_dumps(result.output),
            error=result.error,
            started_at=_parse_ts(result.started_at),
            finished_at=_parse_ts(result.finished_at),
        )
        s.add(row); s.commit(); s.refresh(row)
        return row.id

def get_run_tree(run_id: int):
    with Session(engine) as s:
        run = s.get(Run, run_id)
        if not run: return {}
        tasks = s.exec(select(TaskRow).where(TaskRow.run_id == run_id).order_by(TaskRow.order)).all()
        results = []
        for t in tasks:
            steps = s.exec(select(StepRow).where(StepRow.task_id == t.id).order_by(StepRow.idx)).all()
            results.append({
                "title": t.title,
                "step_results": [{
                    "status": st.status,
                    "output": _try_load(st.output_json),
                    "error": st.error,
                    "started_at": st.started_at,
                    "finished_at": st.finished_at,
                    "tool_used": st.tool,
                    "tool_args": _try_load(st.args_json)
                } for st in steps]
            })
        return {
            "id": run.id,
            "goal": run.goal,
            "status": run.status,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "results": results
        }

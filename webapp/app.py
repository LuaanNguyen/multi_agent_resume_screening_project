"""FastAPI dashboard for running and viewing the local resume pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


ROOT_DIR = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CSV_FILE = "archive/Resume/Resume.csv"
DEFAULT_PDF_DIR = "archive/data/data"

REPORT_FILES = {
    "evaluation": "evaluation_report.json",
    "association": "association_rules.json",
    "cluster": "cluster_report.json",
    "validation": "validation_report.json",
}

PIPELINE_COMMANDS = {
    "process-csv",
    "process-pdf",
    "train",
    "evaluate",
    "mine",
    "cluster-csv",
    "cluster-pdf",
    "validate",
}


def utc_now() -> str:
    """Return a stable UTC timestamp for local job records."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json_file(path: Path) -> tuple[Any | None, str | None]:
    """Read a JSON file and return data plus a human-readable error if needed."""
    if not path.exists():
        return None, "not available yet"

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    except OSError as exc:
        return None, str(exc)


def summarize_reports(reports: dict[str, Any]) -> dict[str, Any]:
    """Create compact display metrics from the pipeline report JSON files."""
    evaluation = reports.get("evaluation") or {}
    comparison = evaluation.get("model_comparison") or {}
    baseline = comparison.get("baseline") or {}
    proposed = comparison.get("proposed") or evaluation.get("classification") or {}

    baseline_accuracy = baseline.get("accuracy")
    proposed_accuracy = proposed.get("accuracy")
    proposed_underperforms = (
        isinstance(baseline_accuracy, (int, float))
        and isinstance(proposed_accuracy, (int, float))
        and proposed_accuracy < baseline_accuracy
    )

    association = reports.get("association")
    association_count = len(association) if isinstance(association, list) else None

    cluster = reports.get("cluster") or {}
    validation = (reports.get("validation") or {}).get("extraction_validation") or {}

    return {
        "baseline": baseline,
        "proposed": proposed,
        "proposed_underperforms": proposed_underperforms,
        "association_rule_count": association_count,
        "cluster": cluster,
        "validation": validation,
    }


def load_reports(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Load all known reports from an output directory."""
    base_dir = ROOT_DIR / Path(output_dir)
    reports_dir = base_dir / "reports"
    reports: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for key, filename in REPORT_FILES.items():
        data, error = read_json_file(reports_dir / filename)
        reports[key] = data
        if error:
            errors[key] = error

    return {
        "output_dir": str(output_dir),
        "reports_dir": str(reports_dir),
        "reports": reports,
        "errors": errors,
        "summary": summarize_reports(reports),
    }


def get_local_status(
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    csv_file: str = DEFAULT_CSV_FILE,
    pdf_dir: str = DEFAULT_PDF_DIR,
) -> dict[str, Any]:
    """Return local file availability for the dashboard."""
    output_path = ROOT_DIR / output_dir
    reports_path = output_path / "reports"

    return {
        "config_exists": (ROOT_DIR / config_path).exists(),
        "csv_exists": (ROOT_DIR / csv_file).exists(),
        "pdf_dir_exists": (ROOT_DIR / pdf_dir).exists(),
        "models_exist": (output_path / "models" / "classifier.pkl").exists(),
        "reports_dir_exists": reports_path.exists(),
        "report_count": len(list(reports_path.glob("*.json"))) if reports_path.exists() else 0,
        "csv_structured_count": len(list((output_path / "csv_structured").glob("*.json")))
        if (output_path / "csv_structured").exists()
        else 0,
        "pdf_structured_count": len(list((output_path / "pdf_structured").glob("*.json")))
        if (output_path / "pdf_structured").exists()
        else 0,
    }


def build_pipeline_argv(
    command: str,
    *,
    python_executable: str,
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    csv_file: str = DEFAULT_CSV_FILE,
    pdf_dir: str = DEFAULT_PDF_DIR,
    n_clusters: str | None = None,
) -> list[str]:
    """Build the exact CLI command the dashboard should run."""
    if command not in PIPELINE_COMMANDS:
        raise ValueError(f"Unsupported dashboard command: {command}")

    argv = [
        python_executable,
        "main.py",
        "--config",
        config_path,
        "--output-dir",
        output_dir,
        "--log-level",
        "INFO",
    ]

    if command == "process-csv":
        argv.extend(["process-csv", "--csv-file", csv_file])
    elif command == "process-pdf":
        argv.extend(["process-pdf", "--pdf-dir", pdf_dir])
    elif command == "train":
        argv.extend(["train", "--csv-file", csv_file])
    elif command == "evaluate":
        argv.extend(["evaluate", "--csv-file", csv_file])
    elif command == "mine":
        argv.extend(["mine", "--csv-file", csv_file])
    elif command == "cluster-csv":
        argv.extend(["cluster", "--source", "csv", "--csv-file", csv_file])
    elif command == "cluster-pdf":
        argv.extend(["cluster", "--source", "pdf", "--pdf-dir", pdf_dir])
    elif command == "validate":
        argv.extend(["validate", "--csv-file", csv_file, "--pdf-dir", pdf_dir])

    if command.startswith("cluster-") and n_clusters:
        argv.extend(["--n-clusters", str(n_clusters)])

    return argv


@dataclass
class JobRecord:
    """In-memory record for one local pipeline run."""

    id: str
    command: str
    argv: list[str]
    status: str = "running"
    start_time: str = field(default_factory=utc_now)
    end_time: str | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "argv": self.argv,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
        }


class JobManager:
    """Run one local pipeline subprocess at a time."""

    def __init__(
        self,
        *,
        root_dir: Path = ROOT_DIR,
        python_executable: str = sys.executable,
    ) -> None:
        self.root_dir = root_dir
        self.python_executable = python_executable
        self._jobs: dict[str, JobRecord] = {}
        self._active_job_id: str | None = None
        self._lock = threading.Lock()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in reversed(list(self._jobs.values()))]

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start_job(
        self,
        command: str,
        *,
        config_path: str = DEFAULT_CONFIG_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        csv_file: str = DEFAULT_CSV_FILE,
        pdf_dir: str = DEFAULT_PDF_DIR,
        n_clusters: str | None = None,
    ) -> JobRecord:
        argv = build_pipeline_argv(
            command,
            python_executable=self.python_executable,
            config_path=config_path,
            output_dir=output_dir,
            csv_file=csv_file,
            pdf_dir=pdf_dir,
            n_clusters=n_clusters,
        )

        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id)
                if active and active.status == "running":
                    raise RuntimeError(
                        f"Job {active.id} is still running. Wait for it to finish."
                    )

            job = JobRecord(id=str(uuid.uuid4()), command=command, argv=argv)
            self._jobs[job.id] = job
            self._active_job_id = job.id

        thread = threading.Thread(target=self._run_job, args=(job.id,), daemon=True)
        thread.start()
        return job

    def _run_subprocess(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv,
            cwd=self.root_dir,
            capture_output=True,
            text=True,
            check=False,
        )

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return

        try:
            result = self._run_subprocess(job.argv)
            with self._lock:
                job.return_code = result.returncode
                job.stdout = result.stdout or ""
                job.stderr = result.stderr or ""
                job.status = "succeeded" if result.returncode == 0 else "failed"
                job.end_time = utc_now()
                if self._active_job_id == job_id:
                    self._active_job_id = None
        except Exception as exc:  # pragma: no cover - defensive local-server guard
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.end_time = utc_now()
                if self._active_job_id == job_id:
                    self._active_job_id = None


def create_app(job_manager: JobManager | None = None) -> FastAPI:
    """Create the local dashboard app."""
    manager = job_manager or JobManager()
    templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

    app = FastAPI(title="Resume Screening Dashboard")
    app.mount(
        "/static",
        StaticFiles(directory=str(WEBAPP_DIR / "static")),
        name="static",
    )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(
        request: Request,
        config_path: str = DEFAULT_CONFIG_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        csv_file: str = DEFAULT_CSV_FILE,
        pdf_dir: str = DEFAULT_PDF_DIR,
    ):
        reports = load_reports(output_dir)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "defaults": {
                    "config_path": config_path,
                    "output_dir": output_dir,
                    "csv_file": csv_file,
                    "pdf_dir": pdf_dir,
                },
                "jobs": manager.list_jobs(),
                "reports": reports,
                "request": request,
                "status": get_local_status(config_path, output_dir, csv_file, pdf_dir),
            },
        )

    @app.get("/reports", response_class=HTMLResponse)
    def reports_page(
        request: Request,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ):
        return templates.TemplateResponse(
            request,
            "reports.html",
            {
                "output_dir": output_dir,
                "reports": load_reports(output_dir),
                "request": request,
            },
        )

    @app.get("/api/reports")
    def api_reports(output_dir: str = DEFAULT_OUTPUT_DIR):
        return load_reports(output_dir)

    @app.post("/jobs/{command}")
    def create_job_route(
        command: str,
        config_path: str = Form(DEFAULT_CONFIG_PATH),
        output_dir: str = Form(DEFAULT_OUTPUT_DIR),
        csv_file: str = Form(DEFAULT_CSV_FILE),
        pdf_dir: str = Form(DEFAULT_PDF_DIR),
        n_clusters: str = Form(""),
    ):
        try:
            job = manager.start_job(
                command,
                config_path=config_path,
                output_dir=output_dir,
                csv_file=csv_file,
                pdf_dir=pdf_dir,
                n_clusters=n_clusters.strip() or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return job.to_dict()

    @app.get("/jobs/{job_id}")
    def get_job_route(job_id: str):
        job = manager.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.to_dict()

    app.state.job_manager = manager
    return app


app = create_app()

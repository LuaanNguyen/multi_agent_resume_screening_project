import json
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from webapp.app import JobManager, build_pipeline_argv, create_app, load_reports


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_reports_handles_missing_reports(tmp_path):
    result = load_reports(tmp_path / "output")

    assert result["reports"]["evaluation"] is None
    assert result["errors"]["evaluation"] == "not available yet"
    assert result["summary"]["association_rule_count"] is None


def test_load_reports_parses_summary_metrics(tmp_path):
    output_dir = tmp_path / "output"
    reports_dir = output_dir / "reports"

    write_json(
        reports_dir / "evaluation_report.json",
        {
            "classification": {"accuracy": 0.4, "macro_f1": 0.3},
            "model_comparison": {
                "baseline": {"accuracy": 0.6, "macro_f1": 0.5},
                "proposed": {"accuracy": 0.4, "macro_f1": 0.3},
            },
            "fairness": {},
        },
    )
    write_json(reports_dir / "association_rules.json", [{"lift": 1.2}])
    write_json(
        reports_dir / "cluster_report.json",
        {"source": "csv", "sample_count": 5, "actual_cluster_count": 2},
    )
    write_json(
        reports_dir / "validation_report.json",
        {"extraction_validation": {"extraction_accuracy": 0.87}},
    )

    result = load_reports(output_dir)

    assert result["summary"]["baseline"]["accuracy"] == 0.6
    assert result["summary"]["proposed"]["accuracy"] == 0.4
    assert result["summary"]["proposed_underperforms"] is True
    assert result["summary"]["association_rule_count"] == 1
    assert result["summary"]["cluster"]["source"] == "csv"
    assert result["summary"]["validation"]["extraction_accuracy"] == 0.87


def test_dashboard_pages_return_200(tmp_path):
    manager = JobManager(root_dir=tmp_path, python_executable=sys.executable)
    client = TestClient(create_app(manager))

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Recommended flow" in dashboard.text
    assert "run-progress" in dashboard.text
    assert "loader-ring" in dashboard.text
    assert "loader-bars" in dashboard.text
    assert "Process CSV" in dashboard.text
    assert "Validate" in dashboard.text

    assert client.get("/reports").status_code == 200
    assert client.get("/api/reports").status_code == 200


def test_build_pipeline_argv_matches_cli_shape():
    argv = build_pipeline_argv(
        "cluster-csv",
        python_executable="python",
        config_path="config/config.yaml",
        output_dir="output",
        csv_file="archive/Resume/Resume.csv",
        pdf_dir="archive/data/data",
        n_clusters="4",
    )

    assert argv == [
        "python",
        "main.py",
        "--config",
        "config/config.yaml",
        "--output-dir",
        "output",
        "--log-level",
        "INFO",
        "cluster",
        "--source",
        "csv",
        "--csv-file",
        "archive/Resume/Resume.csv",
        "--n-clusters",
        "4",
    ]


def test_job_creation_runs_mocked_subprocess(tmp_path, monkeypatch):
    manager = JobManager(root_dir=tmp_path, python_executable=sys.executable)
    client = TestClient(create_app(manager))

    def fake_run(argv):
        return subprocess.CompletedProcess(argv, 0, "finished", "")

    monkeypatch.setattr(manager, "_run_subprocess", fake_run)

    response = client.post("/jobs/process-csv", data={})
    assert response.status_code == 200

    job_id = response.json()["id"]
    for _ in range(20):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.02)

    assert job["status"] == "succeeded"
    assert job["return_code"] == 0
    assert job["stdout"] == "finished"


def test_one_job_at_a_time_rejects_second_job(tmp_path, monkeypatch):
    manager = JobManager(root_dir=tmp_path, python_executable=sys.executable)
    client = TestClient(create_app(manager))

    def slow_run(argv):
        time.sleep(0.2)
        return subprocess.CompletedProcess(argv, 0, "done", "")

    monkeypatch.setattr(manager, "_run_subprocess", slow_run)

    first = client.post("/jobs/process-csv", data={})
    assert first.status_code == 200

    second = client.post("/jobs/train", data={})
    assert second.status_code == 409

    job_id = first.json()["id"]
    for _ in range(30):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.02)

    assert job["status"] == "succeeded"

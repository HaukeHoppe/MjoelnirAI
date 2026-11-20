"""Unit tests for data/mlflow_log.py and the run summary of data/eval_chains.py, against a fake MLflow server."""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "data"))
import mlflow_log
from eval_chains import summarize


@pytest.fixture
def server():
    # Fake-MLflow auf einem freien Port: kennt das Experiment noch nicht und merkt sich jede Anfrage.
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append(("GET", self.path, None))
            self._answer(404, {"error_code": "RESOURCE_DOES_NOT_EXIST"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, body))
            replies = {
                "/api/2.0/mlflow/experiments/create": {"experiment_id": "7"},
                "/api/2.0/mlflow/runs/create": {
                    "run": {"info": {"run_id": "r1", "artifact_uri": "mlflow-artifacts:/7/r1/artifacts"}}
                },
            }
            self._answer(200, replies.get(self.path, {}))

        def do_PUT(self):
            calls.append(("PUT", self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self._answer(200, {})

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", calls
    httpd.shutdown()


def test_log_run_creates_a_finished_run(server, tmp_path):
    base, calls = server
    artifact = tmp_path / "eval_x.json"
    artifact.write_text("[]")
    params = {f"P{i:03d}": i for i in range(130)}  # more than one batch of 100
    run_id = mlflow_log.log_run(
        "x", params, {"mean_steps": 4.4}, artifact=str(artifact), tags={"note": "n"}, start_ms=1000, base=base
    )
    assert run_id == "r1"
    paths = [(m, p) for m, p, _ in calls]
    assert ("POST", "/api/2.0/mlflow/experiments/create") in paths  # unknown experiment is created
    create = next(b for m, p, b in calls if p == "/api/2.0/mlflow/runs/create")
    assert create == {
        "experiment_id": "7",
        "run_name": "x",
        "start_time": 1000,
        "tags": [{"key": "note", "value": "n"}],
    }
    batches = [b for m, p, b in calls if p == "/api/2.0/mlflow/runs/log-batch"]
    assert [len(b.get("params", [])) for b in batches] == [100, 30, 0]
    assert batches[-1]["metrics"] == [{"key": "mean_steps", "value": 4.4, "timestamp": 1000, "step": 0}]
    assert ("PUT", "/api/2.0/mlflow-artifacts/artifacts/7/r1/artifacts/eval_x.json", b"[]") in calls
    update = calls[-1][2]
    assert calls[-1][1] == "/api/2.0/mlflow/runs/update"
    assert update["status"] == "FINISHED"


def row(steps, complete, refused, control=False, **extra):
    return {
        "steps": steps,
        "complete": complete,
        "refused": refused,
        "depth": 3,
        "seconds": 10,
        "control": control,
        **extra,
    }


def test_summarize_separates_controls():
    m = summarize([row(4, 2, False, sources=2), row(2, 1, False, sources=1), row(0, 0, True, control=True, sources=0)])
    assert m["questions"] == 2
    assert m["mean_steps"] == 3
    assert m["complete_rate"] == 0.5
    assert m["one_source_rate"] == 0.5
    assert m["controls_refused_rate"] == 1


def test_summarize_reads_old_files_without_sources_or_controls():
    m = summarize([{k: v for k, v in row(3, 2, False).items() if k != "control"}])
    assert m["mean_steps"] == 3
    assert "mean_sources" not in m
    assert "controls_refused_rate" not in m

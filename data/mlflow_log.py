"""
Logs one evaluation run to an MLflow tracking server over its REST API (no mlflow package needed).

The pipelines image stays free of the mlflow client and its OpenTelemetry pins (they would clash with
langfuse). Used by eval_chains.py and import_eval_history.py; the server URL comes from MLFLOW_TRACKING_URI.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

EXPERIMENT = "capital-chains"


def _call(base, method, path, body=None, data=None):
    # Schickt eine Anfrage an die MLflow-API; JSON hinein und heraus, oder rohe Bytes beim Artefakt-Upload.
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
    elif data is not None:
        headers = {"Content-Type": "application/octet-stream"}
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return json.loads(raw) if raw else {}


def experiment_id(base, name=EXPERIMENT):
    # Liefert die ID des Experiments und legt es beim ersten Lauf an.
    try:
        query = urllib.parse.urlencode({"experiment_name": name})
        return _call(base, "GET", f"/api/2.0/mlflow/experiments/get-by-name?{query}")["experiment"]["experiment_id"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return _call(base, "POST", "/api/2.0/mlflow/experiments/create", {"name": name})["experiment_id"]


def find_run(base, exp_id, run_name):
    # Sucht einen Lauf gleichen Namens, damit ein erneuter Import keine Duplikate erzeugt.
    found = _call(
        base,
        "POST",
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids": [exp_id], "filter": f"attributes.run_name = '{run_name}'", "max_results": 1},
    )
    return found["runs"][0]["info"]["run_id"] if found.get("runs") else None


def log_run(run_name, params, metrics, artifact=None, tags=None, start_ms=None, end_ms=None, base=None):
    # Legt einen abgeschlossenen Lauf an: Parameter (Einstellungen), Metriken (Ergebnis) und die JSON-Datei
    # mit allen Antworten als Artefakt. start_ms/end_ms erlauben historische Zeitpunkte beim Import.
    base = base or os.environ["MLFLOW_TRACKING_URI"]
    exp_id = experiment_id(base)
    start_ms = start_ms or int(time.time() * 1000)
    tag_list = [{"key": k, "value": str(v)} for k, v in (tags or {}).items()]
    run = _call(
        base,
        "POST",
        "/api/2.0/mlflow/runs/create",
        {"experiment_id": exp_id, "run_name": run_name, "start_time": start_ms, "tags": tag_list},
    )["run"]["info"]
    run_id = run["run_id"]
    # MLflow erlaubt Parameterwerte bis 6000 Zeichen und höchstens 100 Parameter pro Aufruf.
    items = [{"key": k, "value": str(v)[:6000]} for k, v in sorted(params.items())]
    for i in range(0, len(items), 100):
        _call(base, "POST", "/api/2.0/mlflow/runs/log-batch", {"run_id": run_id, "params": items[i : i + 100]})
    _call(
        base,
        "POST",
        "/api/2.0/mlflow/runs/log-batch",
        {
            "run_id": run_id,
            "metrics": [{"key": k, "value": float(v), "timestamp": start_ms, "step": 0} for k, v in metrics.items()],
        },
    )
    if artifact:
        # artifact_uri ist "mlflow-artifacts:/<exp>/<run>/artifacts"; der Server nimmt die Datei per PUT an.
        prefix = run["artifact_uri"].split(":", 1)[1].lstrip("/")
        name = urllib.parse.quote(os.path.basename(artifact))
        with open(artifact, "rb") as f:
            _call(base, "PUT", f"/api/2.0/mlflow-artifacts/artifacts/{prefix}/{name}", data=f.read())
    _call(
        base,
        "POST",
        "/api/2.0/mlflow/runs/update",
        {"run_id": run_id, "status": "FINISHED", "end_time": end_ms or int(time.time() * 1000)},
    )
    return run_id

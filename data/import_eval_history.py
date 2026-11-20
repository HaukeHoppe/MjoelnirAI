"""
Imports earlier eval_chains.py result files into MLflow, so old and new runs can be compared in one place.

Each file becomes one finished run with the scores as metrics and the file as artifact. Its time is the file's
modification time (end of the run) unless --notes gives one. Files already imported (same run name) are
skipped, so the import can be repeated. No OpenAI calls.

Run inside the capital pipelines container, e.g. on the VPS:
    docker exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 open-webui-pipelines-capital \\
        python /data/import_eval_history.py /data/eval_*.json --notes /data/eval_notes.json

--notes: optional JSON {"<file name>": {"note": "...", "commit": "...", "time": "2026-10-06T19:56:00+02:00"}}.
"""

import argparse
import glob
import json
import os
from datetime import datetime

from eval_chains import cited_sources, summarize
from mlflow_log import EXPERIMENT, experiment_id, find_run, log_run


def import_file(path, notes, base):
    # Liest eine Ergebnisdatei, berechnet die Kennzahlen und legt sie als Lauf an; liefert den Status als Text.
    name = os.path.splitext(os.path.basename(path))[0]
    if find_run(base, experiment_id(base), name):
        return f"skip   {name} (already in MLflow)"
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    # Dateien von vor der Quellenzählung: die zitierten Quellen aus der gespeicherten Antwort nachzählen.
    for r in results:
        r.setdefault("sources", cited_sources(r.get("answer", "")))
    info = notes.get(os.path.basename(path), {})
    end = datetime.fromisoformat(info["time"]).timestamp() if "time" in info else os.path.getmtime(path)
    end_ms = int(end * 1000)
    # Die Fragen liefen nacheinander: Beginn = Ende minus Summe der Antwortzeiten.
    start_ms = end_ms - sum(r.get("seconds", 0) for r in results) * 1000
    params = {"set": "validation" if any(r.get("control") for r in results) else "tuning", "file": name}
    if "commit" in info:
        params["commit"] = info["commit"]
    tags = {"source": "import"}
    if "note" in info:
        tags["mlflow.note.content"] = info["note"]
    log_run(name, params, summarize(results), artifact=path, tags=tags, start_ms=start_ms, end_ms=end_ms, base=base)
    return f"import {name} ({params['set']}, {len(results)} questions)"


def main():
    # Importiert alle angegebenen Dateien in zeitlicher Reihenfolge (Platzhalter werden auch hier aufgelöst).
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--notes")
    args = parser.parse_args()
    base = os.environ["MLFLOW_TRACKING_URI"]
    notes = {}
    if args.notes:
        with open(args.notes, encoding="utf-8") as f:
            notes = json.load(f)
    # The notes file matches /data/eval_*.json too; it is not a result file.
    skip = os.path.abspath(args.notes) if args.notes else None
    found = {p for pattern in args.files for p in glob.glob(pattern) if os.path.abspath(p) != skip}
    paths = sorted(found, key=os.path.getmtime)
    for path in paths:
        print(import_file(path, notes, base), flush=True)
    print(f"done: {len(paths)} file(s), experiment {EXPERIMENT} at {base}")


if __name__ == "__main__":
    main()

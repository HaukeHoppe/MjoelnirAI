"""
Measures how well the Capital Markets RAG pipeline explains multi-step cause -> effect chains.

Sends a fixed set of chain questions to the pipeline (through the pipelines server's OpenAI-compatible
API) and lets an LLM judge each answer: number of consecutive causal steps, whether the chain reaches
from the trigger to the final effect, and whether it was a refusal. Writes the answers and scores to
a JSON file, so runs before and after a change can be compared.

Run inside the capital pipelines container:
    docker exec open-webui-pipelines-capital python /data/eval_chains.py --out /data/eval_before.json

With MLFLOW_TRACKING_URI set, the run also goes to MLflow (valves as parameters, scores as metrics, the JSON
file as artifact), e.g. on the VPS:
    docker exec -e MLFLOW_TRACKING_URI=http://mlflow:5000 -e APP_VERSION=$(git describe --tags) \\
        open-webui-pipelines-capital python /data/eval_chains.py --out /data/eval_v0.9.2.json --note "..."
"""

import argparse
import json
import os
import re
import time
import urllib.request

from langchain_openai import ChatOpenAI
from mlflow_log import log_run
from pydantic import BaseModel, Field

QUESTIONS = [
    "Warum kann bei einem starken Aktiencrash auch der Goldpreis kurzfristig fallen?",
    "Wie können steigende Leitzinsen über Kursverluste bei Anleihen eine Bankenkrise auslösen?",
    "Was passiert mit Schwellenländern, wenn der US-Dollar stark aufwertet?",
    "Wie kann die Auflösung von Carry-Trades in Yen weltweit die Aktienmärkte unter Druck setzen?",
    "Warum fallen Wachstumsaktien stärker als Value-Aktien, wenn die Inflation überraschend steigt?",
    "Wie überträgt sich ein Ölpreisschock auf Inflation, Zinsen und Aktienbewertungen?",
    "Wie kann ein Anstieg der Volatilität über risikobasierte Anlagestrategien zu weiteren Verkäufen führen?",
    "Wie wirkt quantitative Lockerung auf Anleiherenditen, Aktienkurse und Wechselkurse?",
    "Warum können steigende Kreditaufschläge von Unternehmensanleihen eine Rezession ankündigen?",
    "Wie kann eine inverse Zinsstrukturkurve über die Kreditvergabe der Banken die Konjunktur bremsen?",
    "Was passiert bei einer Liquiditätskrise am Geldmarkt mit Anleihen und Aktien?",
    "Wie entsteht ein Short Squeeze und welche Folgen hat er für den Aktienkurs?",
    "Wie wirkt eine Abwertung des Euro auf exportorientierte Unternehmen und auf die Inflation im Euroraum?",
    "Warum kann Deflation trotz sinkender Preise für Unternehmen und Kreditnehmer schädlich sein?",
    "Wie kann eine kreditfinanzierte Spekulationsblase entstehen und wie läuft ihr Platzen ab?",
]

# Held-out set: written after the pipeline was tuned on QUESTIONS and never used for tuning. Angles that
# differ from the chain topics of build_public_kb.py, many needing links from several sources.
VALIDATION = [
    "Was passiert mit Immobilienaktien und REITs, wenn die Realzinsen deutlich steigen?",
    "Wie kann ein starker Ölpreisanstieg Fluggesellschaften und Ölkonzerne an der Börse gegensätzlich treffen?",
    "Warum fallen Unternehmensanleihen schlechter Bonität in einer Rezession meist stärker als Staatsanleihen?",
    "Wie wirkt eine überraschende Zinssenkung der Fed auf den US-Dollar, Gold und Aktien aus Schwellenländern?",
    "Wie kann ein Einbruch am chinesischen Immobilienmarkt Rohstoffpreise und rohstofflastige Aktienmärkte belasten?",
    "Warum steigen in einer Rezession oft die Kurse langlaufender Staatsanleihen?",
    "Wie kann die Pleite einer großen Bank über den Interbankenmarkt andere Banken in Schwierigkeiten bringen?",
    "Wie wirkt sich eine steigende Arbeitslosigkeit über den Konsum auf Unternehmensgewinne und Aktienkurse aus?",
    "Wie können hohe Staatsdefizite über steigende Anleiherenditen die Bewertung von Aktien belasten?",
    "Wie kann ein plötzlicher Abzug ausländischen Kapitals einen Währungsverfall und steigende Inflation auslösen?",
    "Warum kann eine Zinserhöhung bei variabel verzinsten Krediten zu mehr Kreditausfällen und Bankverlusten führen?",
    "Wie wirkt sich ein schwacher Yen auf japanische Exportunternehmen und den japanischen Aktienmarkt aus?",
    "Was passiert mit Optionsprämien und den Kosten der Absicherung, wenn die Volatilität stark steigt?",
    "Wie kann eine sinkende Kreditvergabe an Unternehmen Investitionen, Beschäftigung und Wachstum dämpfen?",
    "Wie können steigende Lebensmittel- und Energiepreise über die Inflationserwartungen die Geldpolitik beeinflussen?",
    "Warum können stark fallende Anleihekurse Versicherungen und Pensionsfonds unter Druck setzen?",
    "Wie kann eine Bitcoin-Rally über gehebelte Positionen in einen plötzlichen Absturz umschlagen?",
    "Wie wirkt sich eine starke Aufwertung des Schweizer Franken auf Schweizer Exporteure "
    "und die Schweizer Notenbank aus?",
    "Wie beeinflussen steigende Hypothekenzinsen die Bautätigkeit, Baustoffhersteller und Banken?",
    "Warum kann ein Handelskonflikt zwischen den USA und China auch europäische Exportwerte und den Euro treffen?",
]
# Must NOT be answered with content: forecasts, personal recommendations, off-topic.
CONTROLS = [
    "Wie hoch steht der DAX in drei Monaten?",
    "Welche Aktie soll ich jetzt kaufen, um mein Geld schnell zu verdoppeln?",
    "Wie backe ich ein Sauerteigbrot?",
]

JUDGE_PROMPT = """You assess an answer to a question about a capital-markets cause -> effect mechanism.

steps: the number of distinct consecutive cause -> effect steps the answer explains as a chain (A -> B -> C
counts 2). Separate facts that are not linked into a chain do not count.
complete: 2 = the chain reaches from the trigger in the question to the effect asked about, 1 = partly,
0 = no chain.
refused: true if the answer says it cannot answer or only suggests other topics.
depth: 1-5, how well the answer explains the mechanism, its conditions and counter-effects.

Question: {question}

Answer:
{answer}"""


class Judgement(BaseModel):
    steps: int = Field(..., ge=0)
    complete: int = Field(..., ge=0, le=2)
    refused: bool
    depth: int = Field(..., ge=1, le=5)
    note: str = Field(..., description="One sentence on what is missing, German")


def ask(question: str) -> str:
    body = json.dumps(
        {
            "model": "capital_rag_pipeline",
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:9099/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['PIPELINES_API_KEY']}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def cited_sources(answer: str) -> int:
    # Entries of the "Quellen:" list the pipeline appends: only the sources the answer cites.
    parts = answer.split("\n\nQuellen:", 1)
    return len(re.findall(r"^- \[\d+\]", parts[1], re.MULTILINE)) if len(parts) > 1 else 0


def summarize(results: list) -> dict:
    # Fasst einen Lauf zu Kennzahlen zusammen (Kettenfragen und Kontrollfragen getrennt); dieselben Werte
    # stehen in der Konsole und in MLflow. Ältere Ergebnisdateien ohne "sources"/"control" gehen auch.
    chains = [r for r in results if not r.get("control")]
    n = len(chains)
    metrics = {
        "questions": n,
        "mean_steps": sum(r["steps"] for r in chains) / n,
        "complete_rate": sum(r["complete"] == 2 for r in chains) / n,
        "mean_depth": sum(r["depth"] for r in chains) / n,
        "refused_rate": sum(r["refused"] for r in chains) / n,
        "mean_seconds": sum(r["seconds"] for r in chains) / n,
    }
    if all("sources" in r for r in chains):
        metrics["mean_sources"] = sum(r["sources"] for r in chains) / n
        metrics["one_source_rate"] = sum(r["sources"] == 1 for r in chains) / n
    controls = [r for r in results if r.get("control")]
    if controls:
        metrics["controls_refused_rate"] = sum(r["refused"] for r in controls) / len(controls)
    return metrics


def valves() -> dict:
    # Aktuelle Valve-Werte der Pipeline (vom Pipelines-Server), damit ein Lauf seine Einstellungen mitführt.
    request = urllib.request.Request(
        "http://localhost:9099/capital_rag_pipeline/valves",
        headers={"Authorization": f"Bearer {os.environ['PIPELINES_API_KEY']}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def log_to_mlflow(args, results: list):
    # Meldet den Lauf an MLflow, falls MLFLOW_TRACKING_URI gesetzt ist; ein Fehler kostet nur den Eintrag,
    # die Ergebnisdatei ist schon geschrieben.
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        params = {
            **valves(),
            "set": args.set,
            "judge_model": args.judge_model,
            "version": os.getenv("APP_VERSION", "unknown"),
        }
        tags = {"source": "eval_chains", "mlflow.note.content": args.note} if args.note else {"source": "eval_chains"}
        run_name = args.run_name or os.path.splitext(os.path.basename(args.out))[0]
        run_id = log_run(run_name, params, summarize(results), artifact=args.out, tags=tags)
        print(f"MLflow: run {run_name} ({run_id}) logged to {os.environ['MLFLOW_TRACKING_URI']}")
    except Exception as e:  # noqa: BLE001 - the result file is written; only the MLflow entry is missing
        print(f"MLflow: run not logged ({e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge_model", default="gpt-4.1-mini")
    # tuning: the questions the pipeline was tuned on; validation: held-out questions + controls.
    parser.add_argument("--set", choices=["tuning", "validation"], default="tuning")
    # MLflow (only with MLFLOW_TRACKING_URI): run name (default: name of --out) and a note on what changed.
    parser.add_argument("--run_name")
    parser.add_argument("--note")
    args = parser.parse_args()
    judge = ChatOpenAI(model=args.judge_model, temperature=0).with_structured_output(Judgement)

    questions = QUESTIONS if args.set == "tuning" else VALIDATION + CONTROLS
    results = []
    for question in questions:
        start = time.time()
        try:
            answer = ask(question)
        except Exception as e:  # noqa: BLE001 - a failed question is scored as an error, the run goes on
            answer = f"ERROR: {e}"
        seconds = round(time.time() - start)
        text = answer.split("\n\nQuellen:")[0]
        verdict = judge.invoke(JUDGE_PROMPT.format(question=question, answer=text))
        control = question in CONTROLS
        sources = cited_sources(answer)
        results.append(
            {
                "question": question,
                "control": control,
                "seconds": seconds,
                **verdict.model_dump(),
                "sources": sources,
                "answer": answer,
            }
        )
        print(
            f"{'CONTROL ' if control else ''}steps {verdict.steps} complete {verdict.complete} depth {verdict.depth} "
            f"refused {verdict.refused} sources {sources} {seconds:4d}s  {question[:60]}",
            flush=True,
        )
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

    m = summarize(results)
    n = m["questions"]
    print(
        f"\nmean steps {m['mean_steps']:.1f} | complete chains {round(m['complete_rate'] * n)}/{n} "
        f"| mean depth {m['mean_depth']:.1f} | refused {round(m['refused_rate'] * n)}/{n} | mean sources "
        f"{m['mean_sources']:.1f} (one source: {round(m['one_source_rate'] * n)}/{n}) "
        f"| mean time {m['mean_seconds']:.0f}s"
    )
    if "controls_refused_rate" in m:
        print(f"controls correctly not answered: {m['controls_refused_rate']:.0%}")
    log_to_mlflow(args, results)


if __name__ == "__main__":
    main()

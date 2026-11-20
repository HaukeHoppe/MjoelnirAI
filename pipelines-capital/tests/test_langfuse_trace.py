"""Unit tests for the Langfuse tracing in pipe(), with fake Langfuse objects (no network, no OpenAI calls).

Run in the container:
    docker exec open-webui-pipelines-capital python -m pytest /app/pipelines/tests/test_langfuse_trace.py -q
"""

import contextlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import capital_rag_pipeline as crp

BODY = {"stream": True, "user": {"id": "u-1", "name": "Guest", "email": "guest@example.com", "role": "user"}}


class FakeSpan:
    # Merkt sich, was die Pipeline am Wurzel-Span setzt: Output, Level, Score und ob er beendet wurde.
    trace_id = "t" * 32
    id = "s" * 16

    def __init__(self, **kwargs):
        self.start = kwargs
        self.updates, self.scores, self.ended = {}, [], False

    def update(self, **kwargs):
        self.updates.update(kwargs)

    def score_trace(self, **kwargs):
        self.scores.append(kwargs)

    def end(self):
        self.ended = True


class FakeLangfuse:
    # Ersetzt den Langfuse-Client; fail=True simuliert einen nicht erreichbaren Dienst.
    def __init__(self, fail=False):
        self.fail, self.spans = fail, []

    def start_observation(self, **kwargs):
        if self.fail:
            raise ConnectionError("langfuse down")
        self.spans.append(FakeSpan(**kwargs))
        return self.spans[-1]


@pytest.fixture
def pipeline(monkeypatch):
    # Pipeline ohne Index, deren eigentliche Antwort (_pipe) durch einen Stub ersetzt wird.
    attributes = []

    @contextlib.contextmanager
    def fake_propagate(**kwargs):
        attributes.append(kwargs)
        yield

    monkeypatch.setattr(crp, "propagate_attributes", fake_propagate, raising=False)
    monkeypatch.setattr(crp, "LangfuseHandler", lambda trace_context: ("handler", trace_context), raising=False)
    p = crp.Pipeline()
    p.langfuse = FakeLangfuse()
    p.attributes = attributes
    p.handlers = []

    def answer(user_message, model_id, messages, body, trace_handler=None):
        p.handlers.append(trace_handler)
        yield {"event": {"type": "status", "data": {"description": "…", "done": False}}}
        yield "Antwort "
        yield "[1]"

    p._pipe = answer
    return p


def run(p, question="Was ist QE?"):
    return list(p.pipe(question, "capital_rag_pipeline", [], BODY))


def test_question_becomes_one_trace(pipeline):
    out = run(pipeline)
    assert out[1:] == ["Antwort ", "[1]"]
    (span,) = pipeline.langfuse.spans
    assert span.start["input"] == "Was ist QE?"
    assert span.updates["output"] == "Antwort [1]"
    assert span.scores == [{"name": "answered", "value": 1, "data_type": "BOOLEAN"}]
    assert span.ended
    # Alle LangChain-Aufrufe hängen unter diesem Span.
    assert pipeline.handlers == [("handler", {"trace_id": span.trace_id, "parent_span_id": span.id})]


def test_only_the_role_leaves_the_server(pipeline):
    # Keine Nutzerkennung: weder ID noch Name oder E-Mail, nur die Rolle als Tag.
    run(pipeline)
    assert pipeline.attributes == [{"tags": ["role:user"], "trace_name": "capital-rag"}]
    sent = repr(pipeline.langfuse.spans[0].start)
    assert "u-1" not in sent
    assert "guest@example.com" not in sent


@pytest.mark.parametrize(
    ("text", "masked"),
    [
        ("Schreib an max.muster@firma.de bitte", "Schreib an [E-MAIL] bitte"),
        ("Meine IBAN DE89 3704 0044 0532 0130 00 lautet", "Meine IBAN [IBAN] lautet"),
        ("IBAN DE89370400440532013000.", "IBAN [IBAN]."),
        ("Karte 4111 1111 1111 1111 gesperrt", "Karte [KARTENNUMMER] gesperrt"),
        ("Ruf an: +49 30 1234567 oder 030/1234567", "Ruf an: [TELEFON] oder [TELEFON]"),
        ("Mobil 0171 2345678.", "Mobil [TELEFON]."),
        ("Server 192.168.1.20 antwortet", "Server [IP] antwortet"),
        ("Link https://x.de/a?token=abc&u=1 ok", "Link https://x.de/a?[PARAMETER] ok"),
    ],
)
def test_mask_removes_personal_data(text, masked):
    assert crp.mask_personal_data(data=text) == masked


@pytest.mark.parametrize(
    "text",
    [
        "Der Leitzins stieg 2022 um 0,75 Prozentpunkte auf 4,5 % (2008 2009 2010 2011).",
        "Der S&P 500 fiel von 4.796 auf 3.577 Punkte; das BIP wuchs um +2,1 %.",
        "Quantitative Lockerung (QE) senkt die Renditen langfristiger Anleihen [1, 3].",
        "Die EZB kaufte 2015-2018 Anleihen für 2.600 Mrd. Euro; Basel III, Stufe 1.2.",
        "Siehe https://de.wikipedia.org/wiki/Quantitative_Lockerung für Details.",
        "Ticker AAPL, MSFT und ISIN-Anteile von 12.5 % am MSCI World.",
        "Die Bilanzsumme betrug 3.210.000.000 Euro bzw. 1.234.567 Anteile.",
    ],
)
def test_mask_keeps_financial_text(text):
    # Kapitalmarkttexte mit Zahlen, Jahren, Prozenten und Links dürfen nicht verfälscht werden.
    assert crp.mask_personal_data(data=text) == text


def test_mask_walks_nested_data():
    data = {"messages": [{"role": "user", "content": "a@b.de"}], "n": 3, "x": None, "t": ("c@d.de",)}
    assert crp.mask_personal_data(data=data) == {
        "messages": [{"role": "user", "content": "[E-MAIL]"}],
        "n": 3,
        "x": None,
        "t": ["[E-MAIL]"],
    }


def test_no_answer_scores_zero(pipeline):
    pipeline._pipe = lambda *a, **k: iter([crp.NO_ANSWER])
    run(pipeline)
    assert pipeline.langfuse.spans[0].scores[0]["value"] == 0


def test_background_tasks_are_not_traced(pipeline):
    run(pipeline, "### Task: Generate a concise title")
    assert pipeline.langfuse.spans == []
    assert pipeline.handlers == [None]


def test_tracing_off_or_unreachable_still_answers(pipeline):
    pipeline.langfuse = None
    assert run(pipeline)[1:] == ["Antwort ", "[1]"]
    pipeline.langfuse = FakeLangfuse(fail=True)
    assert run(pipeline)[1:] == ["Antwort ", "[1]"]
    assert pipeline.handlers == [None, None]


def test_error_marks_the_trace_and_propagates(pipeline):
    def broken(*args, **kwargs):
        yield "Teil"
        raise RuntimeError("boom")

    pipeline._pipe = broken
    with pytest.raises(RuntimeError):
        run(pipeline)
    span = pipeline.langfuse.spans[0]
    assert span.updates == {"output": "Teil", "level": "ERROR", "status_message": "boom"}
    assert span.scores == []
    assert span.ended


def test_closed_chat_ends_the_trace(pipeline):
    gen = pipeline.pipe("Was ist QE?", "capital_rag_pipeline", [], BODY)
    next(gen)
    next(gen)
    gen.close()
    span = pipeline.langfuse.spans[0]
    assert span.updates["level"] == "WARNING"
    assert span.updates["output"] == "Antwort "
    assert span.ended

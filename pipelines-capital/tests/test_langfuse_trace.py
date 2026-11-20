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


def test_only_user_id_and_role_leave_the_server(pipeline):
    run(pipeline)
    assert pipeline.attributes == [{"user_id": "u-1", "tags": ["role:user"], "trace_name": "capital-rag"}]
    assert "guest@example.com" not in repr(pipeline.langfuse.spans[0].start)


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

"""Unit tests for UsageMeter (token and cost display), without OpenAI calls.

Run in the container:
    docker exec open-webui-pipelines-capital python -m pytest /app/pipelines/tests/test_usage_meter.py -q
"""

import os
import sys
import threading

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from capital_rag_pipeline import DEFAULT_PRICES, UsageMeter, run_config  # noqa: E402


def result(model: str, inp: int, out: int, cached: int = 0, in_llm_output: bool = True) -> LLMResult:
    # Baut ein LLMResult nach, wie LangChain es nach einem OpenAI-Aufruf an on_llm_end übergibt.
    usage = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
    if cached:
        usage["input_token_details"] = {"cache_read": cached}
    message = AIMessage(content="x", usage_metadata=usage, response_metadata={"model_name": model})
    return LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={"model_name": model} if in_llm_output else None,
    )


def test_dated_model_maps_to_longest_prefix():
    # gpt-4.1-mini-<datum> muss zum mini-Preis zählen, nicht zu gpt-4.1.
    meter = UsageMeter(DEFAULT_PRICES)
    meter.on_llm_end(result("gpt-4.1-mini-2025-04-14", 1000, 100))
    meter.on_llm_end(result("gpt-4.1-2025-04-14", 1000, 100))
    assert set(meter.total) == {"gpt-4.1-mini", "gpt-4.1"}


def test_cost_uses_cached_price():
    # 1.000.000 Input, davon 400.000 cached, 100.000 Output bei gpt-4o: 0,6×2,50 + 0,4×1,25 + 0,1×10 = 3,00 $.
    meter = UsageMeter(DEFAULT_PRICES)
    meter.on_llm_end(result("gpt-4o-2024-08-06", 1_000_000, 100_000, cached=400_000))
    assert abs(meter.cost("gpt-4o", meter.total["gpt-4o"]) - 3.0) < 1e-9


def test_model_name_from_response_metadata_when_streaming():
    # Beim Streamen fehlt llm_output; das Modell kommt dann aus response_metadata.
    meter = UsageMeter(DEFAULT_PRICES)
    meter.on_llm_end(result("gpt-4.1", 10, 5, in_llm_output=False))
    assert meter.total == {"gpt-4.1": [10, 0, 5]}


def test_take_resets_pending_but_keeps_total():
    # take() liefert nur den Verbrauch seit dem letzten Schritt, die Gesamtsumme bleibt.
    meter = UsageMeter(DEFAULT_PRICES)
    meter.on_llm_end(result("gpt-4.1", 10, 5))
    assert meter.take() == {"gpt-4.1": [10, 0, 5]}
    assert meter.take() == {}
    meter.on_llm_end(result("gpt-4.1", 1, 1))
    assert meter.total == {"gpt-4.1": [11, 0, 6]}


def test_embedding_tokens_are_counted_with_tiktoken():
    # "hello world" sind in cl100k_base genau 2 Tokens; Embeddings haben keine Output-Tokens.
    meter = UsageMeter(DEFAULT_PRICES)
    meter.add_embedding("text-embedding-3-large", ["hello world", "hello world"])
    assert meter.total == {"text-embedding-3-large": [4, 0, 0]}


def test_describe_formats_german_numbers_and_cached_hint():
    meter = UsageMeter(DEFAULT_PRICES)
    meter.on_llm_end(result("gpt-4o", 12_345, 1_000, cached=2_048))
    meter.add_embedding("text-embedding-3-large", ["hello world"])
    line = meter.describe(meter.total)
    assert "gpt-4o: 12.345 in (2.048 cached) / 1.000 out" in line
    assert "text-embedding-3-large: 2 in / 0 out" in line
    assert meter.describe(meter.total, per_model=False).startswith("12.347 in / 1.000 out · 0,")


def test_unknown_model_costs_zero_and_is_marked():
    meter = UsageMeter(DEFAULT_PRICES)
    meter.on_llm_end(result("o9-preview", 100, 100))
    assert "o9-preview (Preis unbekannt)" in meter.describe(meter.total)
    assert meter.cost("o9-preview", meter.total["o9-preview"]) == 0


def test_tiny_cost_is_shown_as_below_threshold():
    meter = UsageMeter(DEFAULT_PRICES)
    meter.add_embedding("text-embedding-3-large", ["hello world"])
    assert meter.describe(meter.total).endswith("<0,0001 $")


def test_parallel_calls_are_all_counted():
    # grader.batch() meldet aus mehreren Threads gleichzeitig; nichts darf verloren gehen.
    meter = UsageMeter(DEFAULT_PRICES)
    threads = [threading.Thread(target=lambda: [meter.on_llm_end(result("gpt-4.1-mini", 1, 1)) for _ in range(200)])
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert meter.total["gpt-4.1-mini"] == [1600, 0, 1600]


def test_run_config_without_meter_is_unchanged():
    assert run_config(None, max_concurrency=8) == {"max_concurrency": 8}
    meter = UsageMeter(DEFAULT_PRICES)
    assert run_config(meter) == {"callbacks": [meter]}

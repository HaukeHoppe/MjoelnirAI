"""
title: Capital Markets RAG
author: Mjoelnir AI
date: 2026-09-25
version: 0.4
license: MIT
description: RAG over the capital-markets chunks in faiss_public_index (Wikipedia articles and own explanatory texts, built by build_public_kb.py and ingest_capital_chunks.py; INDEX_DIR can point to another index such as faiss_capital_index). Query flow - fusion retrieval (HyPE dense vectors + BM25, adapted from fusion_retrieval.py), concept-graph expansion (graph.json), causal chain search (cause -> effect paths of up to 4 steps through the directed concept graph, with the chunks stating each step), LLM reranking + relevance/grounding grading per chunk (reranking.py / reliable_rag), generation that keeps time-bound opinions attributed to their source, a post-answer grounding check, and a source list that explains why each source was used (explainable_retrieval.py). Explorer mode: without a grounded answer, related covered topics are suggested as verified follow-up questions. Runs in its own pipelines container (pipelines-capital) next to the climate test pipeline.
requirements: langchain-community,langchain-openai,langchain-core,faiss-cpu,openai,pydantic,rank-bm25,numpy,langfuse
"""  # noqa: E501 - Open WebUI reads each front matter field from a single line

import json
import os
import re
import threading
from collections import Counter, OrderedDict
from collections.abc import Generator, Iterator
from difflib import SequenceMatcher
from itertools import pairwise
from typing import Literal

import numpy as np
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

try:
    from langfuse import Langfuse, propagate_attributes
    from langfuse.langchain import CallbackHandler as LangfuseHandler
except ImportError:  # optional: without the package the pipeline runs untraced
    Langfuse = None

# USD per 1M tokens: [input, cached input, output]. Check openai.com/api/pricing when models change.
DEFAULT_PRICES = {
    "gpt-4.1": [2.00, 0.50, 8.00],
    "gpt-4.1-mini": [0.40, 0.10, 1.60],
    "gpt-4o": [2.50, 1.25, 10.00],
    "text-embedding-3-large": [0.13, 0.13, 0.0],
}


class UsageMeter(BaseCallbackHandler):
    # Zählt Tokens und Kosten einer einzelnen Frage, getrennt nach Modell. Pro Frage eine eigene
    # Instanz, weil die Pipeline-Instanz von allen Nutzern gleichzeitig verwendet wird.

    def __init__(self, prices: dict):
        # Zwei Töpfe: "pending" seit dem letzten Schritt, "total" für die ganze Frage.
        # Das Lock schützt beide, weil grader.batch() aus mehreren Threads gleichzeitig meldet.
        self.prices = prices
        self.lock = threading.Lock()
        self.pending, self.total = {}, {}
        # Alle Callbacks dieser Frage für run_config: der Zähler selbst, bei aktivem Langfuse auch dessen Handler.
        self.callbacks = [self]

    def on_llm_end(self, response, **kwargs):
        # Wird von LangChain nach jedem Chat-Aufruf aufgerufen (invoke, batch, stream, structured output)
        # und liest die exakten Token-Zahlen, die OpenAI zurückgemeldet hat, samt verwendetem Modell.
        message = getattr(response.generations[0][0], "message", None)
        usage = getattr(message, "usage_metadata", None)
        if not usage:
            return
        model = (response.llm_output or {}).get("model_name") or message.response_metadata.get("model_name", "?")
        cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
        self._add(model, usage["input_tokens"], cached, usage["output_tokens"])

    def add_embedding(self, model: str, texts: list[str]):
        # Embeddings melden über LangChain keine Usage: tiktoken zählt genau die Tokens, nach denen
        # OpenAI abrechnet. Embeddings haben keine Output-Tokens.
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        self._add(model, sum(len(encoding.encode(t)) for t in texts), 0, 0)

    def _key(self, model: str) -> str:
        # Ordnet z. B. "gpt-4.1-mini-2025-04-14" dem Preiseintrag zu; der längste passende Präfix
        # gewinnt, damit gpt-4.1-mini nicht als gpt-4.1 zählt.
        keys = [k for k in self.prices if model.startswith(k)]
        return max(keys, key=len) if keys else model

    def _add(self, model: str, inp: int, cached: int, out: int):
        # Addiert einen Aufruf in beide Töpfe; eine Zeile ist [input, davon cached, output].
        with self.lock:
            for bucket in (self.pending, self.total):
                row = bucket.setdefault(self._key(model), [0, 0, 0])
                row[0] += inp
                row[1] += cached
                row[2] += out

    def take(self) -> dict:
        # Gibt den Verbrauch seit dem letzten Schritt zurück und leert den pending-Topf.
        with self.lock:
            usage, self.pending = self.pending, {}
        return usage

    def cost(self, model: str, row: list) -> float:
        # Kosten in USD: ungecachter Input + gecachter Input + Output, je × Preis / 1 Mio.
        # Unbekannte Modelle kosten 0 und werden in describe() markiert.
        p_in, p_cached, p_out = self.prices.get(model, (0, 0, 0))
        inp, cached, out = row
        return ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6

    def describe(self, usage: dict, per_model: bool = True) -> str:
        # Text für die Statuszeile im deutschen Zahlenformat. per_model=True: eine Angabe je
        # Modell; False: nur die Summe über alle Modelle (Gesamtzeile).
        def num(n: int) -> str:
            # Ganzzahl mit Punkt als Tausendertrenner, z. B. 12.345.
            return f"{n:,}".replace(",", ".")

        def money(c: float) -> str:
            # Dollarbetrag mit Komma und vier Nachkommastellen; Kleinstbeträge als "<0,0001 $".
            return "<0,0001 $" if 0 < c < 1e-4 else f"{c:.4f} $".replace(".", ",")

        total = sum(self.cost(m, r) for m, r in usage.items())
        if not per_model:
            inp = sum(r[0] for r in usage.values())
            out = sum(r[2] for r in usage.values())
            return f"{num(inp)} in / {num(out)} out · {money(total)}"
        parts = []
        for model, (inp, cached, out) in usage.items():
            name = model if model in self.prices else f"{model} (Preis unbekannt)"
            hint = f" ({num(cached)} cached)" if cached else ""
            parts.append(f"{name}: {num(inp)} in{hint} / {num(out)} out")
        return " · ".join(parts) + f" · {money(total)}"


def run_config(meter, **extra) -> dict:
    # LangChain-Config für einen Aufruf: hängt die Callbacks der Frage an (Zähler, ggf. Langfuse).
    # Ohne Zähler (z. B. generate_start_suggestions.py) bleibt alles wie bisher.
    return {**extra, **({"callbacks": list(meter.callbacks)} if meter else {})}


class Grade(BaseModel):
    relevance: int = Field(..., ge=0, le=10, description="Relevance to the question, 0-10.")
    supports_answer: bool = Field(..., description="True only if the document directly helps answer the question.")
    states_step: bool = Field(
        ...,
        description="True if the document states at least one cause -> effect link of the chain the question asks "
        "about, even if it does not answer the whole question.",
    )
    reason: str = Field(..., description="One short sentence, in the language of the question, why.")


class Claim(BaseModel):
    claim: str = Field(..., description="One claim from the answer.")
    kind: Literal["fact", "meta"] = Field(
        ...,
        description="meta = only says what the sources do/don't cover, or compares claims from the context "
        "without adding information.",
    )
    quote: str = Field(
        ...,
        description="Verbatim passage copied from the context that states the claim; empty if none.",
    )
    claim_cause: str = Field("", description="If the claim states a cause -> effect: the cause named in the claim.")
    quote_cause: str = Field(
        "", description="If the claim states a cause -> effect: the cause the quote gives for that effect."
    )
    same_cause: bool = Field(True, description="False if claim_cause and quote_cause are different causes.")
    supported: bool = Field(..., description="fact: true only if the quote states it. meta: true if it is accurate.")
    answer_sentence: str = Field(
        ..., description="The sentence of the answer that contains the claim, copied verbatim."
    )


class AnswerCheck(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class Corroboration(BaseModel):
    line: int = Field(..., description="Number of the answer line (L<n>) the fact is stated in.")
    block: int = Field(..., description="Number of a further block that states the same.")
    quote: str = Field(..., description="Verbatim sentence from that block stating it.")
    same_statement: bool = Field(..., description="True if the quote states the same cause, effect and certainty.")


class Corroborations(BaseModel):
    items: list[Corroboration] = Field(default_factory=list)


class Suggestion(BaseModel):
    topic: int = Field(..., description="Number of the topic in the list.")
    label: str = Field(..., description="Short topic name in the language of the user's question.")
    question: str = Field(
        ...,
        description="A question in the language of the user's question that the excerpt of this topic answers.",
    )


class Suggestions(BaseModel):
    suggestions: list[Suggestion] = Field(default_factory=list)


class QueryConcepts(BaseModel):
    causes: list[str] = Field(default_factory=list, description="Triggers / causes named in the question")
    effects: list[str] = Field(default_factory=list, description="Effects the question asks about; empty if none")


QUERY_CONCEPTS_PROMPT = """Name the cause -> effect concepts in this capital-markets question as short English
Title Case names of economic quantities, actors or events (e.g. "Stock Market Crash", "Gold Price",
"Interest Rates", "Bank Lending"). causes: the trigger(s) the question starts from. effects: what the question
asks the consequences for; leave empty if it asks for consequences in general.

Question: {question}"""


GRADE_PROMPT = """Rate how well the document helps answer the question.
Consider the specific intent of the question, not just keyword or topic overlap.

relevance: 0-10.
supports_answer: true only if the document contains information that directly helps answer the question.
states_step: true if the document states at least one cause -> effect link that belongs to the chain the question asks about - the trigger, an intermediate step or the final effect (e.g. for "How can a rate decision in Japan hit US tech stocks?": a document stating that Japan's low rates lead to borrowing in yen, or that rising rates lower stock valuations). It does not need to answer the whole question. False for documents that only share the topic or keywords without stating such a link.
reason: one short sentence in the language of the question.

Question: {question}

Document: {title}
{content}"""  # noqa: E501 - prompt text, sent verbatim

ANSWER_PROMPT = """Answer the question using ONLY the context below. If the context doesn't support an answer, say so.
Cite the context blocks you use as [1], [2], ... after every statement that relies on them.

Do NOT use your own knowledge - not even for well-known textbook facts or to explain a mechanism. Every statement, including every reason ("weil ...", "because ..."), must be stated in the context. If the context gives the effect but not the reason (or only part of what was asked), answer only that part and say explicitly that the sources do not explain the rest.

Stay faithful to the wording of the sources:
- Keep their degree of certainty: "könnte" / "kann" stays a possibility, never "typischerweise" or "immer".
- Only state a cause -> effect link that the context states as such. Do not join two separate statements into a new cause -> effect link.
- You may build a chain of several steps when every single step is stated in the context: A -> B in one block, B -> C in another. Write each step as its own sentence with its own citation. Never state the final effect as a direct result of the trigger (A -> C) unless the context states that link itself.

Mechanism questions (why / how / what happens if) are answered as a chain, as long as the context supports it:
**<Short heading naming the mechanism>**
1. **<Step>**: cause -> effect and how it works [n]
2. **<Step>**: ... (as many steps as the context supports, in causal order)
**Bedingungen und Gegeneffekte**: when the chain holds or breaks, and what works against it, as stated in the context.
No introduction or summary sentence that links the trigger directly to the final effect - the steps make that link.
Phrase each step close to the wording of its block and give only the reason the block gives; do not add reasons, actors or effects of your own.
The context may start with "Causal paths": chains of relations found across the sources, each step with the blocks that state it. Use them to find and order the steps; every step still needs its block citation. Other questions are answered in normal prose.
Build the chain from all blocks, not only from the block that covers the most: cite every block that states a step (e.g. [1][3]), and take a step from another block when it is stated there but missing in the block that covers the most.

The context blocks come from the indexed capital-markets sources (each block label names its source) and come in two kinds:
- CONCEPT blocks are timeless explanations. You may state them as general explanations.
- OPINION blocks are time-bound views, forecasts or positioning as stated in the named source. Always attribute them to that source in the same paragraph, using the source title exactly as written in the block label without the [id] and #n (e.g. "Laut Marktkommentar Mai 2026 ..."). Never present them as current facts or as a current recommendation.
- Exception: a "Relation [TIMELESS]" line is a general mechanism, even inside an OPINION block. You may state it as a general explanation. Everything else in an OPINION block, including "Relation [TIME-BOUND]" lines, must be attributed to its source.
- If opinion blocks from different sources disagree, show each view with its source and say that they differ. Do not merge them into one view, and do not end with a summary or conclusion that combines views from different sources.

Answer in the language of the question.

Context:
{context}

Question: {question}"""  # noqa: E501 - prompt text, sent verbatim

CHECK_PROMPT = """Check every factual claim in the answer against the context.

Split the answer into individual claims. Every reason or explanation ("weil ...", "because ...", "dadurch ...") is a separate claim. A sentence that joins two statements ("..., wobei ...", "... und ...", "..., während ...") is two claims.
Write every claim self-contained: replace "dadurch", "dies", "was", "somit", "this" etc. with what they refer to, so each claim names its own cause (e.g. "Sinkende Zinsen steigern die Nachfrage nach Anleihen", not "Die Nachfrage steigt dadurch").

For each claim:
- kind: "meta" if it only says what the sources do or do not cover, or only compares / contrasts claims from the context (e.g. "the two sources recommend different positions") without adding any information of its own. Everything else is "fact". A meta statement that adds any new information is a "fact".
- quote (facts only): copy, character for character, ONE sentence from the context that states the claim. Never join sentences from different places. Only exception: if the statement continues in the directly following sentence(s) of the same passage, which refer back with a pronoun ("Sie", "Er", "Dies", "Diese", "It", "This", ...) or a connector ("deshalb", "dadurch", "daher", "somit", "therefore", ...), copy up to three consecutive sentences. If no single sentence states it, the claim is unsupported and quote stays empty. For meta claims leave it empty.
- supported: for a fact, true only if the quote states the claim - general knowledge does NOT count, even if the claim is true. For a meta claim, true if it accurately describes the context.

The quote must state the claim itself: same cause, same effect, same degree of certainty. A fact is unsupported if it:
- turns a possibility ("könnte", "kann", "may") into a rule ("typischerweise", "immer", "typically");
- links a cause and an effect that the context mentions only separately, or links the effect to a different cause. If the quote names a different cause than the claim (claim: falling rates raise demand; quote: investor confidence raises demand), the claim is unsupported;
- attributes a view to the wrong source.
A statement that merges different views from different sources into one is unsupported (fact or meta).

claim_cause / quote_cause / same_cause: for every claim that states a cause -> effect, write down the cause the claim names and the cause the quote gives for the same effect, then set same_cause. Example: claim "Sinkende Zinsen steigern die Nachfrage nach Anleihen", quote "Wenn Investoren Vertrauen zurückgewinnen, ... steigt die Nachfrage" -> claim_cause "sinkende Zinsen", quote_cause "Vertrauen der Investoren", same_cause false.
answer_sentence: the sentence of the answer the claim was taken from, copied verbatim (before you rewrote the claim).

Context:
{context}

Answer:
{answer}"""  # noqa: E501 - prompt text, sent verbatim

CORROBORATE_PROMPT = """Below is a checked answer with citations [n], split into numbered lines (L1, L2, ...), and the context blocks it was written from.
For every line of the answer that states a fact, look for OTHER blocks (not already cited in that line) that state the same fact themselves.

For each such block:
- line: the number of the answer line.
- block: the number of the further block.
- quote: copy, character for character, ONE sentence from that block that states the fact.
- same_statement: true only if the quote states the same cause, the same effect and the same degree of certainty as a statement in that line. A block that only shares the topic, names a different cause or states a different effect does not count.

Return only blocks that really state the fact. Returning nothing is fine.

Context:
{context}

Answer:
{answer}"""  # noqa: E501 - prompt text, sent verbatim

REVISE_PROMPT = """Rewrite the answer to fix the problems listed below.
- "not stated in the sources": remove the statement. Do not replace it with a sentence that repeats or denies it ("Die Quellen erklären nicht, dass ...") - that is a new claim about the sources and often wrong. Only if the removal leaves part of the question unanswered, you may add one general note such as "Zu <Aspekt der Frage> sagen die Quellen nichts."
- "missing source attribution": keep the statement but name the given source title in the same paragraph (e.g. "Laut <title> ...").
- "merges separate statements": keep the content but write each source statement as its own sentence with its citation; do not connect them with "wobei", "da", "weil" etc.
Do not add any new information. Keep the citations [1], [2], ... of the remaining statements. Answer in the language of the question.
Return only the rewritten answer text, without a heading or label such as "Answer:".

Context:
{context}

Question: {question}

Answer:
{answer}

Problems:
{unsupported}"""  # noqa: E501 - prompt text, sent verbatim

EXPLORE_PROMPT = """{situation}
Below are topics that ARE covered in the indexed capital-markets sources, each with an excerpt from the sources.

Pick up to {n} topics {goal}.
For each picked topic:
- topic: its number from the list.
- label: a short topic name in the language of the user's question.
- question: one follow-up question in the language of the user's question that the excerpt of this topic answers. Ask only for what the excerpt actually explains - do not ask about details it does not contain.
Skip topics that have nothing to do with the user's interest. Returning fewer topics, or none, is fine.

User's question: {question}

Topics:
{topics}"""  # noqa: E501 - prompt text, sent verbatim

EXPLORE_NO_ANSWER = (
    "The user asked a question that the sources do not answer.",
    "that are most closely related to what the user wanted to know, so they can explore nearby material instead",
)
EXPLORE_ANSWERED = (
    "The user asked a question and already got an answer from the sources.",
    "the user would most likely want to explore next to deepen or broaden that answer. "
    "Do not pick topics that only repeat the question",
)

NO_ANSWER = (
    "Dazu habe ich in den Quellen leider keine belegte Antwort gefunden. "
    "Ich antworte nur mit dem, was dort tatsächlich steht – deshalb rate ich hier nicht."
)

# Open WebUI sends its background tasks (title, tags, follow-ups) to the chat
# model itself, as a prompt starting with "### Task:".
FOLLOW_UP_TASK = re.compile(r"^### Task:\s*Suggest .*follow-up questions", re.IGNORECASE)


def normalize(text: str) -> str:
    # Parentheticals ("(Trading Halts)") are dropped: quotes often leave them out.
    text = re.sub(r"\s*\([^)]*\)", "", text.lower())
    text = re.sub(r"[\"'„“”‚‘’«»]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# A sentence continues the previous one if it refers back to it early on: with a
# pronoun ("Volatilität ... beschreibt das Risiko. Sie wird ... gemessen.") or a
# causal connector ("Der Wert der Sicherheiten sinkt. Broker fordern deshalb
# Nachschüsse."). Then the source itself states the link between the sentences.
ANAPHORS = set(
    """
    sie er es dies diese dieser dieses diesen diesem ihr ihre ihren ihrem ihrer
    sein seine seinen seinem seiner deren dessen
    it its they their these this
    """.split()
)
CONNECTORS = set(
    """
    deshalb dadurch daher darum deswegen somit folglich infolgedessen dementsprechend
    therefore thus hence consequently
    """.split()
)


def continues(sentence: str) -> bool:
    words = re.findall(r"\w+", sentence)
    return bool(words) and (words[0] in ANAPHORS or any(w in CONNECTORS for w in words[:8]))


# A numbered step of a chain text ("2. Euro-Abwertung → Importpreise: ..."): one line stating one
# cause -> effect, so its sentences belong together even without a connector.
CHAIN_STEP = re.compile(r"^\s*\d+\.\s")


def in_chain_step(quote: str, context: str) -> bool:
    return any(CHAIN_STEP.match(line) and quote in normalize(line) for line in context.splitlines())


def quote_in_context(quote: str, context: str) -> bool:
    # The checker's quote must really appear in the context, so the check cannot
    # be passed with an invented quote. Minor copy differences are tolerated.
    # A quote is one sentence, up to three consecutive sentences of the same
    # passage where each further sentence refers back to the previous one
    # (continues()), or up to four consecutive sentences within one numbered
    # chain step. Anything else would let a cause from one sentence be attached
    # to an effect from an unrelated one.
    raw_context = context
    quote, context = normalize(quote), normalize(context)
    sentences = re.split(r"(?<=[.!?])\s+", quote)
    if len(quote) < 15 or len(sentences) > 4:
        return False
    if not all(continues(s) for s in sentences[1:]):
        return len(sentences) > 1 and in_chain_step(quote, raw_context)
    if len(sentences) > 3:
        return False
    if quote in context:
        return True
    match = SequenceMatcher(None, context, quote, autojunk=False).find_longest_match(0, len(context), 0, len(quote))
    return match.size >= 0.9 * len(quote)


def joins_sentences(quote: str, context: str) -> bool:
    # True if the quote is several sentences that each appear in the context:
    # the claim merges separate source statements and should be split, rather
    # than being unsupported.
    sentences = re.split(r"(?<=[.!?])\s+", quote.strip())
    return len(sentences) > 1 and all(quote_in_context(s, context) for s in sentences)


# Video attribution is checked in code, not by the checker LLM: the LLM rewrites
# claims to be self-contained, dropping the "Im Video ..." it should look for.


def video_key(text: str) -> str:
    # "LIVE zur FED-Entscheidung_ Webinar" and "LIVE zur FED-Entscheidung: Webinar" match.
    return re.sub(r"\W+", "", text.lower())


def parse_video(source: str) -> tuple:
    # "Marktgespräch [123456789] (de-x-autogen) #1" -> ("Marktgespräch", "123456789")
    # "Marktkommentar Mai 2026_transcript #3" -> ("Marktkommentar Mai 2026", "")
    match = re.match(r"^(.*?)\s*\[(\d+)\]", source)
    if match:
        return (match.group(1), match.group(2))
    return (re.sub(r"_transcript$", "", re.sub(r"\s*#\d+$", "", source)), "")


def required_videos(quote: str, blocks: list[dict]) -> list[tuple]:
    # The videos a quoted statement must be attributed to; empty if it comes from
    # a CONCEPT block or a TIMELESS relation line (may be stated generally).
    matched = [b for b in blocks if quote_in_context(quote, b["text"])]
    for block in matched:
        if not block["opinion"]:
            return []
        for line in block["timeless"]:
            if quote_in_context(quote, line) or normalize(line) in normalize(quote):
                return []
    return sorted({video for block in matched for video in block["videos"]})


def paragraph_of(answer: str, sentence: str) -> str:
    paragraphs = [p for p in re.split(r"\n\s*\n", answer) if p.strip()] or [answer]
    sentence = normalize(sentence)

    def overlap(paragraph: str) -> int:
        paragraph = normalize(paragraph)
        return (
            SequenceMatcher(None, paragraph, sentence, autojunk=False)
            .find_longest_match(0, len(paragraph), 0, len(sentence))
            .size
        )

    return max(paragraphs, key=overlap)


def drop_sentences(answer: str, sentences: list[str]) -> str:
    # Removes the answer sentences the checker flagged (matched fuzzily, since
    # the checker copies them only roughly verbatim). Line structure is kept, so
    # bullet lists stay lists; lines left empty are dropped.
    targets = [normalize(s) for s in sentences if s.strip()]

    def flagged(sentence: str) -> bool:
        sentence = normalize(sentence)
        return any(SequenceMatcher(None, sentence, t, autojunk=False).ratio() >= 0.8 for t in targets)

    lines = []
    for line in answer.split("\n"):
        if not line.strip():
            lines.append("")
            continue
        kept = [s for s in re.split(r"(?<=[.!?])\s+", line.strip()) if not flagged(s)]
        if kept:
            lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def names_video(paragraph: str, videos: list[tuple]) -> bool:
    key = video_key(paragraph)
    return any(video_key(title) in key or (vid and vid in paragraph) for title, vid in videos)


# Function words would otherwise make BM25 match almost every chunk.
STOPWORDS = set(
    """
    der die das den dem des ein eine einen einem einer eines und oder aber als auch
    an auf aus bei bis durch für gegen in im ins mit nach ohne über um unter von vor
    zu zum zur ist sind war waren wird werden wurde hat haben hatte kann können
    muss soll sollte würde ich du er sie es wir ihr man mich mir sich was wie wo
    wann warum welche welcher welches wer nicht kein keine noch nur so sehr schon
    dass wenn dann denn da hier jetzt mal ja nein doch es gibt
    the a an and or but of to in on at for with by from is are was were be been
    it this that these those what how why when where which who do does did can
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def cite_source(source: str) -> str:
    # "Marktgespräch [123456789] (de-x-autogen) #1" -> "Marktgespräch [123456789] #1"
    # "Marktkommentar Mai 2026_transcript #3" -> "Marktkommentar Mai 2026 #3"
    source = re.sub(r"_transcript(\s*#\d+)$", r"\1", source)
    return re.sub(r"\s+\([^)]*\)(\s*#\d+)$", r"\1", source)


def same_question(a: str, b: str, idf: dict) -> bool:
    # A suggestion that only rephrases the user's question ("Wie wichtig sind
    # IPO-Daten für ...?" -> "Wie helfen IPO-Daten bei ...?") is no new topic.
    # Shared words are weighted by rarity (BM25 idf), so sharing "IPO" counts,
    # sharing "Daten" or "wichtig" hardly.
    if SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio() >= 0.75:
        return True
    top = max(idf.values(), default=1.0)

    def weight(tokens: set) -> float:
        return sum(idf.get(t, top) for t in tokens)

    tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
    lighter = min(weight(tokens_a), weight(tokens_b))
    return lighter > 0 and weight(tokens_a & tokens_b) >= 0.75 * lighter


def cited_numbers(answer: str, count: int) -> set:
    # Context block numbers cited in the answer: "[1]", "[1][3]", "[1], [2]", "[1, 3]".
    numbers = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        numbers.update(int(n) for n in group.split(","))
    return {n for n in numbers if 1 <= n <= count}


def minmax(scores: dict) -> dict:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return dict.fromkeys(scores, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Pipeline:
    class Valves(BaseModel):
        # Public knowledge base (build_public_kb.py); any index built by ingest_capital_chunks.py works.
        INDEX_DIR: str = "/data/faiss_public_index"
        # Must match --embedding_model in ingest_capital_chunks.py.
        EMBEDDING_MODEL: str = "text-embedding-3-large"
        # gpt-4.1 follows multi-step chains across blocks better than gpt-4o-mini.
        LLM_MODEL: str = "gpt-4.1"
        # Model used for reranking/grading of retrieved chunks and the query-concept step.
        # gpt-4.1-mini: own rate limits (gpt-4o-mini's daily request limit is shared with builds).
        GRADER_MODEL: str = "gpt-4.1-mini"
        # Model for the claim-by-claim grounding check of the answer. A stronger
        # model than LLM_MODEL, so it catches textbook knowledge the answer model
        # slipped in.
        CHECK_MODEL: str = "gpt-4o"
        # Every chunk has ~9-13 vectors (embedding_text + hypothetical questions),
        # so fetch more raw hits than we need and de-duplicate by chunk_id.
        FETCH_K: int = 40
        # The index uses MAX_INNER_PRODUCT on unit-length vectors, so the score is
        # cosine similarity (higher = closer). Dense hits below this are ignored;
        # a chunk can still come in through BM25 (e.g. exact ticker match).
        MIN_SIMILARITY: float = 0.35
        # Fusion weight: ALPHA * dense + (1 - ALPHA) * BM25 (both min-max normalized).
        ALPHA: float = 0.6
        # Fused candidates passed on to graph expansion and grading.
        CANDIDATE_K: int = 12
        # Graph expansion: chunks asserting the same relation as the top seeds.
        GRAPH_SEED_K: int = 3
        GRAPH_EXPAND_K: int = 4
        # Grading: keep chunks with supports_answer and relevance >= this.
        MIN_RELEVANCE: int = 6
        TOP_K: int = 6
        # Causal chain search: the question's causes and effects are matched to graph concepts, and
        # directed cause -> effect paths of up to CHAIN_MAX_HOPS steps between them are followed through
        # the concept graph. The chunks stating the steps join the candidates; a path chunk only needs
        # CHAIN_MIN_RELEVANCE, because a single step rarely answers the whole question by itself.
        CHAIN_SEARCH: bool = True
        CHAIN_MAX_HOPS: int = 4
        CHAIN_PATHS: int = 3
        CHAIN_EXTRA_K: int = 8
        CHAIN_TOP_K: int = 6
        CHAIN_MIN_RELEVANCE: int = 4
        # Cosine similarity for matching a question concept to a graph concept.
        CONCEPT_MIN_SIMILARITY: float = 0.55
        # The answer is generated in full, then every claim must be backed by a
        # verbatim quote from the context before anything is shown. Unsupported
        # claims are removed (up to MAX_REVISIONS rewrites); if some remain, no
        # answer is given. Disabling this streams the unchecked answer directly.
        CHECK_ANSWER_GROUNDING: bool = True
        MAX_REVISIONS: int = 2
        # After the check: further blocks that state the same as an answer sentence are cited there too, so a
        # step backed by several sources shows all of them. Each added citation needs a verbatim quote from
        # exactly that block; opinion blocks are not added (they would need a source attribution).
        CORROBORATE: bool = True
        CORROBORATE_MAX_PER_LINE: int = 2
        # Progress is shown as one status line that updates step by step. This
        # additionally writes the full step log (incl. rejected claims) into a
        # collapsible "thinking" block - useful for debugging.
        SHOW_THINKING_LOG: bool = False
        # Explorer mode: when there is no grounded answer, suggest related topics
        # the index does cover, each as a follow-up question. A suggestion is only
        # shown if the grader confirms its excerpt answers that question.
        EXPLORE_MODE: bool = True
        EXPLORE_SUGGESTIONS: int = 4
        # A topic (graph concept) must appear in at least this many chunks to be
        # suggested, so the user is not sent to a one-off mention.
        EXPLORE_MIN_CHUNKS: int = 2
        # Shows tokens (input/output) and cost per step and per question in the status line.
        SHOW_COSTS: bool = True
        # USD per 1M tokens as JSON: {"model prefix": [input, cached input, output]}.
        MODEL_PRICES: str = json.dumps(DEFAULT_PRICES)
        # Langfuse tracing: one trace per question with every LLM step. Active only when LANGFUSE_PUBLIC_KEY and
        # LANGFUSE_SECRET_KEY are set in the container (.env); this switch turns it off without a deployment.
        LANGFUSE_ENABLED: bool = True

    def __init__(self):
        self.name = "Capital Markets RAG"
        self.valves = self.Valves()
        self.vectorstore = None
        self.docs = {}
        self.bm25 = None
        self.bm25_ids = []
        self.relation_chunks = {}
        self.concept_chunks = {}
        self.concept_neighbors = {}
        self.concept_summaries = {}
        # Directed graph for the chain search: source -> {target: total edge weight}, and unit vectors
        # of the concept names (rows in concept_names order) to match question concepts.
        self.out_edges = {}
        self.concept_names = []
        self.concept_vectors = None
        self.concept_extractor = None
        # Recent questions -> what the follow-up task needs (see _follow_ups).
        self.recent = OrderedDict()
        self.llm = None
        self.answer_llm = None
        self.reviser = None
        self.grader = None
        self.checker = None
        self.corroborator = None
        self.langfuse = None

    def _load(self):
        # Loads the index built by ingest_capital_chunks.py. Does NOT run ingestion.
        # Missing index must not raise here, or the whole pipelines server fails
        # to start - pipe() reports it instead.
        if not os.path.exists(os.path.join(self.valves.INDEX_DIR, "index.faiss")):
            print(f"[capital_rag] No index in {self.valves.INDEX_DIR} - run ingest_capital_chunks.py")
            self.vectorstore = None
        else:
            # distance_strategy / normalize_L2 are not stored in index.pkl, so they
            # have to be passed again to match how the index was built.
            self.vectorstore = FAISS.load_local(
                self.valves.INDEX_DIR,
                OpenAIEmbeddings(model=self.valves.EMBEDDING_MODEL),
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                normalize_L2=True,
            )
            self._build_keyword_index()
            self._load_graph()

        # stream_usage: OpenAI reports token usage for streamed answers too (cost display).
        self.llm = ChatOpenAI(model=self.valves.LLM_MODEL, temperature=0, streaming=True, stream_usage=True)
        self.answer_llm = ChatOpenAI(model=self.valves.LLM_MODEL, temperature=0)
        grader_llm = ChatOpenAI(model=self.valves.GRADER_MODEL, temperature=0)
        self.grader = grader_llm.with_structured_output(Grade)
        check_llm = ChatOpenAI(model=self.valves.CHECK_MODEL, temperature=0)
        self.checker = check_llm.with_structured_output(AnswerCheck)
        self.corroborator = check_llm.with_structured_output(Corroborations)
        self.explorer = grader_llm.with_structured_output(Suggestions)
        self.concept_extractor = grader_llm.with_structured_output(QueryConcepts)
        # Revisions use the stronger model too: gpt-4o-mini tended to keep the
        # flagged statement in slightly different words.
        self.reviser = check_llm
        self._init_langfuse()

    def _init_langfuse(self):
        # Verbindet sich mit Langfuse, wenn Paket, beide Keys und LANGFUSE_ENABLED da sind; sonst kein Tracing.
        # Base-URL (EU-Cloud) und Umgebung liest der Client aus LANGFUSE_BASE_URL / LANGFUSE_TRACING_ENVIRONMENT.
        keys = os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
        if not (Langfuse and keys and self.valves.LANGFUSE_ENABLED):
            self.langfuse = None
            return
        try:
            self.langfuse = Langfuse()
            print("[capital_rag] Langfuse tracing enabled")
        except Exception as e:  # noqa: BLE001 - tracing is optional, the pipeline runs without it
            print(f"[capital_rag] Langfuse disabled: {e}")
            self.langfuse = None

    def _start_trace(self, question: str, body: dict):
        # Legt den Wurzel-Span einer Frage an und einen LangChain-Handler, der alle Aufrufe darunter hängt.
        # Explizite trace_context statt OpenTelemetry-Kontext: pipe() ist ein Generator und grader.batch()
        # läuft in Threads, ein "aktueller" Span ginge zwischen den Schritten verloren.
        # Nutzer: nur die interne Open-WebUI-ID und die Rolle, nie Name oder E-Mail.
        if self.langfuse is None:
            return None
        user = body.get("user") or {}
        try:
            with propagate_attributes(
                user_id=user.get("id"),
                tags=[f"role:{user['role']}"] if user.get("role") else None,
                trace_name="capital-rag",
            ):
                root = self.langfuse.start_observation(
                    name="capital-rag",
                    input=question,
                    metadata={
                        "llm_model": self.valves.LLM_MODEL,
                        "grader_model": self.valves.GRADER_MODEL,
                        "check_model": self.valves.CHECK_MODEL,
                        "index_dir": self.valves.INDEX_DIR,
                    },
                )
            return root, LangfuseHandler(trace_context={"trace_id": root.trace_id, "parent_span_id": root.id})
        except Exception as e:  # noqa: BLE001 - tracing must never block an answer
            print(f"[capital_rag] Langfuse trace failed: {e}")
            return None

    def _end_trace(self, trace, output: str, error: BaseException | None = None):
        # Schließt den Trace: gezeigte Antwort als Output und Score "answered" (0 = keine belegte Antwort),
        # damit sich in Langfuse die unbeantworteten Fragen filtern lassen. Abbruch oder Fehler: kein Score.
        root, _ = trace
        try:
            if isinstance(error, GeneratorExit):
                root.update(output=output, level="WARNING", status_message="chat closed before the answer was complete")
            elif error is not None:
                root.update(output=output, level="ERROR", status_message=str(error))
            else:
                root.update(output=output)
                root.score_trace(name="answered", value=0 if NO_ANSWER in output else 1, data_type="BOOLEAN")
            root.end()
        except Exception as e:  # noqa: BLE001 - tracing must never break an answer
            print(f"[capital_rag] Langfuse trace not closed: {e}")

    def _build_keyword_index(self):
        # One BM25 document per chunk: title + concepts + content + every text it
        # was embedded under (embedding_text, hypothetical questions).
        self.docs, texts = {}, {}
        self.concept_summaries = {}
        for docstore_id in self.vectorstore.index_to_docstore_id.values():
            doc = self.vectorstore.docstore.search(docstore_id)
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in self.docs:
                self.docs[chunk_id] = doc
                meta = doc.metadata
                texts[chunk_id] = [meta["title"], " ".join(meta["concepts"]), doc.page_content]
                if meta["chunk_type"] == "summary":
                    for concept in meta["concepts"]:
                        self.concept_summaries.setdefault(concept, []).append(chunk_id)
            texts[chunk_id].append(doc.metadata.get("matched_text", ""))
        self.bm25_ids = list(texts)
        self.bm25 = BM25Okapi([tokenize(" ".join(texts[i])) for i in self.bm25_ids])

    def _load_graph(self):
        self.relation_chunks = {}
        self.concept_chunks = {}
        self.concept_neighbors = {}
        graph_path = os.path.join(self.valves.INDEX_DIR, "graph.json")
        if not os.path.exists(graph_path):
            print(f"[capital_rag] No graph.json in {self.valves.INDEX_DIR} - graph expansion disabled")
            return
        with open(graph_path, encoding="utf-8") as f:
            graph = json.load(f)
        for node in graph["nodes"]:
            self.concept_chunks[node["id"]] = [c for c in node["chunk_ids"] if c in self.docs]
        self.out_edges = {}
        for edge in graph["edges"]:
            key = (edge["source"], edge["target"], edge["direction"])
            self.relation_chunks[key] = [e["chunk_id"] for e in edge["evidence"]]
            for a, b in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
                self.concept_neighbors.setdefault(a, Counter())[b] += edge["weight"]
            if edge["source"] != edge["target"]:
                self.out_edges.setdefault(edge["source"], Counter())[edge["target"]] += edge["weight"]
        if self.valves.CHAIN_SEARCH:
            self._load_concept_vectors()

    def _load_concept_vectors(self):
        # Embeddings of all concept names, cached next to the index (concept_vectors.npz) and only
        # recomputed when the graph's concepts change.
        names = sorted(self.concept_chunks)
        path = os.path.join(self.valves.INDEX_DIR, "concept_vectors.npz")
        if os.path.exists(path):
            cached = np.load(path, allow_pickle=False)
            if list(cached["names"]) == names:
                self.concept_names, self.concept_vectors = names, cached["vectors"]
                return
        try:
            vectors = np.array(OpenAIEmbeddings(model=self.valves.EMBEDDING_MODEL).embed_documents(names))
        except Exception as e:  # noqa: BLE001 - optional feature, the pipeline runs without it
            print(f"[capital_rag] Concept vectors failed, chain search disabled: {e}")
            return
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.concept_names, self.concept_vectors = names, vectors.astype(np.float32)
        try:
            np.savez(path, names=np.array(names), vectors=self.concept_vectors)
        except OSError as e:
            print(f"[capital_rag] Could not cache concept vectors: {e}")

    async def on_startup(self):
        self._load()

    async def on_shutdown(self):
        # Noch gepufferte Traces vor dem Beenden an Langfuse senden.
        if self.langfuse is not None:
            self.langfuse.flush()

    async def on_valves_updated(self):
        self._load()

    def _fusion_candidates(self, query: str, meter=None) -> list[tuple]:
        # Adapted from fusion_retrieval.py: min-max normalized dense and BM25
        # scores, combined with ALPHA. Dense scores are per chunk (best vector).
        if meter:
            meter.add_embedding(self.valves.EMBEDDING_MODEL, [query])
        dense = {}
        for doc, score in self.vectorstore.similarity_search_with_score(query, k=self.valves.FETCH_K):
            if score < self.valves.MIN_SIMILARITY:
                continue
            chunk_id = doc.metadata["chunk_id"]
            dense[chunk_id] = max(score, dense.get(chunk_id, score))

        bm25_raw = self.bm25.get_scores(tokenize(query))
        top = np.argsort(bm25_raw)[::-1][: self.valves.FETCH_K]
        keyword = {self.bm25_ids[i]: float(bm25_raw[i]) for i in top if bm25_raw[i] > 0}

        dense_n, keyword_n = minmax(dense), minmax(keyword)
        alpha = self.valves.ALPHA
        fused = {
            chunk_id: alpha * dense_n.get(chunk_id, 0.0) + (1 - alpha) * keyword_n.get(chunk_id, 0.0)
            for chunk_id in set(dense) | set(keyword)
        }
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return ranked[: self.valves.CANDIDATE_K]

    def _graph_expand(self, candidates: list[tuple]) -> list[tuple]:
        # Chunks from other passages/videos that assert the same relation
        # (source -> target, direction) as the top seeds.
        present = {chunk_id for chunk_id, _ in candidates}
        extra = []
        for chunk_id, _ in candidates[: self.valves.GRAPH_SEED_K]:
            for rel in self.docs[chunk_id].metadata.get("relations", []):
                key = (rel["source"], rel["target"], rel["direction"])
                for other in self.relation_chunks.get(key, []):
                    if other in self.docs and other not in present:
                        present.add(other)
                        extra.append((other, 0.0))
        return candidates + extra[: self.valves.GRAPH_EXPAND_K]

    def _match_concepts(self, phrases: list[str], meter=None) -> dict:
        # Question concept phrases -> graph concepts (top 3 per phrase above CONCEPT_MIN_SIMILARITY).
        if not phrases or self.concept_vectors is None:
            return {}
        if meter:
            meter.add_embedding(self.valves.EMBEDDING_MODEL, phrases)
        vectors = np.array(self.vectorstore.embedding_function.embed_documents(phrases), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        matched = {}
        for row in vectors @ self.concept_vectors.T:
            for i in np.argsort(row)[::-1][:3]:
                if row[i] >= self.valves.CONCEPT_MIN_SIMILARITY:
                    name = self.concept_names[i]
                    matched[name] = max(float(row[i]), matched.get(name, 0.0))
        return matched

    def _causal_paths(self, question: str, meter=None) -> list[list[str]]:
        # Directed cause -> effect paths through the concept graph, from the question's causes to its
        # effects (breadth-first, so shorter paths first). Without effects: the strongest onward chains.
        asked = self.concept_extractor.invoke(QUERY_CONCEPTS_PROMPT.format(question=question), config=run_config(meter))
        starts = self._match_concepts(asked.causes, meter)
        ends = self._match_concepts(asked.effects, meter)
        if not starts:
            return []
        max_hops, wanted = self.valves.CHAIN_MAX_HOPS, self.valves.CHAIN_PATHS
        paths = []
        if ends:
            # Breadth-first with up to 3 predecessors per concept (strongest edges first), then every
            # reached effect is traced back to the causes.
            preds, frontier, seen = {}, list(starts), set(starts)
            for _ in range(max_hops):
                nxt = []
                for node in frontier:
                    for target, _ in self.out_edges.get(node, Counter()).most_common(30):
                        if target in starts:
                            continue
                        if len(preds.setdefault(target, [])) < 3 and node not in preds[target]:
                            preds[target].append(node)
                        if target not in seen:
                            seen.add(target)
                            nxt.append(target)
                frontier = nxt

            def back(node: str, path: list[str]) -> Iterator[list[str]]:
                if node in starts:
                    yield [node, *path]
                    return
                if len(path) >= max_hops:
                    return
                for prev in preds.get(node, []):
                    if prev not in path:
                        yield from back(prev, [node, *path])

            for end in sorted(ends, key=ends.get, reverse=True):
                for path in back(end, []):
                    if len(path) > 1 and path not in paths:
                        paths.append(path)
        else:
            # Strongest onward chain from each cause, branching at the first step.
            for start in sorted(starts, key=starts.get, reverse=True)[:2]:
                for first, _ in self.out_edges.get(start, Counter()).most_common(2):
                    path = [start, first]
                    while len(path) <= max_hops:
                        options = [
                            t for t, _ in self.out_edges.get(path[-1], Counter()).most_common(5) if t not in path
                        ]
                        if not options:
                            break
                        path.append(options[0])
                    paths.append(path)

        # Shorter and better-supported paths first.
        def support(p: list[str]) -> int:
            # Wie oft die Schritte des Pfads in den Quellen belegt sind (Summe der Kantengewichte).
            return sum(self.out_edges[a][b] for a, b in pairwise(p))

        paths.sort(key=lambda p: (len(p), -support(p)))
        return paths[:wanted]

    def _path_chunks(self, paths: list[list[str]], present: set) -> list[int]:
        # The chunks that state the steps of the paths: a greedy cover, preferring chunks that state
        # several steps at once (typically a chain text).
        step_chunks = {}
        for path in paths:
            for a, b in pairwise(path):
                ids = {c for (s, t, _), chunks in self.relation_chunks.items() if s == a and t == b for c in chunks}
                step_chunks[(a, b)] = {c for c in ids if c in self.docs}
        uncovered, chosen = set(step_chunks), []
        while uncovered and len(chosen) < self.valves.CHAIN_EXTRA_K:
            best = max(
                {c for step in uncovered for c in step_chunks[step]},
                key=lambda c: (sum(c in step_chunks[s] for s in uncovered), c in present),
                default=None,
            )
            if best is None:
                break
            chosen.append(best)
            uncovered = {s for s in uncovered if best not in step_chunks[s]}
        return chosen

    def _grade(self, query: str, candidates: list[tuple], path_ids: frozenset = frozenset(), meter=None) -> list[tuple]:
        # reranking.py-style 0-10 LLM scoring combined with reliable_rag's
        # relevance grading: unsupported chunks are dropped before generation.
        docs = [self.docs[chunk_id] for chunk_id, _ in candidates]
        grades = self.grader.batch(
            [
                GRADE_PROMPT.format(question=query, title=doc.metadata["title"], content=doc.page_content)
                for doc in docs
            ],
            config=run_config(meter, max_concurrency=8),
            return_exceptions=True,
        )
        kept, steps = [], []
        for (chunk_id, fused), doc, grade in zip(candidates, docs, grades):
            if isinstance(grade, Exception):
                print(f"[capital_rag] Grading failed for chunk {chunk_id}: {grade}")
                continue
            if grade.supports_answer and grade.relevance >= self.valves.MIN_RELEVANCE:
                kept.append((doc, grade, fused))
            elif (chunk_id in path_ids or grade.states_step) and grade.relevance >= self.valves.CHAIN_MIN_RELEVANCE:
                # A step of the asked chain (from the graph's causal paths or as judged by the grader):
                # relevant as a link even if it alone does not answer, so a chain can be built from several
                # sources instead of the one block that covers the most.
                steps.append((doc, grade, fused))
        kept.sort(key=lambda hit: (hit[1].relevance, hit[2]), reverse=True)
        steps.sort(key=lambda hit: hit[1].relevance, reverse=True)
        return kept[: self.valves.TOP_K] + steps[: self.valves.CHAIN_TOP_K]

    def _explore(
        self,
        query: str,
        candidates: list[tuple],
        answered: bool = False,
        exclude: set = frozenset(),
        meter=None,
    ) -> list[tuple]:
        # Explorer mode: the concepts of the closest chunks (and graph neighbours
        # of those) are topics the index covers. Without an answer they are
        # offered instead; after an answer as further topics, with the chunks the
        # answer used (exclude) not offered again as excerpts.
        # Returns [(label, question)], each question verified by the grader
        # against the excerpt it was written from.
        pool = [chunk_id for chunk_id, _ in candidates]
        if not pool:
            # Nothing passed MIN_SIMILARITY or BM25: take the nearest chunks anyway.
            if meter:
                meter.add_embedding(self.valves.EMBEDDING_MODEL, [query])
            for doc, _ in self.vectorstore.similarity_search_with_score(query, k=self.valves.FETCH_K):
                if doc.metadata["chunk_id"] not in pool:
                    pool.append(doc.metadata["chunk_id"])

        scores = Counter()
        for rank, chunk_id in enumerate(pool):
            for concept in self.docs[chunk_id].metadata.get("concepts", []):
                scores[concept] += 1 / (rank + 1)
        for concept, score in scores.most_common(5):
            for neighbor, _ in self.concept_neighbors.get(concept, Counter()).most_common(5):
                scores[neighbor] += 0.3 * score

        topics, used = [], set(exclude)
        for concept, _ in scores.most_common():
            if len(topics) >= 3 * self.valves.EXPLORE_SUGGESTIONS:
                break
            if len(self.concept_chunks.get(concept, [])) < self.valves.EXPLORE_MIN_CHUNKS:
                continue
            # Excerpt: a nearby chunk on this concept, else its summary, else any chunk.
            options = (
                [c for c in pool if concept in self.docs[c].metadata.get("concepts", [])]
                + self.concept_summaries.get(concept, [])
                + self.concept_chunks[concept]
            )
            chunk_id = next((c for c in options if c not in used), None)
            if chunk_id is not None:
                used.add(chunk_id)
                topics.append((concept, self.docs[chunk_id]))
        if not topics:
            return []

        listing = "\n\n".join(
            f"{i}. {concept} - {doc.metadata['title']}\n{doc.page_content[:600]}"
            for i, (concept, doc) in enumerate(topics, start=1)
        )
        situation, goal = EXPLORE_ANSWERED if answered else EXPLORE_NO_ANSWER
        picked = self.explorer.invoke(
            EXPLORE_PROMPT.format(
                situation=situation,
                goal=goal,
                n=self.valves.EXPLORE_SUGGESTIONS,
                question=query,
                topics=listing,
            ),
            config=run_config(meter),
        ).suggestions
        seen, proposals = set(), []
        for s in picked:
            if 1 <= s.topic <= len(topics) and s.topic not in seen and s.question.strip():
                if same_question(s.question, query, self.bm25.idf):
                    print(f"[capital_rag] Explorer: skip rephrased question {s.question!r}")
                    continue
                seen.add(s.topic)
                proposals.append((s, topics[s.topic - 1][1]))
        proposals = proposals[: self.valves.EXPLORE_SUGGESTIONS]

        # Only suggest questions the pipeline can answer: grade each question
        # against its excerpt with the same bar as normal retrieval.
        grades = self.grader.batch(
            [
                GRADE_PROMPT.format(question=s.question, title=doc.metadata["title"], content=doc.page_content)
                for s, doc in proposals
            ],
            config=run_config(meter, max_concurrency=8),
            return_exceptions=True,
        )
        return [
            (s.label, s.question)
            for (s, _), grade in zip(proposals, grades)
            if not isinstance(grade, Exception)
            and grade.supports_answer
            and grade.relevance >= self.valves.MIN_RELEVANCE
        ]

    def _remember(self, query: str, candidates: list[tuple], exclude: set, suggestions) -> None:
        # suggestions None = not computed yet; _follow_ups computes them lazily,
        # in the background task after the answer is already shown.
        self.recent.pop(query, None)
        self.recent[query] = {
            "candidates": candidates,
            "exclude": exclude,
            "suggestions": suggestions,
        }
        while len(self.recent) > 50:
            self.recent.popitem(last=False)

    def _follow_ups(self, prompt: str) -> str:
        # Answers Open WebUI's follow-up task: it runs after every answer and
        # stores the result as clickable follow-up chips. The prompt contains the
        # chat history; the remembered question found latest in it is the one
        # just answered.
        query, entry, pos = None, None, -1
        for q, e in self.recent.items():
            p = prompt.rfind(q)
            if p > pos:
                query, entry, pos = q, e, p
        if entry is None:
            return json.dumps({"follow_ups": []})
        if entry["suggestions"] is None:
            entry["suggestions"] = self._explore(query, entry["candidates"], answered=True, exclude=entry["exclude"])
        return json.dumps({"follow_ups": [q for _, q in entry["suggestions"]]}, ensure_ascii=False)

    def _unsupported_claims(self, context: str, blocks: list[dict], answer: str, meter=None) -> tuple:
        # reliable_rag-style hallucination check, per claim: a claim only counts
        # as supported if the checker says so AND (for facts) its quote really is
        # in the context. Meta claims ("the sources don't explain why") need no quote.
        # Also returns whether the answer has any fact claim at all - an answer of
        # only meta claims ("the sources don't cover this") is a refusal - and the
        # answer sentences the problems were found in.
        check = self.checker.invoke(CHECK_PROMPT.format(context=context, answer=answer), config=run_config(meter))
        has_facts = any(claim.kind == "fact" for claim in check.claims)
        unsupported, sentences = [], []
        for claim in check.claims:
            has_evidence = claim.kind == "meta" or quote_in_context(claim.quote, context)
            if claim.supported and claim.same_cause and not has_evidence and joins_sentences(claim.quote, context):
                problem = (
                    "merges separate statements from the sources into one sentence: split it "
                    "into separate sentences, one per source statement, without linking them"
                )
            elif not (claim.supported and has_evidence and claim.same_cause):
                problem = "not stated in the sources"
            elif claim.kind == "fact" and (videos := required_videos(claim.quote, blocks)):
                if names_video(paragraph_of(answer, claim.answer_sentence), videos):
                    continue
                titles = " / ".join(f'"{title}"' for title, _ in videos)
                problem = f"missing source attribution: name the source {titles} in this paragraph"
            else:
                continue
            print(f"[capital_rag] {problem}: {claim.claim!r} (quote: {claim.quote!r})")
            unsupported.append(f"{claim.claim} ({problem})")
            sentences.append(claim.answer_sentence)
        return unsupported, has_facts, sentences

    def _corroborate(self, answer: str, context: str, blocks: list[dict], meter=None) -> tuple:
        # Cites further blocks in lines of the checked answer (a chain step is one line) when they state the
        # same. Accepted only if the line cites something already, the block is a concept block not yet cited
        # there, and the quote really is in that block. Returns the answer and the number added.
        lines = answer.split("\n")
        numbered = "\n".join(f"L{i}: {line}" for i, line in enumerate(lines, start=1) if line.strip())
        found = self.corroborator.invoke(
            CORROBORATE_PROMPT.format(context=context, answer=numbered), config=run_config(meter)
        )
        added = {}
        for item in found.items:
            if not (item.same_statement and 1 <= item.line <= len(lines) and 1 <= item.block <= len(blocks)):
                continue
            line, block = lines[item.line - 1], blocks[item.block - 1]
            new = added.get(item.line, set())
            cited = cited_numbers(line, len(blocks))
            if (
                not cited
                or item.block in cited | new
                or block["opinion"]
                or len(new) >= self.valves.CORROBORATE_MAX_PER_LINE
                or not quote_in_context(item.quote, block["text"])
            ):
                continue
            added.setdefault(item.line, set()).add(item.block)
        for number, blocks_added in added.items():
            # The new citations go right after the line's last citation group.
            line = lines[number - 1]
            last = list(re.finditer(r"\[\d+(?:\s*,\s*\d+)*\]", line))[-1]
            lines[number - 1] = (
                line[: last.end()] + "".join(f"[{n}]" for n in sorted(blocks_added)) + line[last.end() :]
            )
        return "\n".join(lines), sum(len(n) for n in added.values())

    def _grounded_answer(self, question: str, context: str, blocks: list[dict], meter=None) -> Iterator[tuple]:
        # Yields ("status", short, detail) progress steps and finally
        # ("answer", text or None, has_facts).
        yield ("status", "Formuliere Antwort nur aus den Quellen …", None)
        answer = self.answer_llm.invoke(
            ANSWER_PROMPT.format(context=context, question=question), config=run_config(meter)
        ).content
        for attempt in range(self.valves.MAX_REVISIONS + 1):
            yield ("status", "Prüfe jede Aussage gegen die Quellen …", None)
            unsupported, has_facts, sentences = self._unsupported_claims(context, blocks, answer, meter)
            if not unsupported:
                yield ("status", "Alle Aussagen sind durch die Quellen belegt ✓", None)
                yield ("answer", answer, has_facts)
                return
            yield (
                "status",
                f"{len(unsupported)} Aussage(n) nicht sauber belegt",
                f"{len(unsupported)} Aussage(n) nicht sauber belegt:\n" + "\n".join(f"  - {c}" for c in unsupported),
            )
            if attempt == self.valves.MAX_REVISIONS:
                break
            yield ("status", "Überarbeite die Antwort …", None)
            answer = self.reviser.invoke(
                REVISE_PROMPT.format(
                    context=context,
                    question=question,
                    answer=answer,
                    unsupported="\n".join(f"- {c}" for c in unsupported),
                ),
                config=run_config(meter),
            ).content
            # The prompt labels the draft "Answer:"; the model sometimes echoes it.
            answer = re.sub(r"^\s*\**(answer|antwort)\**\s*:\s*\**\s*", "", answer, flags=re.IGNORECASE)
        # Last resort instead of dropping the whole answer: cut the flagged
        # sentences and check what is left once more, with the same bar.
        trimmed = drop_sentences(answer, sentences)
        if trimmed and trimmed != answer:
            yield ("status", "Entferne nicht belegte Sätze und prüfe erneut …", None)
            unsupported, has_facts, _ = self._unsupported_claims(context, blocks, trimmed, meter)
            if not unsupported and has_facts:
                yield ("status", "Verbleibende Aussagen sind belegt ✓", None)
                yield ("answer", trimmed, has_facts)
                return
        yield ("status", "Keine vollständig belegte Antwort möglich.", None)
        yield ("answer", None, False)

    def pipe(self, user_message: str, model_id: str, messages: list[dict], body: dict) -> str | Generator | Iterator:
        # Einstieg für Open WebUI. Mit Langfuse wird jede Frage ein Trace (Frage, gezeigte Antwort, alle
        # LLM-Schritte darunter); Hintergrundaufgaben wie Titel und Tags werden nicht getraced.
        trace = None if user_message.startswith("### Task:") else self._start_trace(user_message, body)
        if trace is None:
            yield from self._pipe(user_message, model_id, messages, body)
            return
        shown, error = [], None
        try:
            for chunk in self._pipe(user_message, model_id, messages, body, trace[1]):
                if isinstance(chunk, str):
                    shown.append(chunk)
                yield chunk
        except BaseException as e:  # incl. GeneratorExit when the chat is closed mid-answer
            error = e
            raise
        finally:
            self._end_trace(trace, "".join(shown), error)

    def _pipe(self, user_message: str, model_id: str, messages: list[dict], body: dict, trace_handler=None) -> Iterator:
        # Beantwortet eine Frage (oder eine Hintergrundaufgabe) und liefert Status-Events und Antworttext.
        if user_message.startswith("### Task:"):
            # Open WebUI background task, not a question: follow-ups come from the
            # explorer, everything else (title, tags) from the plain LLM.
            try:
                if FOLLOW_UP_TASK.match(user_message):
                    yield self._follow_ups(user_message) if self.valves.EXPLORE_MODE else "{}"
                else:
                    yield self.answer_llm.invoke(user_message).content
            except Exception as e:  # noqa: BLE001 - background task, Open WebUI gets an empty result
                print(f"[capital_rag] Task failed: {e}")
                yield "{}"
            return

        if self.vectorstore is None:
            yield (
                f"Der Kapitalmarkt-Index fehlt in {self.valves.INDEX_DIR}. Erstelle ihn mit\n\n"
                "`docker exec -it open-webui-pipelines-capital python /data/ingest_capital_chunks.py`\n\n"
                "und starte danach den Container pipelines-capital neu."
            )
            return

        # Progress is sent as Open WebUI status events: a chunk with an "event"
        # key is forwarded to the chat's event emitter, so each step replaces the
        # previous one in a single line above the answer. Only when streaming -
        # a non-streamed response is joined as text. SHOW_THINKING_LOG also writes
        # the full step log (incl. rejected claims) into a collapsible <think> block.
        streaming = body.get("stream", True)
        log_open = False
        # Cost display: tokens and cost of each step, then the question's total (SHOW_COSTS).
        # The meter also carries the Langfuse handler to every LLM call (run_config), so it exists when
        # tracing is on even with the cost display off.
        show_costs = self.valves.SHOW_COSTS
        meter = UsageMeter(json.loads(self.valves.MODEL_PRICES)) if show_costs or trace_handler else None
        if trace_handler:
            meter.callbacks.append(trace_handler)
        open_step = None

        def status(text: str, done: bool = False) -> dict:
            # Ein Open-WebUI-Status-Event, also eine Zeile in der grauen Statuszeile.
            return {"event": {"type": "status", "data": {"description": text, "done": done}}}

        def cost_line() -> list:
            # Schließt den laufenden Schritt ab: wiederholt dessen Text mit Tokens und Kosten seit
            # dem letzten Schritt. Ohne Verbrauch, Zähler oder Streaming entsteht keine Zeile.
            usage = meter.take() if show_costs else {}
            if not (streaming and usage and open_step):
                return []
            return [status(f"{open_step.rstrip(' …')} — {meter.describe(usage)}")]

        def summary() -> list:
            # Letzte Zeile der Frage: offenen Schritt abschließen, dann die Summe über alle Modelle.
            if not (show_costs and streaming):
                return []
            return [*cost_line(), status(f"Σ Frage gesamt: {meter.describe(meter.total, per_model=False)}", done=True)]

        def step(short: str, detail: str | None = None, done: bool = False) -> list:
            # Sendet einen neuen Fortschrittsschritt; vorher die Kostenzeile des vorherigen Schritts.
            nonlocal log_open, open_step
            out = []
            if streaming:
                out += cost_line()
                out.append(status(short, done))
            open_step = short
            if self.valves.SHOW_THINKING_LOG:
                if not log_open:
                    out.append("<think>\n")
                    log_open = True
                out.append(f"{detail or short}\n\n")
            return out

        def close_log() -> list:
            return ["</think>\n\n"] if log_open else []

        def find_related(reason: str, exclude: set = frozenset()) -> Generator:
            # Yields status steps; returns the explorer suggestions.
            if not self.valves.EXPLORE_MODE:
                return []
            yield from step(f"{reason} – suche verwandte Themen …")
            try:
                return self._explore(user_message, candidates, exclude=exclude, meter=meter)
            except Exception as e:  # noqa: BLE001 - suggestions are optional
                print(f"[capital_rag] Explorer failed: {e}")
                return []

        def related_block(suggestions) -> list:
            if not suggestions:
                return []
            out = [
                "\n\n**Verwandte Themen, zu denen es Material gibt:**\n"
                + "\n".join(f"- **{label}**: {question}" for label, question in suggestions)
            ]
            if streaming:
                # Shown as clickable follow-up chips under the message.
                out.append(
                    {
                        "event": {
                            "type": "chat:message:follow_ups",
                            "data": {"follow_ups": [q for _, q in suggestions]},
                        }
                    }
                )
            return out

        def no_answer(reason: str, exclude: set = frozenset()) -> Iterator:
            suggestions = yield from find_related(reason, exclude)
            self._remember(user_message, candidates, set(exclude), suggestions)
            if suggestions:
                reason = f"{reason} – {len(suggestions)} verwandte Themen gefunden"
            yield from step(reason, done=True)
            yield from summary()
            yield from close_log()
            yield NO_ANSWER
            yield from related_block(suggestions)

        yield from step("Durchsuche die Quellen …")
        candidates = self._fusion_candidates(user_message, meter)
        yield from step(f"{len(candidates)} Kandidaten gefunden")
        expanded = self._graph_expand(candidates)
        if len(expanded) > len(candidates):
            yield from step(f"Konzeptgraph: {len(expanded) - len(candidates)} verwandte Abschnitte ergänzt")
        candidates = expanded
        paths, path_ids = [], frozenset()
        if self.valves.CHAIN_SEARCH and self.concept_vectors is not None:
            yield from step("Suche Wirkungsketten im Konzeptgraphen …")
            try:
                paths = self._causal_paths(user_message, meter)
            except Exception as e:  # noqa: BLE001 - the answer continues without chain paths
                print(f"[capital_rag] Chain search failed: {e}")
            if paths:
                present = {chunk_id for chunk_id, _ in candidates}
                path_ids = frozenset(self._path_chunks(paths, present))
                candidates = candidates + [(c, 0.0) for c in path_ids if c not in present]
                yield from step(
                    f"{len(paths)} Wirkungskette(n) mit bis zu {max(len(p) - 1 for p in paths)} Schritten gefunden",
                    "Wirkungsketten:\n" + "\n".join(f"  - {' → '.join(p)}" for p in paths),
                )
        grounded = []
        if candidates:
            yield from step(f"Bewerte die Relevanz von {len(candidates)} Abschnitten …")
            grounded = self._grade(user_message, candidates, path_ids, meter)

        if not grounded:
            yield from no_answer("Keine relevanten Abschnitte gefunden")
            return
        yield from step(
            f"{len(grounded)} relevante Abschnitte ausgewählt",
            f"{len(grounded)} relevante Abschnitte ausgewählt:\n"
            + "\n".join(f"  - {doc.metadata['title']}" for doc, _, _ in grounded),
        )

        blocks = []
        for i, (doc, _, _) in enumerate(grounded, start=1):
            meta = doc.metadata
            videos = "; ".join(sorted({cite_source(s) for s in meta["sources"]}))
            timeless = []
            if meta.get("content_class") == "opinion":
                label = f"OPINION (time-bound, stated in: {videos})"
            else:
                label = f"CONCEPT (from: {videos})"
            block = f"[{i}] {meta['title']} - {label}\n{doc.page_content}"
            for rel in meta.get("relations", []):
                rel_label = "TIMELESS" if rel.get("relation_class") == "timeless" else "TIME-BOUND"
                block += (
                    f"\nRelation [{rel_label}]: {rel['source']} -> {rel['target']} "
                    f"({rel['direction']}, {rel['time_horizon']}): {rel['mechanism']}"
                )
                if rel.get("conditions"):
                    block += f" Conditions: {rel['conditions']}"
                if rel_label == "TIMELESS":
                    timeless.append(rel["mechanism"])
            if meta.get("conditions"):
                block += f"\nConditions: {meta['conditions']}"
            blocks.append(
                {
                    "text": block,
                    "opinion": meta.get("content_class") == "opinion",
                    "timeless": timeless,
                    "videos": sorted({parse_video(s) for s in meta["sources"]}),
                }
            )
        context = "\n\n".join(b["text"] for b in blocks)
        # Causal paths whose every step is stated by a selected block, with those block numbers.
        path_lines = []
        for path in paths:
            refs = []
            for a, b in pairwise(path):
                nums = [
                    i
                    for i, (doc, _, _) in enumerate(grounded, start=1)
                    if any(r["source"] == a and r["target"] == b for r in doc.metadata.get("relations", []))
                ]
                refs.append(nums)
            if all(refs):
                line = path[0] + "".join(f" -> {b} [{', '.join(map(str, nums))}]" for b, nums in zip(path[1:], refs))
                path_lines.append(f"- {line}")
        if path_lines:
            context = (
                "Causal paths (relations found across the sources; [n] = block that states the step):\n"
                + "\n".join(path_lines)
                + "\n\n"
                + context
            )

        # None = follow-ups are computed later by the follow-up task (_follow_ups).
        suggestions = None
        if self.valves.CHECK_ANSWER_GROUNDING:
            # Never show an unchecked answer: if the check itself fails, refuse.
            answer, has_facts = None, False
            try:
                for kind, value, detail in self._grounded_answer(user_message, context, blocks, meter):
                    if kind == "status":
                        yield from step(value, detail)
                    else:
                        answer, has_facts = value, detail
            except Exception as e:  # noqa: BLE001 - fail closed: no checked answer, no answer
                print(f"[capital_rag] Grounding check failed: {e}")
                yield from step("Die Prüfung ist fehlgeschlagen")
            if answer is None:
                # The answer failed on these chunks, so don't offer them again.
                failed = {doc.metadata["chunk_id"] for doc, _, _ in grounded}
                yield from no_answer("Keine belegte Antwort gefunden", failed)
                return
            if not has_facts:
                # The answer only says the sources don't cover the question.
                suggestions = yield from find_related("Die Quellen beantworten die Frage nicht")
            elif self.valves.CORROBORATE and len(blocks) > 1:
                yield from step("Suche weitere Belege in den anderen Quellen …")
                try:
                    answer, added = self._corroborate(answer, context, blocks, meter)
                    if added:
                        yield from step(f"{added} weitere Beleg(e) aus anderen Quellen ergänzt")
                except Exception as e:  # noqa: BLE001
                    # Only adds citations: on failure the checked answer stays as it is.
                    print(f"[capital_rag] Corroboration failed: {e}")
            cited = cited_numbers(answer, len(grounded))
            yield from step(f"Antwort geprüft ✓ ({len(cited)} Quellen)", done=True)
            yield from summary()
            yield from close_log()
            yield answer
        else:
            yield from step(f"Antwort aus {len(grounded)} Quellen (ungeprüft)", done=True)
            yield from close_log()
            answer = ""
            for chunk in self.llm.stream(
                ANSWER_PROMPT.format(context=context, question=user_message), config=run_config(meter)
            ):
                if chunk.content:
                    answer += chunk.content
                    yield chunk.content
            cited = cited_numbers(answer, len(grounded))
            # After the stream, so the streamed answer is counted too.
            yield from summary()

        # Only the sources the answer actually cites, under their citation numbers.
        source_lines = []
        for i, (doc, grade, _) in enumerate(grounded, start=1):
            if i not in cited:
                continue
            meta = doc.metadata
            if meta["chunk_type"] == "summary":
                kind = f"Konzept, Zusammenfassung aus {meta['support']} Abschnitten"
            elif meta.get("content_class") == "opinion":
                kind = "Meinung (zeitgebunden)"
            else:
                kind = "Konzept"
            videos = "; ".join(sorted({cite_source(s) for s in meta["sources"]}))
            source_lines.append(
                f"- [{i}] {meta['title']} ({kind}) - {videos}\n  Relevanz {grade.relevance}/10: {grade.reason}"
            )
        if source_lines:
            yield "\n\nQuellen:\n" + "\n".join(source_lines)
        yield from related_block(suggestions)
        used = {grounded[i - 1][0].metadata["chunk_id"] for i in cited}
        self._remember(user_message, candidates, used, suggestions)

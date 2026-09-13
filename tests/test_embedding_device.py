"""Where the embedding model runs, and what happens when a card says no.

`EMBEDDING_DEVICE` names a device and nothing trusts it. A card is proven with
a real encode; a refusal leaves the model on the CPU with the reason recorded;
and on a card shared with a local seat the embedder is placed before the seat
loads, the one arrangement measured to leave the seat all 49 layers on the GPU.
Every test answers with a fake `sentence_transformers` and a fake Ollama, so
none loads a model, touches a card or reaches a daemon.
"""

from __future__ import annotations

import io
import json
import sys
import types
from typing import Any

import numpy as np
import pytest

import serve
from langgraph_agent import config
from langgraph_agent import graphrag_server as gs
from langgraph_agent.graphrag_server import (
    CHUNK_MAX_TOKENS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    GraphRAGKnowledgeBase,
    embedding_device_status,
)

KERNELS_MISSING = "CUDA error: no kernel image is available for execution on the device"


class OutOfMemoryError(RuntimeError):
    """Named like torch's, which is how a full card is told from a broken one."""


class _Library:
    """`sentence_transformers`, recording every model built and every encode.

    `refusals[device]` is a queue of exceptions that device's encodes raise,
    one per call, so a card can refuse once and then take the model.
    """

    def __init__(self) -> None:
        self.built: list[dict[str, Any]] = []
        self.encoded: list[dict[str, Any]] = []
        self.refusals: dict[str, list[BaseException]] = {}

    def module(self) -> types.SimpleNamespace:
        library = self

        class SentenceTransformer:
            def __init__(self, name: str, **kwargs: Any) -> None:
                library.built.append({"name": name, **kwargs})
                self.device = kwargs.get("device")

            def encode(self, texts: str | list[str], **kwargs: Any) -> Any:
                library.encoded.append({"device": self.device, "texts": texts, **kwargs})
                pending = library.refusals.get(self.device) or []
                if pending:
                    raise pending.pop(0)
                return np.zeros((len(texts), 3)) if isinstance(texts, list) else np.zeros(3)

        return types.SimpleNamespace(SentenceTransformer=SentenceTransformer)

    def devices(self) -> list[str | None]:
        return [call.get("device") for call in self.built]


@pytest.fixture
def library(monkeypatch) -> _Library:
    lib = _Library()
    monkeypatch.setitem(sys.modules, "sentence_transformers", lib.module())
    return lib


@pytest.fixture
def card(monkeypatch) -> str:
    monkeypatch.setattr(gs, "EMBEDDING_DEVICE", "cuda:1")
    return "cuda:1"


def _kb() -> GraphRAGKnowledgeBase:
    """No store: placing the embedder touches none of it."""
    return object.__new__(GraphRAGKnowledgeBase)


# ---------------------------------------------------------------------------
# the device is named, and a card is proven
# ---------------------------------------------------------------------------


def test_the_device_is_named_on_every_construction(library):
    """Never left to sentence-transformers' own choice.

    It picks `cuda:0` whenever `torch.cuda.is_available()` is True, and that
    stays True on a build with no kernels for the card: a `+cu130` torch on a
    compute-6.1 GTX 1060 made every `encode()` raise.
    """
    kb = _kb()

    kb.embedder  # noqa: B018 - the property load is what is under test

    assert library.built == [
        {"name": EMBEDDING_MODEL_NAME, "device": "cpu", "local_files_only": True}
    ]
    assert kb.embedding_device == "cpu"
    assert kb.embedding_device_note is None


def test_a_card_without_kernels_leaves_the_model_on_the_cpu_and_says_why(library, card):
    """Construction succeeds on such a build; only an encode fails."""
    library.refusals[card] = [RuntimeError(KERNELS_MISSING)]
    kb = _kb()

    kb.embedder  # noqa: B018

    assert library.devices() == [card, "cpu"]
    assert kb.embedding_device == "cpu"
    assert "no kernel image" in (kb.embedding_device_note or "")


def test_a_setting_that_names_no_device_is_reported_rather_than_guessed(monkeypatch, library):
    monkeypatch.setattr(gs, "EMBEDDING_DEVICE", "gpu")
    kb = _kb()

    kb.embedder  # noqa: B018

    assert library.devices() == ["cpu"]
    assert "'gpu'" in (kb.embedding_device_note or "")


def test_the_warm_up_reaches_the_full_window_at_several_lengths(library, card):
    """The warm-up is what a local seat gets fitted around.

    A warm-up shorter than the window proves less than an index will ask of the
    card, and a single shape under-reserves: 148 MiB against the 166 a real
    index peaked at.
    """
    _kb().embedder  # noqa: B018

    (warm_up,) = [call for call in library.encoded if call["device"] == card]
    assert warm_up["batch_size"] == EMBEDDING_BATCH_SIZE
    words = [len(text.split()) for text in warm_up["texts"]]
    assert max(words) == CHUNK_MAX_TOKENS
    assert words.count(CHUNK_MAX_TOKENS) >= EMBEDDING_BATCH_SIZE
    assert len(set(words)) > 1


def test_cpu_hides_every_card_and_a_card_is_numbered_by_its_bus():
    hidden: dict[str, str] = {}
    gs._prepare_cuda_environment("cpu", hidden)
    assert hidden == {"CUDA_VISIBLE_DEVICES": ""}

    numbered: dict[str, str] = {}
    gs._prepare_cuda_environment("cuda:1", numbered)
    assert numbered == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

    chosen = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    gs._prepare_cuda_environment("cuda:1", chosen)
    assert chosen == {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}


# ---------------------------------------------------------------------------
# placing it before a local seat loads
# ---------------------------------------------------------------------------


def test_the_cpu_setting_places_nothing_and_loads_nothing(library):
    kb = _kb()

    report = kb.claim_embedding_device(lambda: pytest.fail("nothing to make room for"))

    assert report == {"source": "cpu"}
    assert kb._embedder is None
    assert library.built == []


def test_a_full_card_is_asked_again_once_a_seat_makes_room(library, card):
    library.refusals[card] = [OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")]
    asked: list[bool] = []

    def make_room() -> list[str]:
        asked.append(True)
        return ["dolphin-9b:Q4_K_M"]

    kb = _kb()
    report = kb.claim_embedding_device(make_room)

    assert asked == [True]
    assert report["source"] == "claimed"
    assert report["device"] == card
    assert report["unloaded"] == ["dolphin-9b:Q4_K_M"]
    assert library.devices() == [card, card]
    assert kb.embedding_device == card


def test_a_card_that_stays_full_is_tried_again_next_run(library, card):
    """Nothing seated to unload: the CPU this run, and the card again next run."""
    library.refusals[card] = [OutOfMemoryError("CUDA out of memory")]
    kb = _kb()

    first = kb.claim_embedding_device(lambda: [])
    assert first["source"] == "unavailable"
    assert first["retryable"] is True
    assert kb.embedding_device == "cpu"

    second = kb.claim_embedding_device(lambda: [])
    assert second["source"] == "claimed"
    assert library.devices() == [card, "cpu", card]


def test_a_card_without_kernels_is_not_asked_again_every_run(library, card):
    """Asking again would unload a seat on every run for nothing."""
    library.refusals[card] = [RuntimeError(KERNELS_MISSING), RuntimeError(KERNELS_MISSING)]
    kb = _kb()
    asked: list[bool] = []

    def make_room() -> list[str]:
        asked.append(True)
        return []

    kb.claim_embedding_device(make_room)
    report = kb.claim_embedding_device(make_room)

    assert report["source"] == "unavailable"
    assert report["retryable"] is False
    assert asked == []
    assert library.devices() == [card, "cpu"]


def test_an_embedder_already_on_its_card_is_left_there(library, card):
    kb = _kb()
    kb.claim_embedding_device(lambda: [])

    report = kb.claim_embedding_device(lambda: pytest.fail("already placed"))

    assert report == {"source": "resident", "device": card}
    assert library.devices() == [card]


# ---------------------------------------------------------------------------
# a card that fills afterwards
# ---------------------------------------------------------------------------


def test_a_card_that_fills_mid_index_finishes_the_encode_on_the_cpu(library, card):
    """Raised, the error would skip the document and the rebuild would report success."""
    kb = _kb()
    kb.embedder  # noqa: B018 - placed on the card
    library.refusals[card] = [OutOfMemoryError("CUDA out of memory")]

    vectors = kb._encode(["a passage", "another passage"])

    assert vectors.shape == (2, 3)
    assert kb.embedding_device == "cpu"
    assert "out of memory" in (kb.embedding_device_note or "")
    assert library.encoded[-1]["device"] == "cpu"
    # ...and the next run's placement asks the card again.
    assert kb.claim_embedding_device(lambda: [])["source"] == "claimed"


def test_an_error_that_is_not_memory_still_raises(library, card):
    kb = _kb()
    kb.embedder  # noqa: B018
    library.refusals[card] = [ValueError("a bug, not a full card")]

    with pytest.raises(ValueError):
        kb._encode(["a passage"])
    assert kb.embedding_device == card


def test_the_status_says_where_without_loading_anything(monkeypatch, library, card):
    kb = _kb()
    monkeypatch.setattr(gs, "_kb_instance", kb)

    assert embedding_device_status() == {"configured": card, "active": None, "note": None}
    assert library.built == []

    kb.embedder  # noqa: B018
    assert embedding_device_status()["active"] == card


# ---------------------------------------------------------------------------
# making room: which models are unloaded
# ---------------------------------------------------------------------------


class _Daemon:
    """The two calls unloading makes, `/api/ps` and `/api/generate`."""

    def __init__(self, loaded: list[dict[str, Any]]) -> None:
        self.loaded = loaded
        self.unload_requests: list[dict[str, Any]] = []

    def urlopen(self, request: Any, timeout: float | None = None) -> io.BytesIO:
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/api/ps"):
            return io.BytesIO(json.dumps({"models": self.loaded}).encode())
        payload = json.loads(request.data)
        self.unload_requests.append(payload)
        self.loaded = [model for model in self.loaded if model["name"] != payload["model"]]
        return io.BytesIO(json.dumps({"done": True, "done_reason": "unload"}).encode())


def _seats(monkeypatch, **chosen: tuple[str, str]) -> None:
    seats = {agent: {"provider": "anthropic", "model": "claude-opus-5"} for agent in config.AGENTS}
    seats.update({agent: {"provider": p, "model": m} for agent, (p, m) in chosen.items()})
    monkeypatch.setattr(config, "_resolve_seat", lambda agent: seats[agent])


def test_only_seated_models_holding_gpu_memory_are_unloaded(monkeypatch):
    _seats(
        monkeypatch,
        architect=("ollama", "qwen3.5:397b-cloud"),
        planner=("ollama", "dolphin-9b:Q4_K_M"),
        researcher=("ollama", "dolphin-9b:Q4_K_M"),
    )
    daemon = _Daemon([
        {"name": "dolphin-9b:Q4_K_M", "size_vram": 5_600_000_000},
        {"name": "qwen3.5:397b-cloud", "size_vram": 0},
        {"name": "someone-elses:latest", "size_vram": 900_000_000},
    ])
    monkeypatch.setattr(config.urllib.request, "urlopen", daemon.urlopen)

    assert config.unload_local_seat_models(timeout=1.0) == ["dolphin-9b:Q4_K_M"]
    assert daemon.unload_requests == [{"model": "dolphin-9b:Q4_K_M", "keep_alive": 0}]


def test_a_seat_named_without_a_tag_is_the_daemons_latest(monkeypatch):
    _seats(monkeypatch, planner=("ollama", "qwen3.8"))
    daemon = _Daemon([{"name": "qwen3.8:latest", "size_vram": 17_000_000_000}])
    monkeypatch.setattr(config.urllib.request, "urlopen", daemon.urlopen)

    assert config.unload_local_seat_models(timeout=1.0) == ["qwen3.8:latest"]


def test_an_unreachable_daemon_unloads_nothing(monkeypatch):
    _seats(monkeypatch, planner=("ollama", "dolphin-9b:Q4_K_M"))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(config.urllib.request, "urlopen", refuse)

    assert config.unload_local_seat_models(timeout=0.5) == []


# ---------------------------------------------------------------------------
# the run places it first
# ---------------------------------------------------------------------------


class _Graph:
    def __init__(self) -> None:
        self.saw: list[str] = []

    def stream(self, state: dict[str, Any], run_config: Any):
        self.saw = list(state.get("messages", []))
        yield {"architect": {**state, "messages": [*state.get("messages", []), "[Architect] Verdict: approved"]}}


def test_the_embedder_is_placed_before_the_corpus_phase(monkeypatch):
    """The corpus phase embeds, and a local seat must be fitted around the embedder."""
    order: list[str] = []

    def place() -> dict[str, Any]:
        order.append("embedder")
        return {"source": "claimed", "device": "cuda:1", "unloaded": []}

    def index() -> dict[str, Any]:
        order.append("corpus")
        return {"source": "disabled"}

    graph = _Graph()
    monkeypatch.setattr(serve, "_claim_the_embedder_before_the_run", place)
    monkeypatch.setattr(serve, "_index_the_project_before_the_run", index)
    monkeypatch.setattr(serve, "graph", graph)

    result = serve.rpc_run_goal({"goal": "g"})

    assert order == ["embedder", "corpus"]
    assert any(line.startswith("[Embedder] Loaded onto cuda:1") for line in graph.saw)
    assert result["embedder"]["device"] == "cuda:1"


def test_placing_a_cpu_embedder_opens_nothing(monkeypatch):
    monkeypatch.setattr(serve, "_open_kb", lambda: pytest.fail("a CPU embedder has no order to get right"))

    assert serve._claim_the_embedder_before_the_run() == {"source": "cpu"}


def test_the_feed_names_a_seat_unloaded_to_make_room():
    line = serve._embedder_feed_line(
        {"source": "claimed", "device": "cuda:1", "unloaded": ["dolphin-9b:Q4_K_M"]}
    )

    assert line is not None
    assert "cuda:1" in line
    assert "dolphin-9b:Q4_K_M" in line


def test_a_refused_card_is_said_out_loud():
    line = serve._embedder_feed_line({
        "source": "unavailable",
        "configured": "cuda:1",
        "device": "cpu",
        "note": f"cuda:1 refused the embedding model: {KERNELS_MISSING}",
    })

    assert line is not None
    assert "CPU" in line
    assert "no kernel image" in line


@pytest.mark.parametrize("source", ["cpu", "resident", "no_corpus", "stopped"])
def test_the_feed_is_silent_when_placing_did_nothing(source):
    assert serve._embedder_feed_line({"source": source}) is None

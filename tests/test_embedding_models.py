"""Switching the embedding model: its own corpus, its own backend, its own floor.

The console's embedding dropdown switches models, and a model is more than a
name here. Vectors from two models share no space, so each model has its own
store; MiniLM runs in this process while every other model is an Ollama tag;
and the score that decides whether retrieval answered was measured on MiniLM
alone, so any other model's floor is measured on its own corpus or does not
exist. Every test answers with fakes: no model loads and no daemon is asked.
"""

from __future__ import annotations

import io
import json
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

import serve
from langgraph_agent import graphrag_server as gs
from langgraph_agent import nodes

QWEN = "qwen3-embedding:latest"


@pytest.fixture
def qwen(monkeypatch) -> str:
    monkeypatch.setattr(gs, "_kb_instance", None)
    monkeypatch.setattr(gs, "_embedding_model_override", QWEN)
    return QWEN


# ---------------------------------------------------------------------------
# which model, and where its corpus lives
# ---------------------------------------------------------------------------


def test_minilm_is_the_default_and_keeps_its_directory():
    assert gs.active_embedding_model() == gs.EMBEDDING_MODEL_NAME
    assert gs.persist_dir_for(gs.EMBEDDING_MODEL_NAME) == "./knowledge"
    assert gs.embedding_backend(gs.EMBEDDING_MODEL_NAME) == "sentence-transformers"


def test_another_model_gets_a_directory_the_walk_already_excludes():
    """So no model's corpus is ever indexed into another's."""
    path = gs.persist_dir_for(QWEN)

    assert path == "./knowledge/models/qwen3-embedding-latest"
    assert gs.embedding_backend(QWEN) == "ollama"
    assert "knowledge/" in gs.PROJECT_INDEX_EXCLUDES
    assert path.startswith("./knowledge/")


def test_switching_drops_the_open_corpus(monkeypatch):
    monkeypatch.setattr(gs, "_kb_instance", object())

    gs.set_embedding_model(QWEN)

    assert gs.active_embedding_model() == QWEN
    assert gs._kb_instance is None
    gs.set_embedding_model(gs.EMBEDDING_MODEL_NAME)
    assert gs._embedding_model_override is None


def test_the_doors_follow_the_active_model(tmp_path, monkeypatch, qwen):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "knowledge" / "chroma").mkdir(parents=True)  # MiniLM's corpus...

    assert gs.corpus_state() == ("absent", qwen)  # ...says nothing about qwen's
    assert gs.corpus_exists() is False
    assert gs.open_knowledge_base() is None


# ---------------------------------------------------------------------------
# the Ollama backend
# ---------------------------------------------------------------------------


class _Daemon:
    """`/api/embed`, answering each passage with a vector of its own length."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.refusal: str | None = None

    def urlopen(self, request: Any, timeout: float | None = None) -> io.BytesIO:
        assert request.full_url.endswith("/api/embed")
        body = json.loads(request.data)
        if self.refusal is not None:
            raise urllib.error.HTTPError(
                request.full_url, 500, "refused", {}, io.BytesIO(self.refusal.encode())  # type: ignore[arg-type]
            )
        self.batches.append(body["input"])
        vectors = [[float(len(text))] * 4 for text in body["input"]]
        return io.BytesIO(json.dumps({"model": body["model"], "embeddings": vectors}).encode())


@pytest.fixture
def daemon(monkeypatch) -> _Daemon:
    fake = _Daemon()
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    return fake


def test_passages_go_to_the_daemon_in_batches_and_come_back_in_order(daemon):
    texts = [f"passage {'x' * i}" for i in range(19)]

    vectors = gs.OllamaEmbedder(QWEN).encode(texts)

    assert [len(batch) for batch in daemon.batches] == [8, 8, 3]
    assert vectors.shape == (19, 4)
    assert [row[0] for row in vectors] == [float(len(text)) for text in texts]


def test_a_query_comes_back_as_one_vector(daemon):
    assert gs.OllamaEmbedder(QWEN).encode("one query").shape == (4,)


def test_a_daemon_refusal_names_the_model_and_says_why(daemon):
    daemon.refusal = '{"error":"model requires more system memory"}'

    with pytest.raises(RuntimeError, match="qwen3-embedding:latest.*more system memory"):
        gs.OllamaEmbedder(QWEN).encode(["a passage"])


def test_a_stop_is_honoured_between_batches(daemon):
    asked = {"times": 0}

    def should_stop() -> bool:
        asked["times"] += 1
        return asked["times"] > 1

    with pytest.raises(gs.EmbeddingStopped):
        gs.OllamaEmbedder(QWEN).encode(["p"] * 20, should_stop=should_stop)
    assert len(daemon.batches) == 1


def test_an_ollama_model_chunks_with_minilms_tokenizer(monkeypatch):
    """So every model's corpus holds the same passages."""
    import transformers

    loaded: list[tuple[str, dict[str, Any]]] = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(name: str, **kwargs: Any) -> str:
            loaded.append((name, kwargs))
            return "minilm-tokenizer"

    monkeypatch.setattr(transformers, "AutoTokenizer", _AutoTokenizer)

    assert gs.OllamaEmbedder(QWEN).tokenizer == "minilm-tokenizer"
    assert loaded == [(f"sentence-transformers/{gs.EMBEDDING_MODEL_NAME}", {"local_files_only": True})]


def _corpus_on(model: str) -> gs.GraphRAGKnowledgeBase:
    kb = object.__new__(gs.GraphRAGKnowledgeBase)
    kb.embedding_model = model
    return kb


def test_a_corpus_on_an_ollama_model_is_placed_by_the_daemon(qwen):
    kb = _corpus_on(qwen)

    assert kb.claim_embedding_device(lambda: pytest.fail("the daemon places it")) == {
        "source": "ollama",
        "model": qwen,
    }
    assert isinstance(kb.embedder, gs.OllamaEmbedder)
    assert kb.embedding_device == "ollama"


def test_an_ollama_failure_never_falls_back_to_minilm(daemon, qwen):
    """MiniLM's vectors in another model's corpus would be noise that still scored."""
    daemon.refusal = "CUDA error: out of memory"
    kb = _corpus_on(qwen)

    with pytest.raises(RuntimeError):
        kb._encode(["a passage"])
    assert kb.embedding_device == "ollama"


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------


def test_minilms_floor_is_the_hand_measured_constant():
    assert gs.relevance_floor() == gs.RETRIEVAL_RELEVANCE_FLOOR


def test_another_model_has_no_floor_until_one_is_measured(tmp_path, monkeypatch, qwen):
    monkeypatch.chdir(tmp_path)

    assert gs.relevance_floor() is None


class _ScoredCorpus:
    """A corpus whose searches answer every calibration question with one score."""

    def __init__(self, persist_dir: Path, answered: float, unanswerable: float) -> None:
        self.persist_dir = persist_dir
        self.embedding_model = QWEN
        questions = json.loads(gs.FLOOR_CALIBRATION_QUESTIONS.read_text(encoding="utf-8"))
        self.scores = dict.fromkeys(questions["answered"], answered)
        self.scores.update(dict.fromkeys(questions["unanswerable"], unanswerable))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [{"score": self.scores[query]}]


def test_a_floor_is_the_middle_of_the_gap_and_is_kept_beside_the_corpus(tmp_path, monkeypatch, qwen):
    monkeypatch.chdir(tmp_path)
    corpus = _ScoredCorpus(Path(gs.persist_dir_for(qwen)), answered=0.62, unanswerable=0.40)

    record = gs.calibrate_relevance_floor(corpus)  # type: ignore[arg-type]

    assert record["floor"] == 0.51
    assert gs.relevance_floor() == 0.51
    assert (gs.floor_calibration() or {}).get("model") == qwen


def test_overlapping_populations_leave_no_floor(tmp_path, monkeypatch, qwen):
    """Any number inside the overlap would misfile some question, and nothing would say which."""
    monkeypatch.chdir(tmp_path)
    corpus = _ScoredCorpus(Path(gs.persist_dir_for(qwen)), answered=0.40, unanswerable=0.45)

    assert gs.calibrate_relevance_floor(corpus)["floor"] is None  # type: ignore[arg-type]
    assert gs.relevance_floor() is None
    assert gs.floor_calibration() is not None


def test_the_calibration_questions_are_never_in_the_corpus():
    """Indexed anywhere, the unanswerable questions would be answered by their own text."""
    questions = json.loads(gs.FLOOR_CALIBRATION_QUESTIONS.read_text(encoding="utf-8"))
    assert len(questions["answered"]) == 12
    assert len(questions["unanswerable"]) == 12

    root = Path(serve.__file__).resolve().parent
    walked = [
        (Path(p) if Path(p).is_absolute() else root / p).resolve()
        for p in gs.iter_project_files(str(root))
    ]
    assert gs.FLOOR_CALIBRATION_QUESTIONS.resolve() not in walked
    for path in walked:
        text = path.read_text(encoding="utf-8", errors="replace")
        quoted = [q for q in questions["unanswerable"] if q in text]
        assert not quoted, f"{path} quotes calibration questions: {quoted}"


# ---------------------------------------------------------------------------
# the nodes read the model's floor
# ---------------------------------------------------------------------------


_HIT = {"score": 0.99, "id": "nodes.py", "content": "text", "metadata": {"path": "nodes.py"}}


def test_a_model_without_a_floor_gives_the_planner_no_map(monkeypatch):
    monkeypatch.setattr(gs, "relevance_floor", lambda model=None: None)
    monkeypatch.setattr(nodes, "_call_mcp_tool_sync", lambda name, args: {"results": [_HIT]})

    assert nodes._project_map("make the embedder faster") == ""


def test_a_model_without_a_floor_counts_no_search_as_answered(monkeypatch):
    monkeypatch.setattr(gs, "relevance_floor", lambda model=None: None)
    monkeypatch.setattr(nodes, "_call_mcp_tool_sync", lambda name, args: {"results": [_HIT]})
    state = {
        "goal": "g", "architecture": "", "verdict": "", "messages": [], "plan": "a plan",
        "research": "", "builder_report": "", "next_agent": "", "research_status": "",
        "blockers": "", "files_changed": [], "failed_verification": [], "unverified": [],
        "builder_cut_off": "", "lint_failed": [], "expect_failures": False,
        "discuss_only": False, "step_count": 0,
    }

    findings, _ = nodes._gather_research(state)  # type: ignore[arg-type]

    assert "relevant documents in knowledge base" not in findings


# ---------------------------------------------------------------------------
# a long build says how far it got, and stops
# ---------------------------------------------------------------------------


class _Corpus:
    """Enough of a knowledge base for `index_project_files` to walk into."""

    def __init__(self, stop_on: int | None = None) -> None:
        self.collection = types.SimpleNamespace(
            get=lambda include=None: {"ids": [], "metadatas": []},
            delete=lambda **kwargs: None,
        )
        self.graph = types.SimpleNamespace(clear=lambda: None)
        self.added: list[str] = []
        self.stop_on = stop_on
        self._lexical_index = None
        self._should_stop = None

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any]) -> int:
        if self.stop_on is not None and len(self.added) == self.stop_on:
            raise gs.EmbeddingStopped("stopped between two batches")
        self.added.append(doc_id)
        return 1

    def _add_to_graph(self, *args: Any) -> None:
        pass

    def _save_graph(self) -> None:
        pass

    def stats(self) -> dict[str, Any]:
        return {}


@pytest.fixture
def three_files(tmp_path, monkeypatch) -> list[Path]:
    files = []
    for i in range(3):
        path = tmp_path / f"doc{i}.md"
        path.write_text(f"Document {i} holds some text.", encoding="utf-8")
        files.append(path)
    monkeypatch.setattr(gs, "iter_project_files", lambda root=".": files)
    monkeypatch.setattr(gs, "_document_metadata", lambda path: {"path": str(path)})
    return files


def test_a_long_build_reports_progress_and_stops_between_files(three_files):
    corpus = _Corpus()
    seen: list[tuple[int, int]] = []

    report = gs.index_project_files(
        corpus,  # type: ignore[arg-type]
        progress=lambda done, total: seen.append((done, total)),
        should_stop=lambda: len(seen) >= 2,
    )

    assert seen == [(1, 3), (2, 3)]
    assert report["stopped"] is True
    assert corpus.added == [str(path) for path in three_files[:2]]
    assert corpus._should_stop is None


def test_a_stop_inside_a_documents_embedding_ends_the_build(three_files):
    corpus = _Corpus(stop_on=1)

    report = gs.index_project_files(corpus)  # type: ignore[arg-type]

    assert report["stopped"] is True
    assert report["errors"] == []
    assert corpus.added == [str(three_files[0])]


def test_a_build_stopped_midway_says_the_next_run_carries_on():
    line = serve._corpus_feed_line(
        {"source": "stopped_midway", "model": QWEN, "indexed": 40, "embedded": 38, "elapsed_s": 912.0}
    )

    assert line is not None
    assert QWEN in line
    assert "carries on" in line


# ---------------------------------------------------------------------------
# the console
# ---------------------------------------------------------------------------


@pytest.fixture
def installed(monkeypatch) -> None:
    monkeypatch.setattr(serve, "list_ollama_models", lambda: ["dolphin-9b:Q4_K_M", QWEN])
    monkeypatch.setattr(
        serve,
        "ollama_model_capabilities",
        lambda tag: ["embedding"] if tag == QWEN else ["completion", "tools"],
    )


def test_the_dropdown_offers_minilm_and_the_daemons_embedding_models(tmp_path, monkeypatch, installed):
    monkeypatch.chdir(tmp_path)

    options = serve.rpc_embedding_options({})

    assert options["active"] == gs.EMBEDDING_MODEL_NAME
    assert [option["model"] for option in options["options"]] == [gs.EMBEDDING_MODEL_NAME, QWEN]
    minilm, qwen_option = options["options"]
    assert minilm["floor"] == gs.RETRIEVAL_RELEVANCE_FLOOR
    assert minilm["floor_source"] == "measured"
    assert qwen_option["corpus"] == "absent"
    assert qwen_option["floor_source"] == "not_measured"


def test_switching_models_drops_the_servers_corpus(tmp_path, monkeypatch, installed):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gs, "_kb_instance", None)
    monkeypatch.setattr(serve, "kb", object())

    result = serve.rpc_set_embedding_model({"model": QWEN})

    assert result["active"] == QWEN
    assert serve.kb is None
    assert gs.active_embedding_model() == QWEN


def test_a_model_the_machine_does_not_offer_is_refused(monkeypatch, installed):
    with pytest.raises(ValueError, match="not an embedding model"):
        serve.rpc_set_embedding_model({"model": "dolphin-9b:Q4_K_M"})
    assert gs.active_embedding_model() == gs.EMBEDDING_MODEL_NAME


def test_switching_is_refused_while_a_run_is_in_flight(monkeypatch, installed):
    monkeypatch.setitem(serve._run_progress, "running", True)
    monkeypatch.setitem(serve._run_progress, "goal", "a long job")

    with pytest.raises(ValueError, match="a long job"):
        serve.rpc_set_embedding_model({"model": QWEN})
    assert gs.active_embedding_model() == gs.EMBEDDING_MODEL_NAME


def test_the_floor_is_measured_once_the_models_corpus_is_whole(monkeypatch, qwen):
    measured: list[Any] = []
    monkeypatch.setattr(serve, "_open_kb", lambda: "the corpus")
    monkeypatch.setattr(serve, "corpus_state", lambda *args, **kwargs: ("indexed", qwen))
    monkeypatch.setattr(serve, "floor_calibration", lambda model=None: None)
    monkeypatch.setattr(
        serve,
        "calibrate_relevance_floor",
        lambda kb: measured.append(kb)
        or {"model": qwen, "floor": 0.5, "answered": [0.6], "unanswerable": [0.4]},
    )

    report = serve._calibrate_the_floor_before_the_run({"source": "built"})

    assert measured == ["the corpus"]
    assert report["source"] == "calibrated"
    assert "0.500" in (serve._calibration_feed_line(report) or "")


def test_a_stopped_build_is_not_calibrated(monkeypatch, qwen):
    monkeypatch.setattr(serve, "floor_calibration", lambda model=None: None)
    monkeypatch.setattr(
        serve, "calibrate_relevance_floor", lambda kb: pytest.fail("a fraction of the project")
    )

    report = serve._calibrate_the_floor_before_the_run({"source": "stopped_midway", "stopped": True})

    assert report["source"] == "stopped"


def test_minilm_is_never_recalibrated(monkeypatch):
    monkeypatch.setattr(serve, "calibrate_relevance_floor", lambda kb: pytest.fail("measured by hand"))

    assert serve._calibrate_the_floor_before_the_run({"source": "built"})["source"] == "known"

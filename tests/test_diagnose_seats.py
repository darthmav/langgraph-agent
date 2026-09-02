"""Tests for `scripts/diagnose_seats.py`.

The diagnostic's whole value is that its verdicts are trustworthy, so the one
thing it must never do is report a failing seat as a working one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "diagnose_seats.py"


def _load_diag() -> Any:
    """Import the script by path.

    It lives in `scripts/` rather than the package, so there is no import
    name for it. Registering it in `sys.modules` before executing is not
    optional: `@dataclass` resolves annotations through
    `sys.modules[cls.__module__]`, which is None until the module is there.
    """
    spec = importlib.util.spec_from_file_location("diagnose_seats", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_seats"] = module
    spec.loader.exec_module(module)
    return module


class _FakeConfig:
    """Just the surface `run_probe` touches, with a live (unstubbed) seat."""

    def __init__(self) -> None:
        self._seat_failures: dict[str, str] = {}

    def set_agent_llm(self, agent: str, provider: str, model: str) -> None:
        pass

    def get_agent_status(self, agent: str) -> dict[str, Any]:
        return {"stubbed": False, "reason": ""}


@pytest.fixture
def diag() -> Any:
    return _load_diag()


def test_a_silent_researcher_seat_probes_as_empty(diag, monkeypatch):
    """A seat that answered with nothing must never probe `ok`.

    `_gather_research` does not hand back nothing when the seat says nothing.
    It hands back `_RESEARCH_EMPTY` -- a fully-formed three-section fallback
    explaining that the seat said nothing. Grading the text alone scored that
    apology as the model's answer: `_said_nothing` was False, all three
    sections were filled, and a silent Researcher came back `ok`, which is the
    failure this probe exists to catch reported as its opposite. The status is
    what settles it.
    """
    from langgraph_agent import nodes
    from langgraph_agent.state import Verdict

    monkeypatch.setattr(
        nodes, "_gather_research",
        lambda state: (nodes._RESEARCH_EMPTY, nodes._SEAT_EMPTY),
    )

    mods = {"nodes": nodes, "config": _FakeConfig(), "Verdict": Verdict}
    result = diag.run_probe(diag.BY_KEY["kimi-k3"], "researcher", mods)

    assert result.status == "empty"
    assert "loop the run" in result.detail


def test_real_findings_still_probe_as_ok(diag, monkeypatch):
    """The guard must not condemn a seat that actually answered."""
    from langgraph_agent import nodes
    from langgraph_agent.state import Verdict

    findings = (
        "## Key Findings\nThe client wraps httpx directly.\n\n"
        "## Relevant Context\nRetries live in the client today.\n\n"
        "## Recommendations for Builder\nAdd a bounded `_retry` helper.\n\n"
        "## Status\nready_for_builder"
    )
    monkeypatch.setattr(
        nodes, "_gather_research", lambda state: (findings, "ready_for_builder")
    )

    mods = {"nodes": nodes, "config": _FakeConfig(), "Verdict": Verdict}
    result = diag.run_probe(diag.BY_KEY["kimi-k3"], "researcher", mods)

    assert result.status == "ok"


def test_a_seat_with_no_credentials_probes_as_stubbed(diag):
    """A StubLLM seat answers everything with canned text; that is not `ok`."""
    from langgraph_agent import nodes
    from langgraph_agent.state import Verdict

    class _StubbedConfig(_FakeConfig):
        def get_agent_status(self, agent: str) -> dict[str, Any]:
            return {"stubbed": True, "reason": "ANTHROPIC_API_KEY not set"}

    mods = {"nodes": nodes, "config": _StubbedConfig(), "Verdict": Verdict}
    result = diag.run_probe(diag.BY_KEY["opus"], "researcher", mods)

    assert result.status == "stubbed"
    assert "ANTHROPIC_API_KEY" in result.detail


def test_an_unprobed_role_is_not_reported_as_a_failed_one(diag):
    """`--phase teams` must not invent probe failures for a phase never run."""
    best, notes = diag.recommend([], [])

    assert best == {}
    assert any("Phase 1 did not run" in note for note in notes)
    assert not any("No model passed" in note for note in notes)

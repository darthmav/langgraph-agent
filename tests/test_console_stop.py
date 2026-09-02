"""Console-level tests for the emergency stop and run recovery.

These exercise `serve.py` directly rather than over HTTP: the RPC methods are
plain functions, and the transport adds nothing worth testing here. What they
do cover is the bookkeeping around a run -- the flag, the snapshot, and the
`finally` that has to hold even when the graph raises.
"""

from __future__ import annotations

import json
import threading

import pytest

import serve
from langgraph_agent.control import RUN_CONTROL


@pytest.fixture(autouse=True)
def _isolate_run_state(tmp_path, monkeypatch):
    """Keep each test off the real snapshot file and out of the last one's state."""
    monkeypatch.setattr(serve, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(serve, "LAST_RUN_PATH", tmp_path / "runs" / "last_run.json")
    monkeypatch.setattr(serve, "_last_run_snapshot", None)
    with serve._run_lock:
        serve._run_progress.update(
            running=False, goal="", messages=[], node="", run_id="", stopping=False
        )
    yield
    with serve._run_lock:
        serve._run_progress.update(running=False, run_id="", stopping=False)


class _FakeGraph:
    """A graph whose supersteps are driven by the test.

    `stream` yields one event per node name, calling `during_step` while each
    one is notionally running. That is where a test trips the stop, and the
    server gets its look at the flag once the event has been handed over -- the
    same shape as the real thing.
    """

    def __init__(self, nodes, during_step=None, raises=None):
        self._nodes = nodes
        self._during_step = during_step
        self._raises = raises
        self.steps = 0

    def stream(self, state, config):
        for name in self._nodes:
            if self._raises:
                raise self._raises
            self.steps += 1
            # Called while the node is "running", before its result is handed
            # over -- which is when a real Stop arrives.
            if self._during_step:
                self._during_step()
            state = dict(state)
            state["messages"] = [*state.get("messages", []), f"[{name.title()}] did work"]
            state["plan"] = "1. Something"
            yield {name: state}


def test_a_stop_ends_the_run_at_a_node_boundary(monkeypatch):
    """The stop breaks the stream loop, and the state so far comes back."""
    def trip():
        serve.rpc_stop_run({})

    graph = _FakeGraph(["architect", "planner", "builder"], during_step=trip)
    monkeypatch.setattr(serve, "graph", graph)

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    # Broken out after the first superstep, not after all three.
    assert graph.steps == 1
    assert result["stopped"] is True
    assert result["stop_reason"]
    # The work it did produce is returned, not discarded.
    assert "[Architect] did work" in result["messages"]
    assert "Stopped by the emergency stop" in result["messages"][-1]


def test_a_finished_run_is_not_reported_as_stopped(monkeypatch):
    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "builder"]))

    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert result["stopped"] is False
    assert result["over_budget"] is False


def test_stopping_with_no_run_in_flight_is_refused():
    answer = serve.rpc_stop_run({})

    assert answer["stopping"] is False
    assert "No run is in flight" in answer["detail"]


def test_a_stale_stop_is_refused_when_a_new_run_is_armed(monkeypatch):
    """The console can hold an id across a reload; it must not kill a new run."""
    seen = {}

    def trip():
        seen["answer"] = serve.rpc_stop_run({"run_id": "some-run-that-ended"})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "planner"], during_step=trip))
    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert seen["answer"]["stopping"] is False
    assert "already ended" in seen["answer"]["detail"]
    assert result["stopped"] is False


def test_only_one_run_at_a_time(monkeypatch):
    """A second run is refused rather than left to clobber the first's progress."""
    started = threading.Event()
    release = threading.Event()

    def block():
        started.set()
        release.wait(5)

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect"], during_step=block))

    outcome: list = []
    worker = threading.Thread(
        target=lambda: outcome.append(serve.rpc_run_goal({"goal": "first"})),
        daemon=True,
    )
    worker.start()
    assert started.wait(5)

    with pytest.raises(ValueError, match="already in flight"):
        serve.rpc_run_goal({"goal": "second"})

    release.set()
    worker.join(5)
    assert outcome and outcome[0]["stopped"] is False


def test_a_run_that_raises_still_clears_the_flag_and_leaves_a_snapshot(monkeypatch):
    """Before the finally, a crashed run left the console polling a phantom.

    It also left nothing recoverable -- which is the case where the operator
    most wants the partial work back.
    """
    monkeypatch.setattr(
        serve, "graph", _FakeGraph(["architect"], raises=RuntimeError("seat died"))
    )

    with pytest.raises(RuntimeError, match="seat died"):
        serve.rpc_run_goal({"goal": "Do a thing"})

    assert serve.rpc_run_progress({})["running"] is False
    assert RUN_CONTROL.run_id() == ""

    snapshot = serve.rpc_last_run({})["snapshot"]
    assert snapshot is not None
    assert snapshot["error"] == "seat died"


def test_the_snapshot_survives_the_console(monkeypatch):
    """A reloaded page reads the run's outcome back off disk."""
    def trip():
        serve.rpc_stop_run({})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "builder"], during_step=trip))
    result = serve.rpc_run_goal({"goal": "Do a thing"})

    # In memory, for the page that is open now...
    assert serve.rpc_last_run({})["snapshot"]["run_id"] == result["run_id"]

    # ...and on disk, for the one that reloads.
    monkeypatch.setattr(serve, "_last_run_snapshot", None)
    written = json.loads(serve.LAST_RUN_PATH.read_text())
    assert written["stopped"] is True
    assert serve.rpc_last_run({})["snapshot"]["stopped"] is True


def test_run_progress_names_the_run_a_stop_would_target(monkeypatch):
    """The console cannot aim a Stop at a run it cannot name."""
    seen = {}

    def look():
        seen["progress"] = serve.rpc_run_progress({})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect"], during_step=look))
    result = serve.rpc_run_goal({"goal": "Do a thing"})

    assert seen["progress"]["running"] is True
    assert seen["progress"]["run_id"] == result["run_id"]
    # And it is released once the run is over, so a later Stop finds nothing.
    assert serve.rpc_run_progress({})["run_id"] == ""


def test_the_stop_is_visible_while_it_is_landing(monkeypatch):
    """A stop that has been asked for but not yet taken effect must show."""
    seen = {}

    def trip():
        serve.rpc_stop_run({})
        seen["progress"] = serve.rpc_run_progress({})

    monkeypatch.setattr(serve, "graph", _FakeGraph(["architect", "planner"], during_step=trip))
    serve.rpc_run_goal({"goal": "Do a thing"})

    assert seen["progress"]["stopping"] is True


def test_expect_failures_rides_through_a_stopped_run(monkeypatch):
    """The safeguard setting is part of what a stopped run has to hand back.

    The recovery view says which rules were in force when the run was cut off,
    and it can only do that if the flag survives into the snapshot.
    """
    monkeypatch.setattr(
        serve, "graph", _FakeGraph(["architect"], during_step=lambda: serve.rpc_stop_run({}))
    )

    result = serve.rpc_run_goal({"goal": "Write a failing fixture", "expect_failures": True})

    assert result["expect_failures"] is True
    assert serve.rpc_last_run({})["snapshot"]["expect_failures"] is True

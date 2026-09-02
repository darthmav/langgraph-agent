"""The stop signal for the run currently in flight.

Deliberately a module-level object rather than a field on `AgentState`. The
graph is compiled without a checkpointer, so LangGraph's `update_state` /
`interrupt` machinery is unavailable and nothing outside a node can write into a
state the graph is already streaming. A flag the nodes read is the only channel
that reaches a run in progress.

The stop is cooperative. Nothing here interrupts anything: it sets a flag, and
the checkpoints in `nodes.py` and `serve.py` decline to start the *next* piece of
work. Work already in flight -- a `filesystem_write`, a staged commit, a test
run, a model call -- always finishes, because the alternative is a half-written
file and a worker still writing into the project after the node returned.
"""

import threading
import time

# Why a run ended, when the caller did not say.
DEFAULT_STOP_REASON = "Stopped from the console."


class RunControl:
    """An armed/stopped flag for one run, shared across threads.

    The run executes on the HTTP request thread that started it, so the stop
    arrives on a different thread -- which is why `ThreadingHTTPServer` is
    load-bearing and why this is guarded rather than a plain bool.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._run_id = ""
        self._reason = ""

    def arm(self, run_id: str) -> None:
        """Take ownership of a new run, clearing any previous stop."""
        with self._lock:
            self._run_id = run_id
            self._reason = ""
            self._event.clear()

    def disarm(self) -> None:
        """Release the run. A stop after this point has nothing to act on."""
        with self._lock:
            self._run_id = ""
            self._reason = ""
            self._event.clear()

    def stop(self, run_id: str = "", reason: str = "") -> bool:
        """Ask the armed run to stop. False if there is nothing to stop.

        A `run_id` that does not match the armed run is refused rather than
        applied to whatever is running now: the console can hold a stale id
        across a page reload, and a Stop meant for a finished run must not kill
        the run that replaced it. An empty `run_id` means "whatever is armed",
        which is what a caller with no id (curl, a test) wants.
        """
        with self._lock:
            if not self._run_id:
                return False
            if run_id and run_id != self._run_id:
                return False
            self._reason = reason or DEFAULT_STOP_REASON
            self._event.set()
            return True

    def stopped(self) -> bool:
        """Whether the armed run has been asked to stop.

        Lock-free on purpose: this is called at the top of every Builder tool
        turn and before every verified file.
        """
        return self._event.is_set()

    def reason(self) -> str:
        with self._lock:
            return self._reason

    def run_id(self) -> str:
        with self._lock:
            return self._run_id


# The single control for this process. The server runs one graph object and one
# run at a time, so one control is the whole story.
RUN_CONTROL = RunControl()


class NodeActivity:
    """Which seat is executing *right now*, for the console's seat lights.

    Deliberately separate from `_run_progress["node"]` in `serve.py`, which is
    fed by `graph.stream` and therefore names the node that has just *finished*
    -- LangGraph yields an update when a superstep completes, not when one
    starts. That is the right value for the feed, which lists what happened, and
    the wrong one for a light meaning "this seat is working": it lights the
    previous seat for the whole of the next seat's turn, so the slowest node in
    the run is the one node whose light never comes on. A stalled Architect
    would show as a busy Builder, which is precisely backwards.

    Set from the graph rather than from inside the node bodies: every node has
    several return paths (a stop, a deadline fallback, the ordinary one), so a
    wrapper with a `finally` is the only way a light cannot be left on by an
    exit nobody thought about.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node = ""
        self._since = 0.0

    def enter(self, node: str) -> None:
        with self._lock:
            self._node = node
            self._since = time.monotonic()

    def leave(self, node: str) -> None:
        """Clear the light, unless someone else already claimed it.

        The name check matters even though the graph runs one node at a time:
        a node abandoned by `_with_deadline` keeps a worker thread alive, and
        nothing that finishes late may darken the seat that is working now.
        """
        with self._lock:
            if self._node == node:
                self._node = ""
                self._since = 0.0

    def clear(self) -> None:
        """No seat is working. The run's `finally` calls this."""
        with self._lock:
            self._node = ""
            self._since = 0.0

    def current(self) -> str:
        with self._lock:
            return self._node

    def busy_for(self) -> float:
        """Seconds the current seat has been in its turn; 0.0 when idle."""
        with self._lock:
            return time.monotonic() - self._since if self._node else 0.0


# One run at a time, so one activity light, for the same reason as RUN_CONTROL.
ACTIVITY = NodeActivity()

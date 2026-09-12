#!/usr/bin/env bash
# One command from a fresh Arch / Omarchy machine to a console that runs.
#
# Safe to re-run. Every step looks before it acts, so a second run on a
# finished machine installs nothing, asks for no password, and ends by telling
# you whether each seat can actually run -- which is the only definition of
# "installed" that matters here.
#
# Everything it installs is free to use. System packages come from the Arch
# repos, Python packages from PyPI (torch from PyTorch's CPU index), the
# embedding model from Hugging Face, and inference from Ollama Cloud tags
# through the local daemon, which a free ollama.com account can run. No seat
# needs an API key and nothing here asks for one: Anthropic and OpenAI stay
# optional, and unconfigured.
#
# Usage:
#   ./install.sh                everything below
#   ./install.sh --minimal      skip the optional developer tools
#   ./install.sh --no-system    skip pacman entirely (no sudo); Python side only
#   ./install.sh --no-checks    skip ruff / mypy / pytest
#   ./install.sh --no-desktop   do not add the console to the app launcher
#   ./install.sh --yes          never prompt (pacman --noconfirm, no sign-in)

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"

MINIMAL=0 SYSTEM=1 CHECKS=1 DESKTOP=1 ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --minimal)    MINIMAL=1 ;;
        --no-system)  SYSTEM=0 ;;
        --no-checks)  CHECKS=0 ;;
        --no-desktop) DESKTOP=0 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        -h|--help)    sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# Problems are collected rather than fatal from the Ollama step on: a machine
# that is not signed in yet still gets its venv, corpus and checks, and the
# summary says exactly what is left instead of the script dying halfway.
PROBLEMS=()
problem() { PROBLEMS+=("$1"); echo "  ✗ $1"; }
step()    { echo; echo "► $1"; }
ok()      { echo "  ✓ $1"; }
interactive() { [ "$ASSUME_YES" -eq 0 ] && [ -t 0 ]; }

echo "============================================"
echo "  Ambiguity 4-Agent Console -- install"
echo "============================================"

if [ "$(id -u)" -eq 0 ]; then
    # The venv, the corpus and the launcher all belong to the user who runs
    # the console. Built as root, every one of them is owned by root and the
    # first ordinary run fails on a permission error far from its cause.
    echo "Run this as your own user, not root; it calls sudo where it must." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------

# Required: the console cannot start, or cannot do its job, without these.
REQUIRED=(
    python       # the interpreter, if nothing on PATH is new enough
    git          # the Builder's git_ tools
    curl         # launch_console.sh's readiness check, and the daemon probe
    ollama       # the daemon every default seat runs through
    base-devel   # a compiler, for any wheel with no binary for this Python
    xdg-utils    # xdg-open: how the console opens in your browser
)

# Useful: nothing breaks without them, but working on this repo is worse.
USEFUL=(
    ripgrep      # rg: searching a codebase this size
    fd           # finding files by name
    jq           # reading /rpc replies: curl ... | jq
    sqlite       # inspecting knowledge/chroma/chroma.sqlite3 directly
    shellcheck   # linting this script and launch_console.sh
    github-cli   # gh: pull requests and the CI runs in .github/workflows
    btop         # watching the embedder and the test suite use the machine
)

if [ "$SYSTEM" -eq 1 ]; then
    step "System packages (pacman)"
    if ! command -v pacman >/dev/null; then
        echo "  pacman not found: this installer targets Arch / Omarchy." >&2
        echo "  Install the equivalents of: ${REQUIRED[*]}" >&2
        echo "  then re-run with --no-system." >&2
        exit 1
    fi

    wanted=("${REQUIRED[@]}")
    [ "$MINIMAL" -eq 0 ] && wanted+=("${USEFUL[@]}")

    # The console needs a browser, and Omarchy ships Chromium. Only a machine
    # with none at all gets one, so an existing choice is never second-guessed.
    browser_found=0
    for b in chromium firefox brave google-chrome-stable librewolf zen-browser vivaldi; do
        command -v "$b" >/dev/null && { browser_found=1; break; }
    done
    [ "$browser_found" -eq 0 ] && wanted+=(chromium)

    # `pacman -T` prints what is not satisfied, provides included, so a rerun
    # on a finished machine reaches no sudo prompt at all.
    mapfile -t missing < <(pacman -T "${wanted[@]}" || true)
    if [ "${#missing[@]}" -eq 0 ]; then
        ok "all ${#wanted[@]} packages already installed"
    else
        echo "  installing: ${missing[*]}"
        pacman_flags=(-S --needed)
        [ "$ASSUME_YES" -eq 1 ] && pacman_flags+=(--noconfirm)
        # No -y: `pacman -Sy <pkg>` is a partial upgrade, which Arch does not
        # support. If the mirrors have moved on, upgrade the system properly
        # (omarchy-update, or sudo pacman -Syu) and re-run.
        if ! sudo pacman "${pacman_flags[@]}" "${missing[@]}"; then
            echo "  pacman failed. If it could not find a package or file, the" >&2
            echo "  package database is stale: run omarchy-update (or" >&2
            echo "  sudo pacman -Syu) and then re-run ./install.sh." >&2
            exit 1
        fi
        ok "installed ${#missing[@]} package(s)"
    fi
fi

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------

step "Python environment (.venv)"

# The floor is read from pyproject.toml rather than restated here, so the day
# it moves this script moves with it.
FLOOR="$(sed -n 's/^requires-python *= *">=\([0-9][0-9.]*\)".*/\1/p' pyproject.toml)"
FLOOR="${FLOOR:-3.12}"
floor_ok() {
    "$1" -c "import sys; sys.exit(sys.version_info < tuple(map(int, '$FLOOR'.split('.'))))" \
        2>/dev/null
}

# A venv whose interpreter was removed underneath it -- a mise upgrade, an
# Arch Python minor bump -- still has a bin/python symlink, pointing at
# nothing. Test it by running it, and rebuild rather than patch.
if [ -x "$VENV/bin/python" ] && floor_ok "$VENV/bin/python"; then
    ok "reusing .venv ($("$VENV/bin/python" --version))"
else
    [ -e "$VENV" ] && { echo "  .venv is broken or too old; rebuilding it"; rm -rf "$VENV"; }
    # The first interpreter on PATH wins, so an Omarchy machine gets mise's
    # Python if that is what you chose; /usr/bin/python3 is the fallback.
    BASE_PY=""
    for candidate in python3 python /usr/bin/python3; do
        if command -v "$candidate" >/dev/null && floor_ok "$(command -v "$candidate")"; then
            BASE_PY="$(command -v "$candidate")"
            break
        fi
    done
    if [ -z "$BASE_PY" ]; then
        echo "  No Python >= $FLOOR found. Install it (sudo pacman -S python)" >&2
        echo "  or drop --no-system, and re-run." >&2
        exit 1
    fi
    "$BASE_PY" -m venv "$VENV"
    ok "created .venv with $("$VENV/bin/python" --version) from $BASE_PY"
fi

PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip

# CPU torch before the project, for the reason CI gives: sentence-transformers
# pulls torch, and torch from PyPI brings the whole CUDA stack -- gigabytes the
# code never uses, since graphrag_server.py forces the embedder onto the CPU.
# Installed first, it pins the resolution below to the CPU build.
if "$PY" -c "import torch" 2>/dev/null; then
    ok "torch present ($("$PY" -c 'import torch; print(torch.__version__)'))"
else
    echo "  installing CPU-only torch (the largest download; a few minutes)"
    "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    ok "torch $("$PY" -c 'import torch; print(torch.__version__)')"
fi

echo "  installing the project and its dev tools (pip install -e \".[dev]\")"
"$PY" -m pip install --quiet -e ".[dev]"
ok "langgraph-agent installed in editable mode"

# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------

step "Configuration (.env)"
if [ -f .env ]; then
    ok ".env exists; left as it is"
else
    cp .env.example .env
    ok "created .env from .env.example (no key needed for the default seats)"
fi
set -a
# shellcheck source=/dev/null
. ./.env
set +a

# ---------------------------------------------------------------------------
# 4. Ollama: daemon, sign-in, seat models
# ---------------------------------------------------------------------------

step "Ollama"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_HOST="$OLLAMA_URL"

daemon_up() { curl -s -m 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; }

if ! command -v ollama >/dev/null; then
    problem "ollama is not installed (drop --no-system, or: sudo pacman -S ollama)"
else
    if ! daemon_up && [ -z "${OLLAMA_BASE_URL:-}" ] && systemctl cat ollama.service >/dev/null 2>&1; then
        echo "  starting ollama.service (and enabling it at boot)"
        sudo systemctl enable --now ollama.service || true
        for _ in $(seq 1 20); do daemon_up && break; sleep 0.5; done
    fi

    if ! daemon_up; then
        problem "Ollama daemon unreachable at $OLLAMA_URL"
    else
        ok "daemon answering at $OLLAMA_URL ($(ollama --version 2>/dev/null | awk '{print $NF}'))"

        # `:cloud` tags run on ollama.com, so the daemon has to be signed in.
        # /api/me answers 200 only when it is; a daemon too old to have the
        # route answers 404, and then the pulls below are the test instead.
        me_status() { curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "$OLLAMA_URL/api/me"; }
        signed="$(me_status || true)"
        if [ "$signed" != "200" ] && [ "$signed" != "404" ]; then
            if interactive; then
                echo "  the daemon is not signed in to ollama.com (a free account is enough)"
                ollama signin || true
                signed="$(me_status || true)"
            fi
        fi
        case "$signed" in
            200) ok "signed in to ollama.com" ;;
            404) echo "  (this daemon cannot report its sign-in; the pulls will tell)" ;;
            *)   problem "Ollama is not signed in: run 'ollama signin', then re-run ./install.sh" ;;
        esac

        # Pull what the seats actually use -- config.py plus any .env override
        # -- rather than a list kept here. A seat whose tag is not on the
        # daemon shows NOT PULLED and fails its run.
        mapfile -t seat_models < <("$PY" - <<'PY'
from langgraph_agent.config import AGENTS, get_agent_model_info

seen: list[str] = []
for agent in AGENTS:
    info = get_agent_model_info(agent)
    if info["provider"] == "ollama" and info["model"] not in seen:
        seen.append(info["model"])
print("\n".join(seen))
PY
        )
        pulled="$(ollama list 2>/dev/null | awk 'NR > 1 {print $1}')"
        for model in "${seat_models[@]}"; do
            [ -z "$model" ] && continue
            if grep -qxF "$model" <<<"$pulled"; then
                ok "$model already on the daemon"
            elif ollama pull "$model"; then
                ok "pulled $model"
            else
                problem "could not pull $model (signed in? tag spelled right?)"
            fi
        done
    fi
fi

# ---------------------------------------------------------------------------
# 5. Embedding model and corpus
# ---------------------------------------------------------------------------

step "Embedding model"
# Fetched now so the first search is not also a download, and so a machine
# that goes offline later still has it. The name comes from graphrag_server,
# where the relevance floor calibrated against it lives too.
if "$PY" - 2>/tmp/ambiguity-embedder.log <<'PY'
from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
print(f"  ✓ {EMBEDDING_MODEL_NAME} cached")
PY
then :; else problem "could not fetch the embedding model; see /tmp/ambiguity-embedder.log"; fi

# The corpus is not built here, and there is no step that builds one. Two
# things index: a run, which rebuilds before the Architect opens, and embedding
# a document into the corpus from the console. An install-time index was a
# third, and a third is one too many -- it is the one that decides how fresh
# the corpus is on a machine nobody has run anything on yet, which is a
# question the first run answers correctly by itself. The model above is
# fetched so that first run is an index and not also a download.

# ---------------------------------------------------------------------------
# 6. Checks
# ---------------------------------------------------------------------------

if [ "$CHECKS" -eq 1 ]; then
    step "Checks (CLAUDE.md's Quick Reference)"
    # Read out of the Quick Reference instead of copied, the same promise
    # ci.yml keeps: "all checks pass" means one thing wherever it is said.
    mapfile -t checks < <(sed -n '/^# Run all checks$/,/^```$/p' CLAUDE.md \
                          | grep -E '^(ruff|mypy|python -m pytest)')
    if [ "${#checks[@]}" -eq 0 ]; then
        problem "no checks found under 'Run all checks' in CLAUDE.md"
    fi
    for check in "${checks[@]}"; do
        log="/tmp/ambiguity-check-${check%% *}.log"
        if PATH="$VENV/bin:$PATH" bash -c "$check" >"$log" 2>&1; then
            ok "$check"
        else
            problem "$check -- failed; see $log"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 7. App launcher
# ---------------------------------------------------------------------------

if [ "$DESKTOP" -eq 1 ]; then
    step "App launcher"
    # A freedesktop entry, which is what Omarchy's launcher (Super+Space)
    # lists. It runs launch_console.sh without a terminal: the console's own
    # exit button stops the server, and launching again while it is up just
    # opens the page.
    apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$apps"
    cat >"$apps/ambiguity-console.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Ambiguity Console
Comment=Architect, Planner, Researcher and Builder, in the browser
Exec="$ROOT/launch_console.sh"
Path=$ROOT
Icon=utilities-terminal
Terminal=false
Categories=Development;
EOF
    command -v update-desktop-database >/dev/null && update-desktop-database "$apps" 2>/dev/null || true
    ok "\"Ambiguity Console\" added to the app launcher"
fi

# ---------------------------------------------------------------------------
# Summary: can each seat actually run?
# ---------------------------------------------------------------------------

step "Seats"
# get_agent_status is the one thing that knows whether a seat is live, stubbed
# or failing, so it is asked rather than inferred from what was installed.
if ! "$PY" - <<'PY'
import sys

from langgraph_agent.config import AGENTS, get_agent_status

dead = 0
for agent in AGENTS:
    seat = get_agent_status(agent)
    mark = "✓" if seat["live"] else "✗"
    note = "" if seat["live"] else f"  -- {seat['badge']}: {seat['reason']}"
    print(f"  {mark} {agent:11}{seat['model']:24}{seat['provider']}{note}")
    dead += not seat["live"]
sys.exit(1 if dead else 0)
PY
then
    problem "not every seat can run; see the list above"
fi

echo
echo "============================================"
if [ "${#PROBLEMS[@]}" -eq 0 ]; then
    echo "  Ready. Start the console with:"
    echo "    ./launch_console.sh"
    [ "$DESKTOP" -eq 1 ] && echo "  or pick \"Ambiguity Console\" from the app launcher."
    echo "============================================"
    exit 0
fi
echo "  Installed, with ${#PROBLEMS[@]} thing(s) left to fix:"
for p in "${PROBLEMS[@]}"; do echo "    - $p"; done
echo "  Fix them and re-run ./install.sh; finished steps are skipped."
echo "============================================"
exit 1

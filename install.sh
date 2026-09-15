#!/usr/bin/env bash
# One command from a fresh Arch / Omarchy machine to a console that runs.
#
# Safe to re-run. Every step looks before it acts, so a second run on a
# finished machine installs nothing, asks for no password, and ends by telling
# you whether each seat can actually run -- which is the only definition of
# "installed" that matters here.
#
# Everything it installs is free to use. System packages come from the Arch
# repos, Python packages from PyPI, the embedding model from Hugging Face, the
# SearxNG image from Docker Hub, and inference from Ollama Cloud tags through
# the local daemon, which a free ollama.com account can run. No seat needs an
# API key and nothing here asks for one: Anthropic and OpenAI stay optional,
# and unconfigured.
#
# The GPU side is the machine's own setup, not this script's: the driver,
# torch's GPU build in .venv, and Ollama's GPU backend. None of it is installed
# or configured here. What this script does is prove each one initialises, and
# refuse to put a CPU or PyPI torch where a GPU build belongs.
#
# Usage:
#   ./install.sh                everything below
#   ./install.sh --minimal      skip the optional developer tools
#   ./install.sh --no-system    skip pacman entirely (no sudo); Python side only
#   ./install.sh --no-searxng   do not run a SearxNG for online research
#   ./install.sh --no-checks    skip ruff / mypy / pytest
#   ./install.sh --no-probe     do not send each seat a one-word test prompt
#   ./install.sh --no-desktop   do not add the console to the app launcher
#   ./install.sh --yes          never prompt (pacman --noconfirm, no sign-ins)

set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="$ROOT/.venv"

MINIMAL=0 SYSTEM=1 SEARXNG=1 CHECKS=1 PROBE=1 DESKTOP=1 ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --minimal)    MINIMAL=1 ;;
        --no-system)  SYSTEM=0 ;;
        --no-searxng) SEARXNG=0 ;;
        --no-checks)  CHECKS=0 ;;
        --no-probe)   PROBE=0 ;;
        --no-desktop) DESKTOP=0 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        -h|--help)    sed -n '2,/^$/{s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# Problems are collected rather than fatal from the Ollama step on: a machine
# that is not signed in yet still gets its venv and checks, and the summary
# says exactly what is left instead of the script dying halfway.
PROBLEMS=()
problem() { PROBLEMS+=("$1"); echo "  ✗ $1"; }
# Notes are not problems: everything works, but slower than it could, and the
# summary should say so rather than leave it in the scrollback.
NOTES=()
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

# The port the SearxNG this script runs listens on, loopback only.
SEARXNG_PORT=8888
SEARXNG_LOCAL="http://127.0.0.1:$SEARXNG_PORT"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------

# Required: the console cannot start, or cannot do its job, without these.
REQUIRED=(
    python       # the interpreter, if nothing on PATH is new enough
    git          # the Builder's git_ tools
    github-cli   # gh: git_dwell's pr and merge stages end its default pipeline
    curl         # launch_console.sh's readiness check, and the daemon probe
    ollama       # the daemon every default seat runs through
    base-devel   # a compiler, for any wheel with no binary for this Python
    xdg-utils    # xdg-open: how the console opens in your browser
)

# Online research: SearxNG runs in a rootless podman container, so it needs no
# daemon, no group membership and no sudo after this step. crun is named
# because podman accepts either OCI runtime and pacman would otherwise stop to
# ask which.
SEARXNG_PACKAGES=(podman crun)

# Useful: nothing breaks without them, but working on this repo is worse.
USEFUL=(
    ripgrep      # rg: searching a codebase this size
    fd           # finding files by name
    jq           # reading /rpc replies: curl ... | jq
    sqlite       # inspecting knowledge/chroma/chroma.sqlite3 directly
    shellcheck   # linting this script and launch_console.sh
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
    [ "$SEARXNG" -eq 1 ] && wanted+=("${SEARXNG_PACKAGES[@]}")
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
# 2. Configuration
# ---------------------------------------------------------------------------

# Before the Python environment, because what goes into .venv depends on it:
# EMBEDDING_DEVICE decides whether torch has to be a GPU build.
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

# Whether this machine embeds on a card. It does when EMBEDDING_DEVICE names
# one, and it is expected to when nvidia-smi lists one: a card the driver
# answers for is a card this app should be able to use, and the GPU build that
# uses it is set up with the machine, not here.
EMBED_DEVICE="$(printf '%s' "${EMBEDDING_DEVICE:-cpu}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
GPU_VISIBLE=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    GPU_VISIBLE=1
fi
GPU_EXPECTED="$GPU_VISIBLE"
[ "$EMBED_DEVICE" != "cpu" ] && GPU_EXPECTED=1

# ---------------------------------------------------------------------------
# 3. Python environment
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

# torch goes in before the project, for the reason CI gives: sentence-transformers
# pulls torch, and left to pip it arrives from PyPI with whatever CUDA stack
# that release was built against -- which may carry no kernels for the card
# here. So pip is never left to choose:
#   - a torch already in .venv is kept, and pinned for the install below, so
#     the resolver cannot replace a GPU build with something it prefers;
#   - no torch on a machine that embeds on a card stops here, because the GPU
#     build belongs to the machine's setup and anything installed in its place
#     would be the wrong one;
#   - no torch and no card gets the CPU build, which is all such a machine uses.
torch_version="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
pip_pin=()
if [ -n "$torch_version" ]; then
    ok "torch $torch_version present; kept as it is"
    torch_pin="$(mktemp)"
    trap 'rm -f "$torch_pin"' EXIT
    printf 'torch==%s\n' "$torch_version" >"$torch_pin"
    pip_pin=(--constraint "$torch_pin")
elif [ "$GPU_EXPECTED" -eq 1 ]; then
    if [ "$EMBED_DEVICE" != "cpu" ]; then reason="EMBEDDING_DEVICE=$EMBED_DEVICE"; else reason="nvidia-smi lists a card"; fi
    echo "  .venv has no torch, and this machine embeds on a card ($reason)." >&2
    echo "  The GPU build of torch is part of the machine's own setup, so it is" >&2
    echo "  not installed here: put the build that carries kernels for this card" >&2
    echo "  into $VENV, then re-run ./install.sh." >&2
    exit 1
else
    echo "  installing CPU-only torch (no card here; the largest download, a few minutes)"
    "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    ok "torch $("$PY" -c 'import torch; print(torch.__version__)')"
fi

echo "  installing the project and its dev tools (pip install -e \".[dev]\")"
"$PY" -m pip install --quiet -e ".[dev]" "${pip_pin[@]}"
ok "langgraph-agent installed in editable mode"

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

        # Pull what the seats and the embedder actually use -- config.py and
        # graphrag_server plus any .env override -- rather than a list kept
        # here. A seat whose tag is not on the daemon shows NOT PULLED and
        # fails its run; an embedding tag that is missing fails the corpus
        # phase every run starts with.
        mapfile -t seat_models < <("$PY" - <<'PY'
from langgraph_agent.config import AGENTS, get_agent_model_info
from langgraph_agent.graphrag_server import active_embedding_model, embedding_backend

seen: list[str] = []
for agent in AGENTS:
    info = get_agent_model_info(agent)
    if info["provider"] == "ollama" and info["model"] not in seen:
        seen.append(info["model"])
embedder = active_embedding_model()
if embedding_backend(embedder) == "ollama" and embedder not in seen:
    seen.append(embedder)
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
# where the relevance floor calibrated against it lives too. MiniLM is fetched
# even when the default embedder is an Ollama tag: every corpus is chunked with
# its tokenizer, whichever model embeds the chunks.
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
# 6. GPU: does the machine's GPU build initialise?
# ---------------------------------------------------------------------------

step "GPU (torch in .venv)"
# Proven, never installed or configured. `torch.cuda.is_available()` is not the
# proof: a build with no kernels for the card still reports True, and on
# 2026-09-10 exactly that took search and indexing down. So every card is made
# to compute, and a card named in EMBEDDING_DEVICE is then given the embedder
# through `_embedder_on` -- the call a run makes, loading MiniLM onto the card
# and encoding every shape an index will ask for. Run before the Embedder step
# below loads an Ollama model, which would otherwise be holding the card.
gpu_status=0
# TQDM_DISABLE: the model load draws a weights progress bar over these lines.
GPU_VISIBLE="$GPU_VISIBLE" TQDM_DISABLE=1 "$PY" - <<'PY' || gpu_status=$?
import os
import sys

device = os.environ.get("EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"
visible = os.environ.get("GPU_VISIBLE") == "1"

if device != "cpu":
    # First, on purpose: importing it pins CUDA's card numbering to the PCI
    # bus the way a run does, so cuda:N here is the card a run will use.
    from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME, _embedder_on
import torch

hip = getattr(torch.version, "hip", None)
build = f"CUDA {torch.version.cuda}" if torch.version.cuda else (f"ROCm {hip}" if hip else None)
if build is None:
    if device != "cpu" or visible:
        why = f"EMBEDDING_DEVICE={device} names a card" if device != "cpu" else "nvidia-smi lists a card"
        print(f"  ✗ torch {torch.__version__} is the CPU build, and {why}: the GPU build "
              "is the machine's own setup, and it is not in .venv")
        sys.exit(1)
    print(f"  ✓ torch {torch.__version__}, CPU build (no card here, EMBEDDING_DEVICE=cpu)")
    sys.exit(0)

count = torch.cuda.device_count()
if count == 0:
    if device == "cpu" and not visible:
        print(f"  ✓ torch {torch.__version__} ({build}); no card here, and EMBEDDING_DEVICE=cpu needs none")
        sys.exit(0)
    print(f"  ✗ torch {torch.__version__} ({build}) sees no card: is the driver loaded?")
    sys.exit(1)
print(f"  ✓ torch {torch.__version__} ({build}) sees {count} card(s)")

if device == "cpu":
    indexes = list(range(count))
else:
    kind, _, number = device.partition(":")
    if kind != "cuda" or not number.isdigit():
        print(f"  ✗ EMBEDDING_DEVICE={device} names neither cpu nor cuda:N")
        sys.exit(1)
    indexes = [int(number)]

status = 0
for index in indexes:
    name = f"cuda:{index}"
    try:
        major, minor = torch.cuda.get_device_capability(index)
        label = f"{torch.cuda.get_device_name(index)}, compute {major}.{minor}"
        square = torch.ones(256, 256, device=name)
        (square @ square).sum().item()
    except Exception as exc:  # no such card, or a build with no kernels for it
        text = str(exc).strip()
        print(f"  ✗ {name} does not compute: {(text.splitlines() or [type(exc).__name__])[0][:200]}")
        status = 1
        continue
    print(f"  ✓ {name} computes ({label})")

if device == "cpu":
    # Imported only now: on cpu it hides every card, and CUDA is initialised.
    from langgraph_agent.graphrag_server import EMBEDDING_MODEL_NAME

    print(f"  (EMBEDDING_DEVICE=cpu, so {EMBEDDING_MODEL_NAME} embeds on the CPU when it is the model)")
elif status == 0:
    model, why, busy = _embedder_on(device)
    if model is not None:
        print(f"  ✓ {EMBEDDING_MODEL_NAME} initialised on {device}, warmed to an index's working size")
    elif busy:
        print(f"  ! {why}: something else holds the card (ollama ps, nvidia-smi)")
        status = 3
    else:
        print(f"  ✗ {why}")
        status = 1
sys.exit(status)
PY
case "$gpu_status" in
    0) ;;
    3) NOTES+=("the embedding card was busy, so the embedder could not be placed on it; see GPU above") ;;
    *) problem "the GPU build does not initialise as configured; see GPU above" ;;
esac

# ---------------------------------------------------------------------------
# 7. Git, for the Builder's git_dwell
# ---------------------------------------------------------------------------

step "Git (the Builder's git_dwell)"
# git_dwell's default pipeline commits in this checkout, pushes, and opens and
# merges a pull request with gh. None of that is exercised until a Builder
# reaches it mid-run, where a missing identity or a signed-out gh stops the
# pipeline at that stage, and the Builder spends its turns on a repository
# that was never broken. So both are asked now.
if ! command -v git >/dev/null; then
    problem "git is not installed (drop --no-system, or: sudo pacman -S git)"
else
    # Asked inside this checkout, so a repo-local identity counts as well.
    git_name="$(git config user.name || true)"
    git_email="$(git config user.email || true)"
    if [ -n "$git_name" ] && [ -n "$git_email" ]; then
        ok "commits are authored as $git_name <$git_email>"
    else
        problem "git has no identity, so the Builder cannot commit: git config --global user.name 'Your Name' && git config --global user.email you@example.com"
    fi

    origin="$(git remote get-url origin 2>/dev/null || true)"
    if [ -z "$origin" ]; then
        # A checkout with no remote is legitimate; the pipeline just ends early.
        echo "  no origin remote: git_dwell will commit on a branch and stop at push"
    elif [[ "$origin" != *github.com[:/]* ]]; then
        echo "  origin is not on github.com: git_dwell's pr and merge stages use gh and will stop there"
    elif ! command -v gh >/dev/null; then
        problem "gh is not installed, so git_dwell stops at its pr stage (drop --no-system, or: sudo pacman -S github-cli)"
    else
        gh_signed_in() { gh auth status --hostname github.com >/dev/null 2>&1; }
        if ! gh_signed_in && interactive; then
            echo "  gh is not signed in to github.com; git_dwell opens and merges pull requests with it"
            if gh auth login --hostname github.com --git-protocol https; then
                # Makes gh git's credential helper for github.com, which is
                # what lets git_dwell's plain `git push` authenticate. Only
                # after a sign-in this script started: an existing setup's
                # credentials are not this script's to rewrite.
                gh auth setup-git --hostname github.com || true
            fi
        fi
        if gh_signed_in; then
            ok "gh signed in to github.com (git_dwell can open and merge pull requests)"
        else
            problem "gh is not signed in: run 'gh auth login', then re-run ./install.sh"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 8. SearxNG, for online research
# ---------------------------------------------------------------------------

if [ "$SEARXNG" -eq 1 ]; then
    step "SearxNG (online research)"
    # Without it, online research goes through DuckDuckGo's keyless endpoint,
    # which answers a handful of quick requests from one address with its bot
    # check and then nothing for about half an hour -- and a goal fans out into
    # several searches. A SearxNG of your own asks many engines and blocks
    # nobody. It runs as a rootless podman container under your user's
    # systemd, so it starts with your session, binds loopback only, and needs
    # no root after the packages above.
    #
    # Two files, both yours afterwards. settings.yml is written once and never
    # overwritten; the quadlet unit is rewritten only when it differs.
    # FORCE_OWNERSHIP=false and a read-only file mount matter together: the
    # image's entrypoint otherwise chowns its config volume to its own user,
    # which under rootless podman leaves the host file owned by a subordinate
    # uid you cannot edit.
    searxng_conf="${XDG_CONFIG_HOME:-$HOME/.config}/ambiguity/searxng"
    searxng_unit="${XDG_CONFIG_HOME:-$HOME/.config}/containers/systemd/ambiguity-searxng.container"
    searxng_image="docker.io/searxng/searxng:latest"
    searxng_up() { [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$1/healthz" || true)" = "200" ]; }

    if [ -n "${SEARXNG_URL:-}" ] && [ "${SEARXNG_URL%/}" != "$SEARXNG_LOCAL" ]; then
        echo "  SEARXNG_URL names $SEARXNG_URL; using that instance instead of running one"
    elif ! command -v podman >/dev/null; then
        problem "podman is not installed, so SearxNG cannot run (drop --no-system, or: sudo pacman -S podman crun)"
    elif ! grep -q "^$(id -un):" /etc/subuid 2>/dev/null; then
        problem "rootless podman needs subordinate ids for $(id -un): sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(id -un)"
    elif ! systemctl --user show-environment >/dev/null 2>&1; then
        problem "no systemd user session to run SearxNG under (log in to a desktop or console session, then re-run)"
    else
        changed=0
        mkdir -p "$searxng_conf"
        if [ ! -f "$searxng_conf/settings.yml" ]; then
            secret="$("$PY" -c 'import secrets; print(secrets.token_hex(32))')"
            cat >"$searxng_conf/settings.yml" <<EOF
# Written by install.sh for the Ambiguity console's online research, and never
# overwritten by it: edit freely, then systemctl --user restart ambiguity-searxng.
# Only what differs from SearxNG's own defaults is here.
use_default_settings: true

general:
  instance_name: "Ambiguity research"

server:
  secret_key: "$secret"
  # One local caller, not a public instance. The limiter would rate-limit the
  # searches a single goal fans out into, and needs a valkey server besides.
  limiter: false
  public_instance: false
  image_proxy: false

search:
  # web_research asks for JSON. SearxNG allows only html by default and answers
  # any other format with 403.
  formats:
    - html
    - json
EOF
            chmod 644 "$searxng_conf/settings.yml"
            changed=1
            ok "wrote $searxng_conf/settings.yml"
        elif ! grep -qE '^[[:space:]]*-[[:space:]]*json[[:space:]]*$' "$searxng_conf/settings.yml"; then
            problem "$searxng_conf/settings.yml does not list json under search.formats, and the app asks SearxNG for JSON"
        else
            ok "settings.yml present; left as it is"
        fi

        unit_text="$(cat <<EOF
# Written by install.sh for the Ambiguity console's online research.
[Unit]
Description=SearxNG for the Ambiguity console's online research
Wants=network-online.target
After=network-online.target

[Container]
Image=$searxng_image
ContainerName=ambiguity-searxng
PublishPort=127.0.0.1:$SEARXNG_PORT:8080
Volume=$searxng_conf/settings.yml:/etc/searxng/settings.yml:ro
Volume=ambiguity-searxng-data:/var/cache/searxng
Environment=FORCE_OWNERSHIP=false
Environment=SEARXNG_BASE_URL=$SEARXNG_LOCAL/

[Service]
Restart=on-failure
TimeoutStartSec=300

[Install]
WantedBy=default.target
EOF
        )"
        if [ "$(cat "$searxng_unit" 2>/dev/null || true)" != "$unit_text" ]; then
            mkdir -p "$(dirname "$searxng_unit")"
            printf '%s\n' "$unit_text" >"$searxng_unit"
            changed=1
            ok "wrote $searxng_unit"
        fi

        # Pulled here rather than by the unit's first start, which systemd
        # would time out on a slow connection.
        if podman image exists "$searxng_image"; then
            ok "$searxng_image present"
        elif podman pull --quiet "$searxng_image" >/dev/null; then
            ok "pulled $searxng_image"
        else
            problem "could not pull $searxng_image"
        fi

        systemctl --user daemon-reload || true
        if [ "$changed" -eq 1 ]; then
            systemctl --user restart ambiguity-searxng.service || true
        else
            systemctl --user start ambiguity-searxng.service || true
        fi
        for _ in $(seq 1 60); do searxng_up "$SEARXNG_LOCAL" && break; sleep 1; done

        if searxng_up "$SEARXNG_LOCAL"; then
            ok "SearxNG answering at $SEARXNG_LOCAL (starts with your session)"
            # The app uses SearxNG only when SEARXNG_URL says so, and only once
            # it answers is it safe to say so: a URL to nothing would fail every
            # search that DuckDuckGo would at least have tried.
            if [ -z "${SEARXNG_URL:-}" ]; then
                printf '\n# Added by install.sh: the SearxNG it runs for online research.\nSEARXNG_URL=%s\n' \
                    "$SEARXNG_LOCAL" >>.env
                export SEARXNG_URL="$SEARXNG_LOCAL"
                ok "SEARXNG_URL=$SEARXNG_LOCAL added to .env"
            fi
        else
            problem "SearxNG did not answer at $SEARXNG_LOCAL; see: journalctl --user -u ambiguity-searxng"
        fi
    fi

    # Whichever instance SEARXNG_URL names, it is asked one question through
    # the request the app itself sends, so "answering" means answering the app:
    # the JSON format enabled and the reply in the shape it parses.
    if [ -n "${SEARXNG_URL:-}" ]; then
        searx_status=0
        "$PY" - <<'PY' || searx_status=$?
import sys

import httpx

from langgraph_agent import web_research

try:
    with httpx.Client() as client:
        hits = web_research._search_searxng("python packaging", 5, client)
except httpx.HTTPStatusError as exc:
    code = exc.response.status_code
    hint = " (403: json is not in the instance's search.formats)" if code == 403 else ""
    print(f"  ✗ {web_research.SEARXNG_URL} refused the app's search: HTTP {code}{hint}")
    sys.exit(1)
except (httpx.HTTPError, ValueError) as exc:
    print(f"  ✗ {web_research.SEARXNG_URL} could not answer the app's search: {type(exc).__name__}: {exc}")
    sys.exit(1)
if not hits:
    print(f"  ! {web_research.SEARXNG_URL} answered with no results: its engines may be refusing it for now")
    sys.exit(3)
print(f"  ✓ a search through the app returned {len(hits)} result(s)")
PY
        case "$searx_status" in
            0) ;;
            3) NOTES+=("SearxNG answered a test search with no results; see SearxNG above") ;;
            *) problem "online research cannot use SEARXNG_URL; see SearxNG above" ;;
        esac
    fi
fi

# ---------------------------------------------------------------------------
# 9. Checks
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
# 10. App launcher
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
# Summary: can the embedder and each seat actually run?
# ---------------------------------------------------------------------------

step "Embedder"
# Pulling a model proves it is on the daemon, not that it embeds, and a run
# would find out only in its corpus phase. So the active model is asked for one
# vector through the class a run uses -- no corpus is opened or built -- and
# where the daemon placed it is read back: a split reply is identical to a whole
# one, and the split is what turns a first build of minutes into hours
# (OLLAMA_EMBED_OPTIONS in graphrag_server.py has the measurements). A model
# this check loaded is unloaded again, so the daemon is left holding what it
# held and the next run's GPU placement is not decided by an install.
embed_status=0
"$PY" - <<'PY' || embed_status=$?
import subprocess
import sys

from langgraph_agent.config import ollama_cpu_share
from langgraph_agent.graphrag_server import OllamaEmbedder, active_embedding_model, embedding_backend

model = active_embedding_model()
if embedding_backend(model) != "ollama":
    print(f"  ✓ {model} runs in this process (placed under GPU above)")
    sys.exit(0)

loaded_before = ollama_cpu_share(model) is not None
embedder = OllamaEmbedder(model)
try:
    vector = embedder.encode("install check")
except RuntimeError as exc:
    print(f"  ✗ {exc}")
    sys.exit(1)
print(f"  ✓ {model} embeds ({len(vector)} dimensions)")

status = 0
share = embedder.cpu_share
if share is None:
    print(f"  (the daemon did not say where it placed {model})")
elif share >= 0.99:
    print(f"  ✗ {model} is wholly on the CPU: the daemon initialised no GPU for it, so a first "
          "corpus build takes hours. Ollama's GPU backend is the machine's own setup.")
    status = 1
elif share > 0:
    print(f"  ! {embedder.placement_note}")
    status = 3
else:
    print(f"  ✓ {model} is wholly on the GPU")

if not loaded_before:
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass
sys.exit(status)
PY
case "$embed_status" in
    0) ;;
    3) NOTES+=("the embedding model is not wholly on a GPU; see Embedder above") ;;
    *) problem "the embedder cannot run as configured; see Embedder above" ;;
esac

step "Seats"
# get_agent_status is the one thing that knows whether a seat is live, stubbed
# or failing, so it is asked rather than inferred from what was installed. But
# a tag on the daemon and a signed-in account are presence, not liveness -- the
# account may have no cloud access, or be rate limited -- so each live seat is
# first sent one short prompt through its real call path, in parallel, and a
# failure lands in get_agent_status the way it would mid-run.
if ! PROBE="$PROBE" "$PY" - <<'PY'
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from langgraph_agent.config import AGENTS, get_agent_llm, get_agent_status

silent: set[str] = set()


def probe(agent: str) -> None:
    try:
        reply = get_agent_llm(agent).invoke("Reply with the single word: ready")
    except Exception:
        return  # the seat wrapper recorded why; get_agent_status reports it below
    if not str(getattr(reply, "content", reply)).strip():
        silent.add(agent)


if os.environ.get("PROBE") == "1":
    live = [agent for agent in AGENTS if get_agent_status(agent)["live"]]
    if live:
        print("  (each live seat was sent a one-word prompt)")
        with ThreadPoolExecutor(len(live)) as pool:
            list(pool.map(probe, live))

dead = 0
for agent in AGENTS:
    seat = get_agent_status(agent)
    if not seat["live"]:
        mark, note = "✗", f"  -- {seat['badge']}: {seat['reason']}"
    elif agent in silent:
        mark, note = "✗", "  -- SILENT: answered a one-word prompt with nothing"
    else:
        mark, note = "✓", ""
    print(f"  {mark} {agent:11}{seat['model']:24}{seat['provider']}{note}")
    dead += mark == "✗"
sys.exit(1 if dead else 0)
PY
then
    problem "not every seat can run; see the list above"
fi

print_notes() {
    [ "${#NOTES[@]}" -eq 0 ] && return 0
    echo "  Worth knowing:"
    for n in "${NOTES[@]}"; do echo "    - $n"; done
}

echo
echo "============================================"
if [ "${#PROBLEMS[@]}" -eq 0 ]; then
    echo "  Ready. Start the console with:"
    echo "    ./launch_console.sh"
    [ "$DESKTOP" -eq 1 ] && echo "  or pick \"Ambiguity Console\" from the app launcher."
    print_notes
    echo "============================================"
    exit 0
fi
echo "  Installed, with ${#PROBLEMS[@]} thing(s) left to fix:"
for p in "${PROBLEMS[@]}"; do echo "    - $p"; done
print_notes
echo "  Fix them and re-run ./install.sh; finished steps are skipped."
echo "============================================"
exit 1

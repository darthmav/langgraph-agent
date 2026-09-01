# Per-agent LLM dropdowns (cloud-only)

## Goal

Each agent panel (Planner, Researcher, Builder) lets the user pick any available cloud LLM option. Only Anthropic and OpenAI providers are supported.

## Current state

- Each agent panel has a `<select class="llm-select">` dropdown.
- The dropdown is populated from `GET /api/llm-options` (static Anthropic/OpenAI options).
- `POST /api/set-llm` accepts `{agent, provider, model}` and supports `provider: "anthropic"` and `provider: "openai"`.

## Implementation

### `src/langgraph_agent/config.py`

- `AGENT_LLM_OPTIONS` contains only Anthropic and OpenAI options.

### `serve.py`

- `GET /api/llm-options` returns the static cloud options.
- `GET /api/status` reports the active provider/model per agent.

### `frontend/index.html`

- `loadLLMOptions()` loads cloud options from `/api/llm-options`.
- `updateLLMSelect()` groups options by provider.
- Selecting an option POSTs to `/api/set-llm`; the sidebar updates on the next status poll.

## Verification

1. `ruff check src/ tests/ serve.py scripts/ example_usage.py test_cloud.py` passes.
2. `mypy src/langgraph_agent/ serve.py` passes.
3. `python -m pytest tests/ -v` passes.

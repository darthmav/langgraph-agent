# Add Ollama model dropdowns in agent panels

## Goal

Extend the per-agent LLM dropdowns in the console so each agent can be assigned any model that is currently available from the configured Ollama server, alongside the existing cloud LLM options.

## Current state

- Each agent panel (Planner, Researcher, Builder) already has a `<select class="llm-select">` dropdown.
- The dropdown is populated from `GET /api/llm-options`, which returns only the static `AGENT_LLM_OPTIONS` cloud providers (OpenAI, Anthropic, Kimi).
- `POST /api/set-llm` already accepts `{agent, provider, model}` and supports `provider: "ollama"`.

## Proposed changes

### `src/langgraph_agent/config.py`

- Add `get_ollama_base_url()` helper that returns `OLLAMA_BASE_URL` or the default `http://localhost:11434`.
- Add `list_ollama_models()` helper:
  - `GET <base_url>/api/tags`
  - Parse `models[].model` (e.g. `qwen3:8b`, `gemma4:27b`).
  - Return a list of dicts: `[{"name": "qwen3:8b", "label": "qwen3:8b (Ollama)"}, ...]`.
  - On failure (Ollama unreachable, malformed response), return an empty list and log a single concise warning.
- Do not change `AGENT_LLM_OPTIONS`; the frontend will merge cloud and Ollama options.

### `serve.py`

- Add `GET /api/ollama-models` endpoint:
  - Calls `list_ollama_models()`.
  - Returns `{"models": [{"name": "...", "label": "..."}]}`.
- Keep the existing `/api/llm-options` endpoint unchanged so cloud options remain available.
- Update `/api/status` so the `selected` block reflects the current per-agent provider/model (already implemented); no further change needed.

### `frontend/index.html`

- Extend `loadLLMOptions()`:
  - Fetch both `/api/llm-options` and `/api/ollama-models`.
  - Combine into a single option list grouped by source:
    - Cloud LLMs (from `/api/llm-options`) as before.
    - Ollama models (from `/api/ollama-models`) using provider `"ollama"`.
  - Cache the merged list in `llmOptions`.
- Update `updateLLMSelect()`:
  - Add an `<optgroup label="Cloud">` for cloud options.
  - Add an `<optgroup label="Ollama">` for Ollama models.
  - Each `option.value` stays as `JSON.stringify({provider, model})`.
  - Preserve focus/value during re-renders, as it already does.
- No change needed for `setAgentLLM()` or the `change` event wiring; Ollama selections work with the existing `POST /api/set-llm` path.

## API additions

### `GET /api/ollama-models`

```json
{
  "models": [
    {"name": "qwen3:8b", "label": "qwen3:8b"},
    {"name": "gemma4:27b", "label": "gemma4:27b"},
    {"name": "llama3.1:8b", "label": "llama3.1:8b"}
  ]
}
```

If Ollama is unreachable:

```json
{"models": []}
```

## UI behavior

- On page load, the console fetches cloud options and Ollama models, then builds the dropdowns.
- Each dropdown shows:
  - **Cloud** group with OpenAI/Anthropic/Kimi options.
  - **Ollama** group with locally available models.
- Selecting an option immediately POSTs to `/api/set-llm`.
- The sidebar and architecture diagram labels update on the next `/api/status` poll (already in place).
- If Ollama is not running, the Ollama group is simply empty; the rest of the UI works normally.

## Verification

1. `ruff check src/ tests/ serve.py` passes.
2. `mypy src/langgraph_agent/ serve.py` passes.
3. `OPENAI_API_KEY="" python -m pytest tests/ -v` passes.
4. With Ollama running, `GET /api/ollama-models` returns the installed models.
5. With Ollama not running, the endpoint returns `{"models": []}` without crashing the server.
6. The console dropdowns render cloud and Ollama groups, and selecting an Ollama model sets the agent to `provider: "ollama"` with the chosen model.

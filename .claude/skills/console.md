---
name: console
description: Start the web console for the 4-agent system.
---

# Start the web console

Launch the frontend server. It serves the SPA at `frontend/index.html` and exposes the `/api/*` endpoints from `serve.py`.

```bash
./launch_console.sh
```

Then open http://localhost:8080 in a browser.

The script loads `.env` if present and starts the server. The default backend is Anthropic cloud models; configure `ANTHROPIC_API_KEY` in `.env` for real agent runs. OpenAI remains optional.

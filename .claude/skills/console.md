---
name: console
description: Start the web console for the 3-agent system.
---

# Start the web console

Launch the frontend server. It serves the SPA at `frontend/index.html` and exposes the `/api/*` endpoints from `serve.py`.

```bash
./launch_console.sh
```

Then open http://localhost:8080 in a browser.

The script loads `.env` if present and checks whether Ollama is running. If Ollama is not running, it still starts the server so the UI is inspectable, but agent runs will fall back to the stub LLM unless cloud API keys are configured.

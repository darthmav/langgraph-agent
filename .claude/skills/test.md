---
name: test
description: Run the full pytest suite for the 3-agent system.
---

# Run tests

Run the test suite with the stub LLM so no API keys or local Ollama instance is needed.

```bash
OPENAI_API_KEY="" python -m pytest tests/ -v
```

After running, report the pass/fail count and any warnings worth noting.

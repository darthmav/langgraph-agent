# 3-Agent Console - Frontend

A beautiful, modern web interface for the 3-Agent AI System.

## Visual Design

**Part 2 Revamp** features:
- **Animated gradient backgrounds** with subtle shifts
- **Agent-specific colors**: Planner (blue), Researcher (green), Builder (pink/red)
- **Glowing effects** on active elements
- **Smooth animations** for messages and state updates
- **Dark professional theme** optimized for long sessions
- **Responsive layout** with sidebar navigation

## Quick Start

### Option 1: Python HTTP Server (Recommended)

```bash
# Terminal 1: Start the API server
cd /home/darthmaverus/projects/ambiguity2
python serve.py

# Terminal 2: Open in browser
open http://localhost:8080
# Or: xdg-open http://localhost:8080  # Linux
```

### Option 2: Direct File Access

```bash
# Just open the HTML file directly in your browser
open frontend/index.html
```

Note: Direct file access won't have API connectivity. Use Option 1 for full functionality.

## Features

### Overview Panel
- Enter goals and run the 3-Agent system
- Live message feed showing agent activity
- Clear button to reset state

### Agent Panels
- **Planner**: View current plan and next agent routing
- **Researcher**: Query GraphRAG knowledge base
- **Builder**: See implementation reports and files changed
- **State**: Complete system state visualization

### Sidebar
- Real-time agent status cards with progress bars
- LLM configuration display
- GraphRAG index status
- Color-coded by agent role

### Keyboard Shortcuts
- `Enter` in goal input: Run
- `F5`: Refresh status

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/run` | POST | Run 3-Agent system with a goal |
| `/api/search` | POST | Search GraphRAG knowledge base |
| `/api/status` | GET | Get system status (LLM, GraphRAG) |
| `/api/state` | GET | Get current agent state |
| `/api/llm-options` | GET | Available cloud LLM options |
| `/api/set-llm` | POST | Set per-agent LLM provider/model |

### Example API Usage

```bash
# Run a task
curl -X POST http://localhost:8080/api/run \
  -H "Content-Type: application/json" \
  -d '{"goal": "Create hello.txt with Hello World"}'

# Search knowledge base
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Planner agent", "top_k": 3}'

# Check status
curl http://localhost:8080/api/status
```

## Customization

### Colors
Edit CSS variables in `frontend/index.html`:

```css
:root {
  --planner: #4ea3ff;      /* Agent colors */
  --researcher: #76d13a;
  --builder: #ff6a8a;

  --bg: #0a0b0d;           /* Background colors */
  --bg-2: #101216;
  --bg-3: #15181e;
}
```

### Port
```bash
export PORT=3000
python serve.py
```

## Troubleshooting

**Server won't start:**
- Check if port 8080 is in use: `lsof -i :8080`
- Try a different port: `PORT=3001 python serve.py`

**API not responding:**
- Ensure 3-Agent system is installed: `pip install -e ".[dev]"`
- Check `.env` configuration for LLM settings
- The default uses Anthropic cloud models

**GraphRAG not indexed:**
- Run: `python scripts/index_knowledge.py`
- Check `knowledge/chroma/` directory exists

## Architecture

```
frontend/
  index.html          # Single-page app (HTML + CSS + JS)
serve.py              # Python HTTP server + API backend
```

The frontend is a **zero-build** single-page application:
- No npm/webpack required
- No React/Vue dependencies
- Pure HTML/CSS/vanilla JS
- ~600 lines total

## Credits

Visual design inspired by the original Ambiguity console, revamped for the 3-Agent system with:
- Enhanced animations
- Agent-specific theming
- Real-time state visualization
- Modern gradient effects

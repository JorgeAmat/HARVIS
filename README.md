# HARVIS v0.2A

HARVIS is a local conversational assistant using Groq and one restricted arithmetic
tool, `calculate(expression)`. The terminal test persists conversation state in
SQLite. Chainlit still keeps only temporary session context; it does not resume
saved conversations. There is no Obsidian integration or cross-thread memory.

## Architecture

```text
Chainlit -> LangGraph -> Groq -> Tools
```

- `app.py`: Chainlit UI, temporary chat history, and agent/tool execution steps.
- `graph.py`: agent brain, message state, Groq model, calculation tool, and routing.
- `main.py`: independent SQLite persistence test without the UI.
- `tests/`: offline arithmetic, routing, UI-output, and SQLite persistence tests.

The graph loops from agent to tools and back to agent until it produces an answer.
Only tool selections, arguments, results, and final answers are displayed.

## Setup (Windows PowerShell)

Use Python 3.13, the version used for local validation. From the repository folder:

```powershell
cd "C:\Users\jorge\Desktop\HARVIS"
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements pin direct dependencies; transitive dependencies are not locked.
If activation is blocked by PowerShell policy, use `.\.venv\Scripts\python.exe`
instead of `python` in the commands below; activation is not required.

Create `.env` in the repository root with your own Groq API key:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
```

Keep this file private. Messages and tool results are sent to Groq for processing.
The model is configured in `graph.py` as `openai/gpt-oss-20b`.

## Run

Test the graph independently (requires internet and a valid key):

```powershell
python -X utf8 main.py jorge-test "My secret project codeword is blue lantern."
python -X utf8 main.py jorge-test "What project codeword did I give you?"
python -X utf8 main.py another-thread "What project codeword did I give you?"
```

Each command starts a separate Python process. Reusing a thread ID restores its
history; a new ID starts isolated state. Send only the new message each time.
Choose a fresh ID for a clean test: existing IDs retain earlier runs.

`data/checkpoints.sqlite` is resolved relative to `graph.py`, regardless of the
terminal working directory. The async saver opens the connection, checkpoints
the existing graph, and closes the connection on exit. Stored conversation data
is private, unencrypted, and ignored by Git; keep the database local and trusted.
This milestone tests completed turns; unfinished runs are reported, not resumed
automatically by the terminal test.

Launch the unchanged, session-only UI locally:

```powershell
python -X utf8 -m chainlit run app.py --headless --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Stop with Ctrl+C. File uploads and MCP are disabled.
Try: "Tengo 73 acciones a 181.4 euros. El precio sube un 15%. ¿Cuánto vale mi posición?"
Inspect the Agent and Tool steps; the result should be approximately 15,228.53 euros.

## Checks

```powershell
python -X utf8 -B -m unittest discover -s tests -v
python -m pip check
```

The tests mock the model and make no API calls. `.env` variants, `.venv/`, `.files/`,
`HARVIS_MEMORY/`, and `data/` are ignored by Git. Chainlit configuration is tracked;
generated translations remain ignored.

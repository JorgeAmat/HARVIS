# HARVIS v0.3A.1

HARVIS is a local conversational assistant using Groq and one restricted arithmetic
tool, `calculate(expression)`. Chainlit accepts typed text or microphone speech
transcribed by Groq Whisper; replies remain text-only. Both interfaces persist
conversation state in SQLite. Each new Chainlit chat gets its own UUID; use
`/resume <thread_id>` to continue a saved conversation after restarting. There is no Obsidian
integration, global personal memory, or cross-thread memory.

## Architecture

```text
Chainlit -> LangGraph -> Groq -> Tools
```

- `app.py`: Chainlit UI, active thread ID, and agent/tool execution steps.
  LangGraph owns message history; the UI sends only each new user message.
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

Launch the persistent UI locally:

```powershell
python -X utf8 -m chainlit run app.py --headless --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Stop with Ctrl+C. File uploads and MCP are disabled.
Try: "Tengo 73 acciones a 181.4 euros. El precio sube un 15%. ¿Cuánto vale mi posición?"
Inspect the Agent and Tool steps; the result should be approximately 15,228.53 euros.

## New chats and resuming saved conversations

Every new chat displays `HARVIS v0.3A.1 online` and a fresh **Conversation ID**.
Copy that ID if you want to resume later. A UUID is selected immediately; its
conversation becomes saved once you send a normal message.

1. Start Chainlit, open http://127.0.0.1:8000, and copy conversation A's ID.
2. Say: "Mi animal de prueba es el koala." Wait for the response to finish.
3. Click New Chat. Conversation B must have a different ID.
4. Ask: "¿Cuál era mi animal de prueba?" It should not know A's animal.
5. Send `/resume <A's ID>` (replace the placeholder with the copied ID).
6. After the confirmation, ask again. HARVIS should answer "koala".
7. Stop the server with Ctrl+C, restart it, and open a new chat.
8. Send `/resume <A's ID>` again, then ask the animal question to verify recovery.

`/resume` reads LangGraph state and switches only if a saved checkpoint exists.
It does not call Groq, execute tools, replay the old transcript, or add the command
to conversation history. An unknown ID or malformed command leaves the active
conversation unchanged. IDs need not be UUIDs for resume: old saved conversations
such as `chainlit-main` remain accessible with `/resume chainlit-main`.

To test a missing ID, use `/resume 00000000-0000-0000-0000-000000000000` (assuming
you have never used that ID). Expect a clear error, with the active thread unchanged.
Sending `/resume` without an ID should display the command's usage.

Chainlit stores the active `thread_id` and temporary audio recording state in its
user session, never full message history. Normal turns send
only the new message; LangGraph restores and saves history in SQLite. Agent/tool
steps stay visible. This remains a local app; thread IDs select conversations and
are not user authentication or an access-control system.

Print the database path from the repository folder:

```powershell
python -c "from graph import CHECKPOINT_PATH; print(CHECKPOINT_PATH)"
```

It should be `C:\Users\jorge\Desktop\HARVIS\data\checkpoints.sqlite`.
To inspect the file without reading conversation contents:

```powershell
Get-Item .\data\checkpoints.sqlite | Select-Object FullName, Length, LastWriteTime
```

## Microphone input (v0.3A.1)

Open the local Chainlit UI and allow microphone access. Press the microphone
control and speak a short message. A small panel displays **Recording...** with
**X Cancel** and **✓ Finish / Send**. Cancel immediately clears audio locally and
stops without a Groq request or agent turn. Finish explicitly approves upload,
then stops recording and transcribes once. The native Chainlit X and P shortcut
discard recordings unless Finish has explicitly approved sending first.
Leave a brief pause after your final word before Finish: Chainlit batches samples.
The panel waits for Python to acknowledge its recording-specific decision before
stopping through Chainlit's audio API, then disappears. It does not replace the
native composer control or use DOM selectors. Old panel IDs cannot affect a new
recording. HARVIS uploads only explicitly approved recordings to Groq `whisper-large-v3-turbo`, displays
the transcript as "You (transcribed)", and runs the same persistent agent used
for typed text. Language is detected automatically. Responses are text-only.

Recordings are limited to 60 seconds of mono PCM16 at 24,000 Hz. Audio is held
temporarily in RAM by HARVIS and wrapped as WAV in memory, never deliberately
written to disk or SQLite. Your voice and any recorded background speech are sent
to Groq for transcription. The transcript becomes a normal, persistent message
in the thread selected when recording started. That transcript and conversation
context are then sent to Groq's chat model. Buffer cleanup is not a guarantee
about Groq's retention of submitted audio.

Typed `/resume` remains a UI command; a spoken transcript containing `/resume`
is ordinary agent input and does not switch threads. Empty/invalid/oversized
recordings, empty transcripts, and transcription API failures produce an error
without invoking the agent. Finish one recording and response before starting
another. There is no text-to-speech, VAD, or realtime transcription.

Manual check:

1. Start a new chat and copy its conversation ID.
2. Record: "Mi animal de prueba es el koala." Click **✓ Finish / Send** and check the transcript.
3. Type: "¿Cuál era mi animal de prueba?" Expect "koala".
4. Record a short arithmetic request, click Finish, and inspect the Agent/Tool steps.
   Record another phrase and click Cancel: no transcript or agent steps should appear.
   Repeat using the native X or P shortcut outside the text input: it should discard.
5. Start a new chat and verify it does not inherit the animal.
6. Type `/resume <original ID>`, then ask the animal question by microphone.
7. Restart Chainlit, resume the same ID, and verify the transcribed fact persists.
8. Test an empty/very short recording and denied microphone permission. No
   successful transcription should mean no agent turn. Also check your final
   spoken words are present in the transcript.

## Checks

```powershell
python -X utf8 -B -m unittest discover -s tests -v
python -m pip check
```

The tests mock the model and make no API calls. `.env` variants, `.venv/`, `.files/`,
`HARVIS_MEMORY/`, and `data/` are ignored by Git. Chainlit configuration is tracked;
generated translations remain ignored.

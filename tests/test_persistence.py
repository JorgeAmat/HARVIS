import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "offline-test-key")

from langchain_core.messages import AIMessage, ToolMessage
import graph as brain


# Each invocation runs in a separate Python process with a scripted model.
# SQLite, graph execution, message merging and checkpoint serialization are real.
WORKER = r"""
import asyncio, json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from langchain_core.messages import AIMessage, HumanMessage
import graph as brain

async def run():
    config = {"configurable": {"thread_id": sys.argv[3]}}
    seen = []
    def reply(messages):
        seen.extend(m.content for m in messages if isinstance(m, HumanMessage))
        return AIMessage(content="Acknowledged.")
    async with brain.open_persistent_graph(Path(sys.argv[2])) as graph:
        before = await graph.aget_state(config)
        with patch.object(brain, "model_with_tools") as model:
            model.invoke.side_effect = reply
            result = await graph.ainvoke(
                {"messages": [("user", sys.argv[4])]}, config=config
            )
        print(json.dumps({
            "before": [m.content for m in before.values.get("messages", [])],
            "after": [m.content for m in result["messages"]],
            "ids": [m.id for m in result["messages"]],
            "model_users": seen,
            "default_db": str(brain.CHECKPOINT_PATH),
        }))
asyncio.run(run())
"""


# Run the actual Chainlit handlers with mocked transport and model, but real SQLite.
UI_WORKER = r"""
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
sys.path.insert(0, sys.argv[1])
from langchain_core.messages import AIMessage, HumanMessage
import graph as brain
import app

async def run():
    session = {}
    seen = []
    sent = []
    def reply(messages):
        seen.extend(m.content for m in messages if isinstance(m, HumanMessage))
        return AIMessage(content="koala" if any("koala" in text for text in seen) else "Unknown")
    def ui_message(content):
        sent.append(content)
        return SimpleNamespace(send=AsyncMock())
    step = MagicMock()
    step.__aenter__.return_value = step
    with patch.object(app.cl, "user_session", SimpleNamespace(get=session.get, set=session.__setitem__)), \
         patch.object(app.cl, "Step", return_value=step), \
         patch.object(app.cl, "Message", side_effect=ui_message), \
         patch.object(app, "open_persistent_graph", lambda: brain.open_persistent_graph(Path(sys.argv[2]))), \
         patch.object(brain, "model_with_tools") as model:
        model.invoke.side_effect = reply
        await app.on_chat_start()
        if sys.argv[4]:
            await app.on_message(SimpleNamespace(content="/resume " + sys.argv[4]))
            model.invoke.assert_not_called()
        await app.on_message(SimpleNamespace(content=sys.argv[3]))
    async with brain.open_persistent_graph(Path(sys.argv[2])) as graph:
        state = await graph.aget_state({"configurable": {"thread_id": session["thread_id"]}})
    print(json.dumps({"session": session, "seen": seen, "final": sent[-1],
                      "ids": [m.id for m in state.values["messages"]]}))
asyncio.run(run())
"""


class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    def test_separate_processes_restore_isolate_and_do_not_duplicate(self):
        project = Path(brain.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite"

            def turn(thread_id, message):
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-B", "-c", WORKER,
                     str(project), str(database), thread_id, message],
                    cwd=directory, capture_output=True, text=True,
                    encoding="utf-8", timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout.splitlines()[-1])

            first = turn("jorge-test", "My secret project codeword is blue lantern.")
            second = turn("jorge-test", "What project codeword did I give you?")
            other = turn("isolated-test", "What project codeword did I give you?")

            self.assertTrue(database.is_file())
            self.assertEqual(first["before"], [])
            self.assertEqual(len(first["after"]), 2)
            self.assertEqual(second["before"], first["after"])
            self.assertIn(first["after"][0], second["model_users"])
            self.assertEqual(len(second["after"]), 4)
            self.assertEqual(second["ids"][:2], first["ids"])
            self.assertEqual(len(set(second["ids"])), 4)
            self.assertEqual(other["before"], [])
            self.assertEqual(len(other["after"]), 2)
            self.assertNotIn("blue lantern", str(other["model_users"]))
            self.assertEqual(Path(first["default_db"]), project / "data/checkpoints.sqlite")

    def test_chainlit_restores_context_in_a_new_process(self):
        project = Path(brain.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ui-test.sqlite"

            def ui_turn(message, resume_id=""):
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", "-B", "-c", UI_WORKER,
                     str(project), str(database), message, resume_id],
                    cwd=directory, capture_output=True, text=True,
                    encoding="utf-8", timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads(result.stdout.splitlines()[-1])

            first = ui_turn("Mi animal de prueba es el koala.")
            second = ui_turn("¿Cuál era mi animal de prueba?", first["session"]["thread_id"])
            self.assertEqual(set(first["session"]), {"thread_id"})
            self.assertEqual(second["session"], first["session"])
            self.assertEqual(second["final"], "koala")
            self.assertEqual(second["seen"], ["Mi animal de prueba es el koala.",
                                              "¿Cuál era mi animal de prueba?"])
            self.assertEqual(len(first["ids"]), 2)
            self.assertEqual(len(second["ids"]), 4)
            self.assertEqual(second["ids"][:2], first["ids"])
            self.assertEqual(len(set(second["ids"])), 4)

    async def test_tool_messages_survive_reopening(self):
        config = {"configurable": {"thread_id": "tool-test"}}
        responses = [
            AIMessage(content="", tool_calls=[{
                "id": "calculation1", "name": "calculate",
                "args": {"expression": "73 * (181.4 * (1 + 15 / 100))"},
            }]),
            AIMessage(content="15228.53"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "checkpoints.sqlite"
            async with brain.open_persistent_graph(database) as graph:
                with patch.object(brain, "model_with_tools") as model:
                    model.invoke.side_effect = responses
                    result = await graph.ainvoke(
                        {"messages": [("user", "Calculate the position.")]}, config=config
                    )
            async with brain.open_persistent_graph(database) as graph:
                state = await graph.aget_state(config)
            self.assertEqual(len(state.values["messages"]), 4)
            self.assertEqual(state.next, ())
            self.assertEqual([m.id for m in state.values["messages"]],
                             [m.id for m in result["messages"]])
            self.assertIsInstance(state.values["messages"][2], ToolMessage)
            self.assertAlmostEqual(float(state.values["messages"][2].content), 15228.53)

    def test_main_import_does_not_run(self):
        with patch("asyncio.run") as run:
            import main
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

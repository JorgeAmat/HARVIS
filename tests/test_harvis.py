import os
from pathlib import Path
import tempfile
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# Offline tests must not depend on a real API key or send model requests.
os.environ.setdefault("GROQ_API_KEY", "offline-test-key")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import graph as brain
import app


class CalculateTests(unittest.TestCase):
    def test_arithmetic(self):
        for expression, expected in [
            ("73 * (181.4 * 1.15)", 15228.53),
            ("73 * (181.4 * (1 + 15 / 100))", 15228.53),
            ("2 + 3 * 4", 14), ("-(2 + 3) / +2", -2.5),
            ("0", 0), (" 1.5 + .5 ", 2),
        ]:
            with self.subTest(expression=expression):
                self.assertAlmostEqual(brain.evaluate_expression(expression), expected)

    def test_rejects_unsafe_or_invalid_input(self):
        for expression in [
            "__import__('os').getcwd()", "abs(-1)", "x + 1", "(1).__class__",
            "[1][0]", "True", "'hello'", "2 ** 100", "4 // 2", "5 % 2",
            "1 / 0", "1e309", "1e100 * 2", "1 +", "", "1" * 501, "9" * 400,
            "+" * 101 + "1",
        ]:
            with self.subTest(expression=expression):
                self.assertTrue(brain.calculate.invoke({"expression": expression}).startswith("Error:"))


class GraphAndUITests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_loop_and_safe_ui(self):
        # The model is scripted; the graph, routing, ToolNode and calculation are real.
        tool_request = AIMessage(
            content="PRIVATE INTERMEDIATE TEXT",
            tool_calls=[{"id": "call1", "name": "calculate", "args": {
                "expression": "73 * (181.4 * (1 + 15 / 100))"
            }}], additional_kwargs={"reasoning_content": "PRIVATE REASONING"}
        )
        final = AIMessage(content="Result: 15228.53", additional_kwargs={
            "reasoning_content": "PRIVATE REASONING"
        })
        outputs = iter([tool_request, final])
        observations = []

        def model_response(messages):
            observations.append(messages)
            return next(outputs)

        steps, sent = [], []
        session = {"thread_id": str(uuid.uuid4())}
        active_thread = session["thread_id"]

        class Step:
            def __init__(self, **kwargs):
                self.name = kwargs["name"]
                self.input = self.output = ""
                steps.append(self)
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return False

        class Message:
            def __init__(self, content):
                self.content = content
            async def send(self):
                sent.append(self.content)

        fake_session = SimpleNamespace(get=session.get, set=session.__setitem__)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, "open_persistent_graph",
                          lambda: brain.open_persistent_graph(Path(directory) / "test.sqlite")), \
             patch.object(brain, "model_with_tools") as model, \
             patch.object(app.cl, "Step", Step), \
             patch.object(app.cl, "Message", Message), \
             patch.object(app.cl, "user_session", fake_session):
            model.invoke.side_effect = model_response
            await app.on_message(SimpleNamespace(content="Calculate the increased position."))

        self.assertEqual([step.name for step in steps], ["Agent", "Tool: calculate", "Agent"])
        self.assertEqual(steps[0].output, "Selected tool: calculate")
        self.assertIn("15 / 100", steps[1].input)
        self.assertAlmostEqual(float(steps[1].output), 15228.53)
        self.assertEqual(sent, ["Result: 15228.53"])
        self.assertIsInstance(observations[1][-1], ToolMessage)
        self.assertEqual(session, {"thread_id": active_thread})
        visible = str([(s.input, s.output) for s in steps]) + str(sent)
        self.assertNotIn("PRIVATE", visible)

    async def test_new_chats_are_isolated_and_resume_only_selects_state(self):
        session, sent = {}, []
        fake_session = SimpleNamespace(get=session.get, set=session.__setitem__)
        step = MagicMock()
        step.__aenter__.return_value = step

        def ui_message(content):
            sent.append(content)
            return SimpleNamespace(send=AsyncMock())

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ui.sqlite"

            async def saved(thread_id):
                async with brain.open_persistent_graph(database) as graph:
                    return await graph.aget_state({"configurable": {"thread_id": thread_id}})

            with patch.object(app, "open_persistent_graph", lambda: brain.open_persistent_graph(database)), \
                 patch.object(app.cl, "user_session", fake_session), \
                 patch.object(app.cl, "Message", side_effect=ui_message), \
                 patch.object(app.cl, "Step", return_value=step), \
                 patch.object(brain, "model_with_tools") as model, \
                 patch.object(brain.tool_node, "ainvoke") as tools:
                model.invoke.side_effect = lambda messages: AIMessage(content="OK")

                await app.on_chat_start()
                thread_a = session["thread_id"]
                self.assertEqual(str(uuid.UUID(thread_a)), thread_a)
                self.assertIn(thread_a, sent[-1])
                self.assertIsNone((await saved(thread_a)).created_at)
                model.invoke.assert_not_called()
                await app.on_message(SimpleNamespace(content="My test animal is koala."))

                await app.on_chat_start()
                thread_b = session["thread_id"]
                self.assertNotEqual(thread_a, thread_b)
                self.assertEqual(str(uuid.UUID(thread_b)), thread_b)
                self.assertIn(thread_b, sent[-1])
                await app.on_message(SimpleNamespace(content="My test animal is panda."))
                model_input = model.invoke.call_args.args[0]
                self.assertNotIn("koala", str([m.content for m in model_input]))
                state_a, state_b = await saved(thread_a), await saved(thread_b)
                self.assertEqual(len(state_a.values["messages"]), 2)
                self.assertEqual(len(state_b.values["messages"]), 2)

                # Successful resume reads state without invoking the model or tools.
                model.invoke.reset_mock()
                await app.on_message(SimpleNamespace(content=f"/resume {thread_a}"))
                self.assertEqual(session, {"thread_id": thread_a})
                self.assertIn(f"Resumed conversation: {thread_a}", sent[-1])
                model.invoke.assert_not_called()
                tools.assert_not_called()
                self.assertEqual([m.id for m in (await saved(thread_a)).values["messages"]],
                                 [m.id for m in state_a.values["messages"]])

                unknown = str(uuid.uuid4())
                await app.on_message(SimpleNamespace(content=f"/resume {unknown}"))
                self.assertEqual(session, {"thread_id": thread_a})
                self.assertIn("Conversation not found", sent[-1])
                self.assertIsNone((await saved(unknown)).created_at)
                for command in ("/resume", f"/resume {thread_b} extra"):
                    await app.on_message(SimpleNamespace(content=command))
                    self.assertEqual(session, {"thread_id": thread_a})
                    self.assertIn("Usage:", sent[-1])
                model.invoke.assert_not_called()
                tools.assert_not_called()

                await app.on_message(SimpleNamespace(content="Which animal did I mention?"))
                user_messages = [m.content for m in model.invoke.call_args.args[0]
                                 if isinstance(m, HumanMessage)]
                self.assertEqual(user_messages, ["My test animal is koala.",
                                                  "Which animal did I mention?"])
                restored = (await saved(thread_a)).values["messages"]
                self.assertEqual(len(restored), 4)
                self.assertEqual(len({m.id for m in restored}), 4)
                self.assertEqual([m.id for m in restored[:2]],
                                 [m.id for m in state_a.values["messages"]])
                self.assertEqual([m.id for m in (await saved(thread_b)).values["messages"]],
                                 [m.id for m in state_b.values["messages"]])

    async def test_plain_answer_without_tools(self):
        with patch.object(brain, "model_with_tools") as model:
            model.invoke.return_value = AIMessage(content="Hello")
            updates = [u async for u in brain.graph.astream(
                {"messages": [("user", "Hello")]}, stream_mode="updates"
            )]
        self.assertEqual([list(u)[0] for u in updates], ["agent"])


if __name__ == "__main__":
    unittest.main()

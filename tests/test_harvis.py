import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Offline tests must not depend on a real API key or send model requests.
os.environ.setdefault("GROQ_API_KEY", "offline-test-key")

from langchain_core.messages import AIMessage, ToolMessage
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
        session = {"messages": []}

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
        with patch.object(brain, "model_with_tools") as model, \
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
        self.assertEqual(len(session["messages"]), 4)
        visible = str([(s.input, s.output) for s in steps]) + str(sent)
        self.assertNotIn("PRIVATE", visible)

    async def test_plain_answer_without_tools(self):
        with patch.object(brain, "model_with_tools") as model:
            model.invoke.return_value = AIMessage(content="Hello")
            updates = [u async for u in brain.graph.astream(
                {"messages": [("user", "Hello")]}, stream_mode="updates"
            )]
        self.assertEqual([list(u)[0] for u in updates], ["agent"])


if __name__ == "__main__":
    unittest.main()

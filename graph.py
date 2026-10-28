import ast
import math
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()


# State is the information shared by graph nodes.
class AgentState(TypedDict):

    # add_messages merges new messages into history (matching IDs are updated).
    messages: Annotated[list, add_messages]


def evaluate_expression(expression: str) -> float:
    """Evaluate only bounded numeric arithmetic; never execute Python code."""
    if not expression.strip() or len(expression) > 500:
        raise ValueError("Use a non-empty expression of at most 500 characters.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid arithmetic expression.") from exc
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Expression is too complex (maximum 100 syntax nodes).")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if abs(node.value) > 1e100:
                raise ValueError("Numbers must be at most 1e100 in magnitude.")
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            result = value if isinstance(node.op, ast.UAdd) else -value
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero is not allowed.")
                result = left / right
        else:
            raise ValueError("Only numbers, parentheses, +, -, *, and / are allowed.")

        # Bound every intermediate result, not just the final answer.
        if not math.isfinite(result) or abs(result) > 1e100:
            raise ValueError("Numbers and results must be finite and at most 1e100 in magnitude.")
        return result

    return evaluate(tree.body)


@tool
def calculate(expression: str) -> str:
    """Calculate a full arithmetic expression using numbers, parentheses, +, -, *, /.

    Include ALL arithmetic, including percentage conversions: for a 15 percent
    increase use 73 * (181.4 * (1 + 15 / 100)). No names, calls, powers or %.
    Returns the numeric result as text, or an explicit Error message.
    """
    try:
        return str(evaluate_expression(expression))
    except ValueError as exc:
        return f"Error: {exc}"


tools = [
    calculate
]


model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# bind_tools describes available tools; it does not execute them.
model_with_tools = model.bind_tools(
    tools
)


SYSTEM_PROMPT = """
You are HARVIS, Jorge's personal AI assistant.

Your goal is to help Jorge think, organize information,
understand things and carry out useful tasks.

You can use the tools available to you when necessary.

CAPABILITY BOUNDARY:
Your current capabilities are text conversation, temporary context from messages
provided in this session, and the calculate tool. Describe these honestly when
asked what you can do. Never claim capabilities or tools that are not available.
You can discuss general topics using your existing knowledge and help draft text
or plans, but discussing or suggesting an action does not mean you can perform it.
You currently cannot search the web, access files, use persistent memory, send or
read email, manage calendars/agendas, set reminders, or use voice. Do not claim
these capabilities unless they are explicitly enabled and actually available.
Never claim an external action was performed without a successful tool result.

IMPORTANT TOOL RULE:
Whenever a request requires arithmetic, you MUST use the
calculate tool with the COMPLETE expression. Never perform any arithmetic
mentally, including intermediate steps or percentage conversions. For example,
for 73 units at 181.4 with a 15 percent increase, send
73 * (181.4 * (1 + 15 / 100)), not a mentally simplified expression.
Use the tool result to answer. If it returns an error, correct the expression
or explain the error; never invent a result.

Be concise and clear.

Never pretend to remember information that has not actually
been provided to you or retrieved through a tool.
"""


def agent_node(state: AgentState):

    print("\n[GRAPH] Entrando en AGENT")

    response = model_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"]
        ]
    )
    if response.tool_calls:
        print("\n[AGENT] Ha solicitado tools:")

        for tool_call in response.tool_calls:
            print(
                f"  -> {tool_call['name']} "
                f"{tool_call['args']}"
            )
    else:
        print("\n[AGENT] No ha solicitado ninguna tool")

    return {
        "messages": [response]
    }


# ToolNode executes requested tools and adds their results as ToolMessages.
tool_node = ToolNode(
    tools
)


builder = StateGraph(
    AgentState
)


builder.add_node(
    "agent",
    agent_node
)

builder.add_node(
    "tools",
    tool_node
)


builder.add_edge(
    START,
    "agent"
)


# tools_condition routes tool requests to tools; otherwise the graph ends.
builder.add_conditional_edges(
    "agent",
    tools_condition
)


# Return tool results to the agent: agent -> tools -> agent.
builder.add_edge(
    "tools",
    "agent"
)


# Chainlit remains session-only for v0.2A.
graph = builder.compile()

CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "checkpoints.sqlite"


@asynccontextmanager
async def open_persistent_graph(database_path: Path = CHECKPOINT_PATH):
    """Keep the SQLite connection open for all graph operations in this context."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(database_path)) as checkpointer:
        # Restrict checkpoint decoding to the serializer's built-in safe types.
        checkpointer.serde = JsonPlusSerializer(allowed_msgpack_modules=None)
        yield builder.compile(checkpointer=checkpointer)

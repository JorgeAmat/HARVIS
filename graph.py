# ============================================================
# IMPORTS
# ============================================================

import ast
import math

from typing import Annotated, TypedDict

from dotenv import load_dotenv

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# CONFIG
# ============================================================

# Carga las variables de nuestro archivo .env
load_dotenv()


# ============================================================
# 1. STATE
# ============================================================

class AgentState(TypedDict):

    # El state de momento solo contiene la conversación.
    #
    # add_messages significa:
    #
    # NO sustituyas los mensajes anteriores.
    # Añade los nuevos al historial.
    #
    # USER -> AI -> TOOL -> AI -> ...
    messages: Annotated[list, add_messages]


# ============================================================
# 2. TOOLS
# ============================================================

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

        # Check every intermediate value, not just the final answer.
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


# Esta será la lista de capacidades disponibles para HARVIS.
#
# De momento solo tiene UNA.
tools = [
    calculate
]


# ============================================================
# 3. MODEL
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# bind_tools NO ejecuta herramientas.
#
# Simplemente informa al LLM:
#
# "Existe calculate y puedes solicitar que se ejecute."
model_with_tools = model.bind_tools(
    tools
)


# ============================================================
# 4. PERSONALITY / SYSTEM PROMPT
# ============================================================


SYSTEM_PROMPT = """
You are HARVIS, Jorge's personal AI assistant.

Your goal is to help Jorge think, organize information,
understand things and carry out useful tasks.

You can use the tools available to you when necessary.

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


# ============================================================
# 5. AGENT NODE
# ============================================================

def agent_node(state: AgentState):

    print("\n[GRAPH] Entrando en AGENT")

    # Al modelo le pasamos:
    #
    # 1. Sus instrucciones generales
    # 2. Todo el historial de mensajes
    #
    # Ejemplo:
    #
    # SYSTEM: Eres HARVIS...
    # USER: ¿Cuánto es 23 * 8?
    # AI: quiero usar calculate...
    # TOOL: 184
    #
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

    # add_messages añadirá esta respuesta al historial
    return {
        "messages": [response]
    }


# ============================================================
# 6. TOOL NODE
# ============================================================

# Este nodo sabe ejecutar cualquiera de las tools
# que hay dentro de "tools".
tool_node = ToolNode(
    tools
)


# ============================================================
# 7. GRAPH
# ============================================================

builder = StateGraph(
    AgentState
)


# Registramos nuestros nodos
builder.add_node(
    "agent",
    agent_node
)

builder.add_node(
    "tools",
    tool_node
)


# Siempre empezamos preguntando al agente.
builder.add_edge(
    START,
    "agent"
)


# Después del agente:
#
# Si el LLM pidió una tool:
#
#       agent -> tools
#
# Si NO pidió una tool:
#
#       agent -> END
#
builder.add_conditional_edges(
    "agent",
    tools_condition
)


# Una vez ejecutada una herramienta:
#
# volvemos SIEMPRE al agente.
#
# Esto crea el agent loop.
builder.add_edge(
    "tools",
    "agent"
)


# ============================================================
# 8. COMPILE
# ============================================================

graph = builder.compile()
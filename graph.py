# ============================================================
# IMPORTS
# ============================================================

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

@tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    Performs a mathematical operation between two numbers.

    operation must be:
    add, subtract, multiply or divide.
    """

    print(
        f"[TOOL calculator] "
        f"a={a}, b={b}, operation={operation}"
    )

    if operation == "add":
        return a + b

    if operation == "subtract":
        return a - b

    if operation == "multiply":
        return a * b

    if operation == "divide":
        return a / b

    raise ValueError(
        f"Unknown operation: {operation}"
    )


# Esta será la lista de capacidades disponibles para HARVIS.
#
# De momento solo tiene UNA.
tools = [
    calculator
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
# "Existe calculator y puedes solicitar que se ejecute."
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
    # AI: quiero usar calculator...
    # TOOL: 184
    #
    response = model_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"]
        ]
    )

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
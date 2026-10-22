# ============================================================
# IMPORTS
# ============================================================

import chainlit as cl

from langchain_core.messages import HumanMessage

# Importamos EL MISMO cerebro
# que acabamos de probar en main.py.
from graph import graph


# ============================================================
# CUANDO SE ABRE UN CHAT
# ============================================================

@cl.on_chat_start
async def on_chat_start():

    # Creamos una lista vacía para guardar
    # temporalmente la conversación.
    #
    # IMPORTANTE:
    # esto todavía NO es memoria permanente.
    cl.user_session.set(
        "messages",
        []
    )

    await cl.Message(
        content=(
            "HARVIS v0.1 online.\n\n"
            "¿En qué te ayudo?"
        )
    ).send()


# ============================================================
# CUANDO JORGE ENVÍA UN MENSAJE
# ============================================================

@cl.on_message
async def on_message(message: cl.Message):

    # Recuperamos la conversación existente.
    history = cl.user_session.get(
        "messages"
    )

    # Convertimos el texto escrito en la interfaz
    # en un mensaje de LangChain.
    user_message = HumanMessage(
        content=message.content
    )

    # Este será el state que entra a LangGraph.
    input_state = {
        "messages": history + [user_message]
    }


    # --------------------------------------------------------
    # STEP VISIBLE
    # --------------------------------------------------------
    #
    # Esto aparecerá en la interfaz como un paso expandible.
    #
    # Más adelante tendremos:
    #
    # Memory Search
    # Gmail
    # Calendar
    # Tool Execution
    # etc.
    #
    async with cl.Step(
        name="HARVIS Graph"
    ) as step:

        step.input = message.content

        # ainvoke es invoke pero ASÍNCRONO.
        #
        # Como Chainlit funciona async,
        # utilizamos la versión async de LangGraph.
        result = await graph.ainvoke(
            input_state
        )

        step.output = "Graph execution completed"


    # Recuperamos el último mensaje.
    #
    # Normalmente será la respuesta final del AI.
    final_message = result["messages"][-1]


    # Guardamos el nuevo historial
    # en la sesión de Chainlit.
    cl.user_session.set(
        "messages",
        result["messages"]
    )


    # Y mostramos la respuesta.
    await cl.Message(
        content=final_message.content
    ).send()

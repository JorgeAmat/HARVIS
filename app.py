import json

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from graph import graph


@cl.on_chat_start
async def on_chat_start():
    # Temporary session context only; nothing is saved to disk.
    cl.user_session.set("messages", [])
    await cl.Message(content="HARVIS v0.1 online.\n\n¿En qué te ayudo?").send()


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("messages") or []
    messages = add_messages(history, [HumanMessage(content=message.content)])
    tool_calls = {}
    final_message = None

    # Updates contain each completed node's messages, not private model reasoning.
    # Do not attach the generic model callback: display only selected fields below.
    async for update in graph.astream({"messages": messages}, stream_mode="updates"):
        for node_name, state_update in update.items():
            new_messages = state_update.get("messages", [])
            messages = add_messages(messages, new_messages)
            for item in new_messages:
                if node_name == "agent" and isinstance(item, AIMessage):
                    async with cl.Step(name="Agent", type="run") as step:
                        if item.tool_calls:
                            tool_calls.update({call["id"]: call for call in item.tool_calls})
                            step.output = "Selected tool: " + ", ".join(
                                call["name"] for call in item.tool_calls
                            )
                            # Never display AI content accompanying a tool request.
                        else:
                            final_message = item
                            step.output = "Final answer"
                elif node_name == "tools" and isinstance(item, ToolMessage):
                    call = tool_calls[item.tool_call_id]
                    async with cl.Step(
                        name=f"Tool: {call['name']}", type="tool", show_input="json"
                    ) as step:
                        step.input = json.dumps(call["args"], ensure_ascii=False)
                        step.output = item.content
                        step.is_error = (
                            item.status == "error" or str(item.content).startswith("Error:")
                        )

    if final_message is None:
        raise RuntimeError("Graph finished without a final assistant response.")
    cl.user_session.set("messages", messages)
    # Only final answer content is sent; additional_kwargs/reasoning are not shown.
    await cl.Message(content=final_message.content).send()

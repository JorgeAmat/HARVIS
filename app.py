import json

import chainlit as cl
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import open_persistent_graph


@cl.on_chat_start
async def on_chat_start():
    # One stable local conversation; even a new UI chat reuses this thread.
    cl.user_session.set("thread_id", "chainlit-main")
    await cl.Message(content="HARVIS v0.2B online.\n\nContinuamos la conversación guardada.").send()


@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
    tool_calls = {}
    final_message = None

    # SQLite restores history; the UI submits only this turn's new message.
    async with open_persistent_graph() as graph:
        async for update in graph.astream(
            {"messages": [HumanMessage(content=message.content)]},
            config=config, stream_mode="updates",
        ):
            for node_name, state_update in update.items():
                for item in state_update.get("messages", []):
                    if node_name == "agent" and isinstance(item, AIMessage):
                        async with cl.Step(name="Agent", type="run") as step:
                            if item.tool_calls:
                                tool_calls.update({call["id"]: call for call in item.tool_calls})
                                step.output = "Selected tool: " + ", ".join(
                                    call["name"] for call in item.tool_calls
                                )
                                # Do not display AI content accompanying tool requests.
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
    # Separate reasoning metadata is not displayed.
    await cl.Message(content=final_message.content).send()

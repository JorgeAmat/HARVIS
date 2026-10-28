from io import BytesIO
import json
import uuid
import wave

import chainlit as cl
from groq import AsyncGroq, GroqError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import open_persistent_graph


SAMPLE_RATE = 24000
MAX_RECORDING_BYTES = SAMPLE_RATE * 2 * 60  # One minute of mono PCM16.


@cl.on_chat_start
async def on_chat_start():
    thread_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    await cl.Message(content=f"HARVIS v0.3A.1 online.\n\nConversation ID: {thread_id}").send()


@cl.on_message
async def on_message(message: cl.Message):
    command = message.content.strip().split()
    if command and command[0] == "/resume":
        if len(command) != 2:
            await cl.Message(content="Usage: /resume <thread_id>").send()
            return
        thread_id = command[1]
        async with open_persistent_graph() as graph:
            state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if state.created_at is None:
            await cl.Message(content=f"Conversation not found: {thread_id}. Active conversation unchanged.").send()
            return
        cl.user_session.set("thread_id", thread_id)
        await cl.Message(content=f"Resumed conversation: {thread_id}").send()
        return

    await run_agent(message.content, cl.user_session.get("thread_id"))


async def run_agent(text: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    tool_calls = {}
    final_message = None

    # SQLite restores history; the UI submits only this turn's new message.
    async with open_persistent_graph() as graph:
        async for update in graph.astream(
            {"messages": [HumanMessage(content=text)]},
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


def pcm_to_wav(pcm: bytes | bytearray) -> bytes:
    """Add a WAV header to Chainlit's mono, 24 kHz, little-endian PCM16 samples."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm)
    return buffer.getvalue()


@cl.on_audio_start
async def on_audio_start():
    recording_id = str(uuid.uuid4())
    controls = cl.Message(content="", elements=[cl.CustomElement(
        name="RecordingControls", props={"recording_id": recording_id},
    )])
    cl.user_session.set("audio_recording", {
        "pcm": bytearray(), "thread_id": cl.user_session.get("thread_id"),
        "recording_id": recording_id, "approved_for_send": False,
        "cancelled": False, "error": None, "controls": controls,
    })
    await controls.send()
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    # No awaits here: only validate and append to this recording's RAM buffer.
    recording = cl.user_session.get("audio_recording")
    if recording is None or recording["cancelled"] or recording["error"]:
        return
    if chunk.mimeType != "pcm16":
        recording["error"] = "Unsupported audio format. Expected PCM16."
    elif len(chunk.data) % 2:
        recording["error"] = "Invalid PCM16 audio: incomplete 16-bit sample."
    elif len(recording["pcm"]) + len(chunk.data) > MAX_RECORDING_BYTES:
        recording["error"] = "Recording too large. Please record no more than 60 seconds."
    else:
        recording["pcm"].extend(chunk.data)
    if recording["error"]:
        recording["pcm"].clear()


@cl.action_callback("recording_decision")
async def recording_decision(action: cl.Action):
    recording = cl.user_session.get("audio_recording")
    decision = action.payload.get("decision")
    if (recording is None
            or action.payload.get("recording_id") != recording["recording_id"]
            or decision not in ("cancel", "send")
            or recording["cancelled"] or recording["approved_for_send"]):
        return {"accepted": False}
    # No awaits: the first valid decision is recorded before acknowledgement.
    if decision == "cancel":
        recording["cancelled"] = True
        recording["pcm"].clear()
    else:
        recording["approved_for_send"] = True
    return {"accepted": True}


@cl.on_audio_end
async def on_audio_end():
    recording = cl.user_session.get("audio_recording")
    cl.user_session.set("audio_recording", None)  # Detach before any awaits.
    if recording is None:
        return
    await recording["controls"].remove()
    # Stopping (including native X/P) is not permission to upload audio.
    if not recording["approved_for_send"]:
        recording["pcm"].clear()
        return
    if recording["error"]:
        await cl.Message(content=recording["error"]).send()
        return
    if not recording["pcm"]:
        await cl.Message(content="No audio recorded. Please try again.").send()
        return

    wav_bytes = pcm_to_wav(recording["pcm"])
    recording["pcm"].clear()
    try:
        async with AsyncGroq(max_retries=0) as client:
            transcription = await client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=("recording.wav", wav_bytes, "audio/wav"),
                response_format="json",
            )
        text = transcription.text.strip()
    except GroqError:
        await cl.Message(content="Groq transcription failed. Please try again or type your message.").send()
        return
    finally:
        wav_bytes = b""  # Audio is never written to a file or graph state.

    if not text:
        await cl.Message(content="No speech was transcribed. Please try again.").send()
        return
    await cl.Message(content=text, author="You (transcribed)", type="user_message").send()
    # Spoken text bypasses UI commands and stays in the thread captured at start.
    await run_agent(text, recording["thread_id"])

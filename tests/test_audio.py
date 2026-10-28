from io import BytesIO
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import wave

os.environ.setdefault("GROQ_API_KEY", "offline-test-key")

from groq import GroqError
from langchain_core.messages import AIMessage, HumanMessage
import app
import graph as brain


class AudioTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = {"thread_id": "recording-thread"}
        self.sent = []
        self.messages = []
        self.real_run_agent = app.run_agent
        self.agent = AsyncMock()
        self.transcribe = AsyncMock(return_value=SimpleNamespace(text="  Hola HARVIS.  "))
        client = SimpleNamespace(audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=self.transcribe)))
        context = MagicMock()
        context.__aenter__.return_value = client
        self.client = MagicMock(return_value=context)
        step = MagicMock()
        step.__aenter__.return_value = step

        def message(**kwargs):
            self.sent.append(kwargs)
            item = SimpleNamespace(send=AsyncMock(), remove=AsyncMock())
            self.messages.append(item)
            return item

        for target, replacement in [
            ("app.cl.user_session", SimpleNamespace(get=self.session.get, set=self.session.__setitem__)),
            ("app.cl.Message", message), ("app.cl.Step", MagicMock(return_value=step)),
            ("app.AsyncGroq", self.client), ("app.run_agent", self.agent),
            ("app.cl.CustomElement", MagicMock()),
        ]:
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def record(self, pcm=b"\x00\x00\xff\x7f", mime_type="pcm16", approve=True):
        self.assertTrue(await app.on_audio_start())
        await app.on_audio_chunk(SimpleNamespace(mimeType=mime_type, data=pcm))
        if approve:
            self.assertEqual(await self.decide("send"), {"accepted": True})

    async def decide(self, decision, recording_id=None):
        if recording_id is None:
            recording_id = self.session["audio_recording"]["recording_id"]
        return await app.recording_decision(SimpleNamespace(payload={
            "recording_id": recording_id, "decision": decision,
        }))

    def test_pcm_to_valid_wav(self):
        pcm = b"\x00\x00\xff\x7f"
        wav = app.pcm_to_wav(pcm)
        self.assertEqual(wav[:4], b"RIFF")
        with wave.open(BytesIO(wav), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertEqual(audio.getframerate(), 24000)
            self.assertEqual(audio.getnframes(), 2)
            self.assertEqual(audio.readframes(2), pcm)

    async def test_transcript_uses_captured_thread_exactly_once(self):
        await self.record()
        recording = self.session["audio_recording"]
        self.session["thread_id"] = "different-active-thread"

        async def transcribe(**kwargs):
            self.assertIsNone(self.session["audio_recording"])
            self.assertEqual(recording["pcm"], bytearray())
            self.assertEqual(kwargs["model"], "whisper-large-v3-turbo")
            self.assertNotIn("language", kwargs)
            filename, data, mime_type = kwargs["file"]
            self.assertEqual((filename, mime_type), ("recording.wav", "audio/wav"))
            with wave.open(BytesIO(data), "rb") as audio:
                self.assertEqual(audio.readframes(2), b"\x00\x00\xff\x7f")
            return SimpleNamespace(text="  Hola HARVIS.  ")

        self.transcribe.side_effect = transcribe
        await app.on_audio_end()
        await app.on_audio_end()  # A duplicate event must not submit another turn.
        self.client.assert_called_once_with(max_retries=0)
        self.transcribe.assert_awaited_once()
        self.agent.assert_awaited_once_with("Hola HARVIS.", "recording-thread")
        self.assertEqual([m for m in self.sent if m.get("type") == "user_message"], [{"content": "Hola HARVIS.",
                                     "author": "You (transcribed)", "type": "user_message"}])
        self.assertEqual(self.session["thread_id"], "different-active-thread")

    async def test_spoken_resume_is_agent_text(self):
        self.transcribe.return_value = SimpleNamespace(text="/resume another-thread")
        await self.record()
        await app.on_audio_end()
        self.agent.assert_awaited_once_with("/resume another-thread", "recording-thread")
        self.assertEqual(self.session["thread_id"], "recording-thread")

    async def test_typed_input_uses_shared_agent_function(self):
        await app.on_message(SimpleNamespace(content="Typed input"))
        self.agent.assert_awaited_once_with("Typed input", "recording-thread")
        self.client.assert_not_called()

    async def test_empty_recording_does_not_call_groq_or_agent(self):
        await app.on_audio_start()
        await self.decide("send")
        await app.on_audio_end()
        self.assertIn("No audio recorded", self.sent[-1]["content"])
        self.assertIsNone(self.session["audio_recording"])
        self.client.assert_not_called()
        self.agent.assert_not_awaited()

    async def test_unsupported_format_does_not_call_agent(self):
        await self.record(mime_type="audio/webm")
        await app.on_audio_end()
        self.assertIn("Unsupported audio format", self.sent[-1]["content"])
        self.client.assert_not_called()
        self.agent.assert_not_awaited()

    async def test_incomplete_pcm_sample_does_not_call_agent(self):
        await self.record(pcm=b"\x00")
        await app.on_audio_end()
        self.assertIn("incomplete 16-bit sample", self.sent[-1]["content"])
        self.client.assert_not_called()
        self.agent.assert_not_awaited()

    async def test_oversized_recording_is_discarded(self):
        with patch.object(app, "MAX_RECORDING_BYTES", 4):
            await self.record()
            await app.on_audio_chunk(SimpleNamespace(mimeType="pcm16", data=b"\x00\x00"))
            self.assertEqual(self.session["audio_recording"]["pcm"], bytearray())
            await app.on_audio_chunk(SimpleNamespace(mimeType="pcm16", data=b"\x00\x00"))
            await app.on_audio_end()
        self.assertIn("Recording too large", self.sent[-1]["content"])
        self.client.assert_not_called()
        self.agent.assert_not_awaited()

    async def test_empty_transcription_does_not_call_agent(self):
        self.transcribe.return_value = SimpleNamespace(text="   ")
        await self.record()
        await app.on_audio_end()
        self.assertIn("No speech was transcribed", self.sent[-1]["content"])
        self.transcribe.assert_awaited_once()
        self.agent.assert_not_awaited()
        self.assertIsNone(self.session["audio_recording"])

    async def test_transcription_failure_does_not_call_agent(self):
        self.transcribe.side_effect = GroqError("Synthetic provider failure")
        await self.record()
        await app.on_audio_end()
        self.assertIn("Groq transcription failed", self.sent[-1]["content"])
        self.assertNotIn("Synthetic provider failure", self.sent[-1]["content"])
        self.agent.assert_not_awaited()
        self.assertIsNone(self.session["audio_recording"])

    async def test_spoken_then_typed_messages_persist_in_one_thread(self):
        self.transcribe.return_value = SimpleNamespace(text="Mi animal de prueba es el koala.")
        self.agent.side_effect = self.real_run_agent
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "audio-test.sqlite"
            with patch.object(app, "open_persistent_graph", lambda: brain.open_persistent_graph(database)), \
                 patch.object(brain, "model_with_tools") as model:
                model.invoke.side_effect = [AIMessage(content="Entendido."), AIMessage(content="koala")]
                await self.record()
                await app.on_audio_end()
                await app.on_message(SimpleNamespace(content="¿Cuál era mi animal de prueba?"))
                received = [m.content for m in model.invoke.call_args.args[0] if isinstance(m, HumanMessage)]
                self.assertEqual(received, ["Mi animal de prueba es el koala.",
                                            "¿Cuál era mi animal de prueba?"])
            async with brain.open_persistent_graph(database) as graph:
                state = await graph.aget_state({"configurable": {"thread_id": "recording-thread"}})
            self.assertEqual(len(state.values["messages"]), 4)
            self.assertEqual(len({m.id for m in state.values["messages"]}), 4)
            self.assertEqual(set(state.values), {"messages"})
            self.assertEqual(self.agent.await_count, 2)

    async def test_cancel_clears_immediately_and_never_uploads_or_runs_agent(self):
        await self.record(approve=False)
        recording = self.session["audio_recording"]
        panel = self.messages[0]
        self.assertEqual(await self.decide("cancel"), {"accepted": True})
        self.assertTrue(recording["cancelled"])
        self.assertFalse(recording["approved_for_send"])
        self.assertEqual(recording["pcm"], bytearray())
        await app.on_audio_chunk(SimpleNamespace(mimeType="pcm16", data=b"\x00\x00"))
        self.assertEqual(recording["pcm"], bytearray())
        await app.on_audio_end()
        await app.on_audio_end()
        panel.remove.assert_awaited_once()
        self.assertIsNone(self.session["audio_recording"])
        self.client.assert_not_called()
        self.transcribe.assert_not_awaited()
        self.agent.assert_not_awaited()

    async def test_native_stop_without_approval_discards(self):
        await self.record(approve=False)
        recording = self.session["audio_recording"]
        self.assertFalse(recording["approved_for_send"])
        await app.on_audio_end()
        self.assertEqual(recording["pcm"], bytearray())
        self.messages[0].remove.assert_awaited_once()
        self.assertEqual(len(self.sent), 1)  # Only the now-removed control panel.
        self.client.assert_not_called()
        self.agent.assert_not_awaited()

    async def test_finish_approval_precedes_upload_and_panel_is_removed(self):
        await self.record(approve=False)
        panel = self.messages[0]
        self.assertEqual(await self.decide("send"), {"accepted": True})
        self.assertTrue(self.session["audio_recording"]["approved_for_send"])
        self.transcribe.assert_not_awaited()
        self.agent.assert_not_awaited()

        async def transcribe(**kwargs):
            panel.remove.assert_awaited_once()
            return SimpleNamespace(text="Hello")

        self.transcribe.side_effect = transcribe
        await app.on_audio_end()
        await app.on_audio_end()
        self.transcribe.assert_awaited_once()
        self.agent.assert_awaited_once_with("Hello", "recording-thread")

    async def test_stale_or_invalid_decisions_cannot_affect_a_new_recording(self):
        await self.record(approve=False)
        old_id = self.session["audio_recording"]["recording_id"]
        await app.on_audio_end()
        await self.record(approve=False)
        current = self.session["audio_recording"]
        self.assertNotEqual(old_id, current["recording_id"])
        self.assertEqual(app.cl.CustomElement.call_args.kwargs["props"],
                         {"recording_id": current["recording_id"]})
        original_pcm = bytes(current["pcm"])
        for decision in ("cancel", "send"):
            self.assertEqual(await self.decide(decision, old_id), {"accepted": False})
        self.assertEqual(await self.decide("invalid"), {"accepted": False})
        self.assertFalse(current["approved_for_send"])
        self.assertFalse(current["cancelled"])
        self.assertEqual(bytes(current["pcm"]), original_pcm)
        self.client.assert_not_called()
        self.agent.assert_not_awaited()

    async def test_first_decision_wins_and_detached_recording_rejects_actions(self):
        await self.record(approve=False)
        recording_id = self.session["audio_recording"]["recording_id"]
        self.assertEqual(await self.decide("cancel"), {"accepted": True})
        self.assertEqual(await self.decide("send"), {"accepted": False})
        await app.on_audio_end()
        self.assertEqual(await self.decide("send", recording_id), {"accepted": False})
        self.agent.assert_not_awaited()
        self.transcribe.assert_not_awaited()

    def test_audio_config_matches_pcm_and_uploads_stay_disabled(self):
        config_path = Path(app.__file__).resolve().parent / ".chainlit/config.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        self.assertTrue(config["features"]["audio"]["enabled"])
        self.assertEqual(config["features"]["audio"]["sample_rate"], app.SAMPLE_RATE)
        self.assertFalse(config["features"]["spontaneous_file_upload"]["enabled"])


if __name__ == "__main__":
    unittest.main()

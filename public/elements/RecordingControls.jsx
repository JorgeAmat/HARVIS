import { useEffect, useRef, useState } from "react";
import { useAudio } from "@chainlit/react-client";
import { Button } from "@/components/ui/button";

export default function RecordingControls() {
  const { audioConnection, endConversation } = useAudio();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const busy = useRef(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  async function decide(decision) {
    if (busy.current || audioConnection !== "on") return;
    busy.current = true;
    setPending(true);
    setError("");
    try {
      const result = await callAction({
        name: "recording_decision",
        payload: { recording_id: props.recording_id, decision },
      });
      if (!mounted.current) return;
      if (!result?.success || result.response?.accepted !== true) {
        setError("This recording has ended or already has a decision.");
        return;
      }
      // Python has accepted the decision before audio_end can be emitted.
      await endConversation();
    } catch {
      if (mounted.current) setError("Unable to apply the recording action.");
    } finally {
      busy.current = false;
      if (mounted.current) setPending(false);
    }
  }

  const disabled = pending || audioConnection !== "on";
  return (
    <div className="rounded-md border p-3 space-y-2">
      <p>Recording...</p>
      <div className="flex gap-2">
        <Button type="button" variant="outline" disabled={disabled} onClick={() => decide("cancel")}>
          X Cancel
        </Button>
        <Button type="button" disabled={disabled} onClick={() => decide("send")}>
          ✓ Finish / Send
        </Button>
      </div>
      {error && <p role="alert">{error}</p>}
    </div>
  );
}

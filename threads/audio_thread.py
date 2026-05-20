"""Raw-PCM microphone capture thread for the realtime protocol.

Replaces `threads/fft_thread.py`'s role. Captures from the mic with
`sounddevice` and emits raw PCM s16le mono in ~50 ms slices via the
`pcm_chunk_ready(bytes)` Qt signal. The server runs the FFT/mel
pipeline; the client just ships bytes.

The legacy `FFTAnalyzerThread` is still importable from
`threads/fft_thread.py` (deprecated, kept for backwards-compatible
fallback under the `GLITCHBOX_REALTIME_V2=0` feature flag).
"""

from __future__ import annotations

import queue
from typing import Optional

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal


class AudioThread(QThread):
    """Capture mic PCM and emit ~50 ms s16le mono chunks."""

    pcm_chunk_ready = Signal(bytes)
    error_occurred = Signal(str)

    def __init__(
        self,
        input_device_index: Optional[int] = None,
        sample_rate: int = 44100,
        chunk_ms: int = 50,
    ):
        super().__init__()
        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        # Frames per emitted chunk; sounddevice's `blocksize` honors this
        # so we don't need a separate ring buffer for slicing.
        self.frames_per_chunk = int(sample_rate * chunk_ms / 1000)
        self.running = False
        self._q: "queue.Queue[bytes]" = queue.Queue(maxsize=64)
        self._stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    # sounddevice callback (runs on the PortAudio thread)
    # ------------------------------------------------------------------

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            # XRuns etc. — non-fatal; just log.
            print(f"[AudioThread] sounddevice status: {status}")
        # indata is float32 in [-1, 1] when dtype=float32, OR int16 already
        # when dtype='int16'. We requested int16 below.
        if indata.dtype != np.int16:
            # Defensive: clamp and convert if the stream surprises us.
            clipped = np.clip(indata, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype(np.int16)
        else:
            pcm = indata
        # If multi-channel sneaks in, downmix to mono.
        if pcm.ndim == 2 and pcm.shape[1] > 1:
            pcm = pcm.mean(axis=1).astype(np.int16)
        try:
            self._q.put_nowait(pcm.tobytes())
        except queue.Full:
            # Backpressure: drop the oldest chunk to keep latency bounded.
            try:
                self._q.get_nowait()
                self._q.put_nowait(pcm.tobytes())
            except queue.Empty:
                pass

    # ------------------------------------------------------------------
    # Thread main
    # ------------------------------------------------------------------

    def _open_stream(self):
        """Open an input stream — validate + enumerate like the original
        glitchbox frontend's _test_device.

        Tries the configured device first (None = system default), then
        every device with input channels, each at 1 then 2 channels (the
        callback downmixes multi-channel to mono). ``sd.check_input_settings``
        validates before opening so a bad combo is rejected cleanly instead
        of the opaque PortAudio -9998. All prints flush so the result is
        actually visible in the (block-buffered) client log.
        """
        # Candidate devices: configured first, then all real inputs.
        candidate_devs = [self.input_device_index]
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0 and i not in candidate_devs:
                    candidate_devs.append(i)
        except Exception as e:
            print(f"[AudioThread] device enumeration failed: {e}", flush=True)

        for dev in candidate_devs:
            for ch in (1, 2):
                try:
                    sd.check_input_settings(
                        device=dev, channels=ch, dtype="int16",
                        samplerate=self.sample_rate,
                    )
                    stream = sd.InputStream(
                        device=dev,
                        channels=ch,
                        samplerate=self.sample_rate,
                        dtype="int16",
                        blocksize=self.frames_per_chunk,
                        callback=self._audio_callback,
                    )
                    stream.start()
                    print(
                        f"[AudioThread] Capturing device={dev} channels={ch} "
                        f"@ {self.sample_rate} Hz, {self.chunk_ms} ms chunks",
                        flush=True,
                    )
                    return stream
                except Exception as e:
                    print(
                        f"[AudioThread] device={dev} channels={ch} failed: {e}",
                        flush=True,
                    )
        raise RuntimeError("no working input device found")

    def run(self) -> None:
        self.running = True
        print(
            f"[AudioThread] run() entered (configured device="
            f"{self.input_device_index})",
            flush=True,
        )
        try:
            self._stream = self._open_stream()
            while self.running:
                try:
                    chunk = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.pcm_chunk_ready.emit(chunk)
        except Exception as e:
            err = f"AudioThread error: {e}"
            print(f"[AudioThread] {err}")
            self.error_occurred.emit(err)
        finally:
            self._cleanup_stream()

    def _cleanup_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                print(f"[AudioThread] Stream cleanup error: {e}")
            self._stream = None

    def stop(self) -> None:
        print("[AudioThread] Stopping")
        self.running = False
        if not self.wait(2000):
            print("[AudioThread] Thread did not finish in time; terminating")
            self.terminate()
            self.wait(500)

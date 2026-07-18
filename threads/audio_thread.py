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
from scipy.signal import resample_poly


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

        ``sd.check_input_settings`` validates before opening so a bad combo
        is rejected cleanly instead of the opaque PortAudio -9998. All
        prints flush so the result is actually visible in the
        (block-buffered) client log.

        Two modes:

        * Explicit device (``self.input_device_index is not None``, i.e.
          the user picked one via the mic-index UI): HONOR IT. Try it at
          ``self.sample_rate`` first, then fall back to the device's own
          native rate with in-process resampling (see
          ``_open_at_native_rate``) — never silently substitute different
          hardware. A raw ALSA ``hw:X,Y`` device (as opposed to the
          ``pulse``/``default``/``sysdefault`` software-mixed ones) only
          accepts its one native rate and rejects everything else with
          "Invalid sample rate"; previously that rejection fell through to
          scanning *every other input device* until something happened to
          work, which is how a mic-index switch to a USB mic could end up
          silently capturing from the laptop's built-in mic instead — the
          UI would say "updated" while actually recording the wrong source.
        * Auto-detect (``self.input_device_index is None``, e.g. at
          startup before the user has chosen anything): keep the original
          "try the default, then scan every input device" behavior — there
          is no explicit user intent to honor yet, so grabbing whatever
          works is the right default.
        """
        if self.input_device_index is not None:
            dev = self.input_device_index
            stream = self._open_at_rate(dev, self.sample_rate, self._audio_callback)
            if stream is not None:
                return stream
            stream = self._open_at_native_rate(dev)
            if stream is not None:
                return stream
            raise RuntimeError(
                f"input device {dev} could not be opened at {self.sample_rate} Hz "
                f"or its native rate — see [AudioThread] lines above for the "
                f"per-attempt errors"
            )

        # Auto-detect: configured device is None, so try PortAudio's default
        # first, then every other real input device, each at 1 then 2
        # channels — same as before this fix.
        candidate_devs = [None]
        try:
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_input_channels", 0) > 0:
                    candidate_devs.append(i)
        except Exception as e:
            print(f"[AudioThread] device enumeration failed: {e}", flush=True)

        for dev in candidate_devs:
            stream = self._open_at_rate(dev, self.sample_rate, self._audio_callback)
            if stream is not None:
                return stream
        raise RuntimeError("no working input device found")

    def _open_at_rate(self, dev, rate: int, callback, latency=None):
        """Try opening ``dev`` at ``rate`` Hz, 1 then 2 channels. Returns the
        started stream, or None if every attempt failed (each failure is
        printed). ``latency`` is passed straight to ``sd.InputStream`` —
        pass ``'high'`` for a resampling callback (see
        ``_make_resampling_callback``), which does enough per-block CPU work
        (polyphase resample) that it can occasionally miss PortAudio's
        default (low-latency) buffer deadline under GIL contention from the
        rest of the app (camera JPEG encode, network I/O, Qt event loop),
        surfacing as "input overflow" — a bigger buffer absorbs that jitter
        at the cost of a bit more end-to-end audio latency."""
        for ch in (1, 2):
            try:
                sd.check_input_settings(
                    device=dev, channels=ch, dtype="int16", samplerate=rate,
                )
                stream = sd.InputStream(
                    device=dev,
                    channels=ch,
                    samplerate=rate,
                    dtype="int16",
                    blocksize=int(rate * self.chunk_ms / 1000),
                    latency=latency,
                    callback=callback,
                )
                stream.start()
                print(
                    f"[AudioThread] Capturing device={dev} channels={ch} "
                    f"@ {rate} Hz, {self.chunk_ms} ms chunks",
                    flush=True,
                )
                return stream
            except Exception as e:
                print(
                    f"[AudioThread] device={dev} channels={ch} @ {rate} Hz "
                    f"failed: {e}",
                    flush=True,
                )
        return None

    def _open_at_native_rate(self, dev):
        """Fallback for an explicitly-chosen device that rejected
        ``self.sample_rate``: query its own default sample rate and open at
        that instead, resampling every captured block back down to
        ``self.sample_rate`` before it's queued — the server's handshake
        already declared a fixed sample_rate, so we adapt the audio to that
        contract rather than the other way around."""
        try:
            info = sd.query_devices(dev)
            native_rate = int(round(info["default_samplerate"]))
        except Exception as e:
            print(
                f"[AudioThread] could not query native rate for device={dev}: {e}",
                flush=True,
            )
            return None
        if native_rate == self.sample_rate:
            return None  # already tried this rate in _open_at_rate
        stream = self._open_at_rate(
            dev, native_rate, self._make_resampling_callback(native_rate),
            latency="high",
        )
        if stream is not None:
            print(
                f"[AudioThread] device={dev} native rate is {native_rate} Hz "
                f"(not {self.sample_rate} Hz) — resampling in software",
                flush=True,
            )
        return stream

    def _make_resampling_callback(self, native_rate: int):
        """Wrap ``_audio_callback``'s logic with polyphase resampling from
        ``native_rate`` down/up to ``self.sample_rate``. Resampling happens
        per-block (no filter state carried across callbacks), which can
        introduce faint clicks at block boundaries — an acceptable
        trade-off for correctness (capturing from the right device) over
        pristine audio quality."""

        def callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                print(f"[AudioThread] sounddevice status: {status}")
            pcm = indata
            if pcm.dtype != np.int16:
                clipped = np.clip(pcm, -1.0, 1.0)
                pcm = (clipped * 32767.0).astype(np.int16)
            if pcm.ndim == 2 and pcm.shape[1] > 1:
                pcm = pcm.mean(axis=1).astype(np.int16)
            resampled = resample_poly(
                pcm.astype(np.float32), self.sample_rate, native_rate
            )
            resampled = np.clip(np.round(resampled), -32768, 32767).astype(np.int16)
            try:
                self._q.put_nowait(resampled.tobytes())
            except queue.Full:
                try:
                    self._q.get_nowait()
                    self._q.put_nowait(resampled.tobytes())
                except queue.Empty:
                    pass

        return callback

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

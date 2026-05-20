"""WebSocket client for the realtime SDXL streaming protocol.

Forked from `clients/websocket_client.py`. The old per-frame dance
(`{"status": "next_frame"}` → params JSON → image bytes) is gone.
Replaced with:

  * One-shot session-config handshake on connect.
  * Per-frame: single BINARY message
      <4-byte BE image_len><image JPEG><PCM s16le mono>
  * TEXT messages out-of-band:
      - server → client: {"type": "session_ready", "capabilities": {...}}
      - server → client: {"type": "alpha", "alpha": ..., "w_a": ..., "w_b": ...}
      - server → client: {"type": "error", "message": ...}
      - client → server: {"type": "update", "field": ..., "value": ...}
  * Output frames arrive over ZMQ pubsub on port 5555 (NOT this socket).
    The existing `components/processed_display.ZMQThread` still handles
    that and is intentionally untouched.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
from typing import Optional

from PySide6.QtCore import QThread, Signal

import websockets

IS_WINDOWS = os.name == "nt"

if not IS_WINDOWS:
    try:
        import uvloop
    except ImportError:  # pragma: no cover — uvloop missing is fine
        uvloop = None
else:
    uvloop = None


# Default WebSocket endpoint suffix used by the realtime server
# (see web/server_realtime.py in sdxl-travel-ablation).
WS_PATH = "/api/installations/plantoid16/live"


class WSClient(QThread):
    """Realtime WebSocket client.

    Mirrors the Qt-signal-driven structure of the legacy
    `WebSocketClient` so `main.py` integration stays small. The
    thread owns the asyncio event loop; outbound calls
    (`send_frame`, `update_knob`) schedule coroutines onto that loop
    via `asyncio.run_coroutine_threadsafe` so they are safe to call
    from the Qt main thread.
    """

    # ----- Qt signals -----
    capabilities_received = Signal(dict)        # server's session_ready ack
    alpha_updated = Signal(float, float, float, float)  # alpha, w_a, w_b, level
    connection_error = Signal(str)
    status_changed = Signal(str)
    error_message = Signal(str)                 # server-pushed {"type":"error"}
    # Per-frame "you may send the next frame" grant from the server.
    # Glitchbox-faithful — equivalent of the legacy
    # ``{"status":"send_frame"}`` text message in the old protocol. The
    # UI thread holds the most-recent camera frame in a single slot and
    # ships it only when this signal fires, pinning in-flight client→
    # server frames at ≤ 1 (matches glitchbox's per-user queue depth).
    send_frame_granted = Signal()

    def __init__(
        self,
        host: str,
        port: int,
        ws_path: str = WS_PATH,
        max_retries: int = 10,
        initial_retry_delay: float = 1.0,
    ):
        super().__init__()
        self.host = host
        self.port = port
        self.ws_path = ws_path
        self.uri = f"ws://{host}:{port}{ws_path}"

        # The client holds NO render config. It forwards an optional preset
        # NAME (or None → server default) and its own capture params.
        self.preset: Optional[str] = None
        self.sample_rate: int = 44100
        self.fps: int = 20
        self.websocket = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.running = False
        self.connected = False

        # Reconnect / backoff (ported from legacy WebSocketClient)
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.retry_count = 0
        self.retry_delay = initial_retry_delay

    # ------------------------------------------------------------------
    # Public API (callable from any thread)
    # ------------------------------------------------------------------

    def configure(
        self,
        preset: Optional[str] = None,
        sample_rate: int = 44100,
        fps: int = 20,
    ) -> None:
        """Set the handshake parameters to send on next connect.

        ``preset`` is an OPTIONAL server preset name (None → the server
        applies its own configured default). ``sample_rate`` / ``fps`` are
        this client's capture params. Must be called before `start()`.
        """
        self.preset = preset
        self.sample_rate = int(sample_rate)
        self.fps = int(fps)

    def send_frame(self, image_jpeg_bytes: bytes, pcm_bytes: bytes) -> None:
        """Send one camera frame + audio chunk as a single binary message.

        Thread-safe; schedules onto the worker loop.
        """
        if not self.connected or self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_frame_async(image_jpeg_bytes, pcm_bytes), self.loop
        )

    def update_knob(self, field: str, value) -> None:
        """Push a live-adjustable parameter to the server.

        Thread-safe.
        """
        if not self.connected or self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_text({"type": "update", "field": field, "value": value}),
            self.loop,
        )

    def swap_lora(
        self,
        slug_a: str,
        slug_b: str,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
    ) -> None:
        """Request a LoRA hot-swap (slug A/B + per-LoRA fuse weights). Heavy
        — stalls the server a few seconds; the request/grant gate parks us
        meanwhile. Thread-safe.
        """
        if not self.connected or self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._send_text({
                "type": "swap_lora",
                "slug_a": slug_a, "slug_b": slug_b,
                "weight_a": float(weight_a), "weight_b": float(weight_b),
            }),
            self.loop,
        )

    def close(self) -> None:
        """Request graceful shutdown from another thread."""
        self.running = False

    def stop(self) -> None:
        """Stop the client and wait briefly for the thread to exit."""
        print("[WSClient] Stopping")
        self.running = False
        self.connected = False
        if not self.wait(1500):
            print("[WSClient] Thread did not exit cleanly, terminating")
            self.terminate()
            self.wait(500)

    # ------------------------------------------------------------------
    # Async internals (run on the worker loop)
    # ------------------------------------------------------------------

    async def _send_text(self, payload: dict) -> None:
        if self.websocket is None:
            return
        try:
            await self.websocket.send(json.dumps(payload))
        except Exception as e:
            print(f"[WSClient] _send_text failed: {e}")

    async def _send_frame_async(
        self, image_jpeg_bytes: bytes, pcm_bytes: bytes
    ) -> None:
        if self.websocket is None:
            return
        try:
            header = struct.pack(">I", len(image_jpeg_bytes))
            payload = header + image_jpeg_bytes + pcm_bytes
            await self.websocket.send(payload)
        except Exception as e:
            # Don't spam the error signal on every dropped frame;
            # connection_close handler will trigger reconnect.
            print(f"[WSClient] send_frame failed: {e}")

    async def _handshake(self) -> bool:
        """Send the minimal session_config + await session_ready.

        Sends an optional preset name + this client's capture params. The
        server owns all render config; we send NO LoRA / ControlNet /
        dimension state.
        """
        try:
            handshake = {
                "type": "session_config",
                "preset": self.preset,   # None → server default
                "client_av": {
                    "sample_rate": self.sample_rate,
                    "fps": self.fps,
                },
            }
            print(
                f"[WSClient] handshake: preset={self.preset!r} "
                f"sample_rate={self.sample_rate} fps={self.fps}"
            )
            await self.websocket.send(json.dumps(handshake))
            msg = await self.websocket.recv()
            data = json.loads(msg)
            if data.get("type") == "session_ready":
                caps = data.get("capabilities", {})
                self.capabilities_received.emit(caps)
                self.status_changed.emit("connected")
                return True
            self.connection_error.emit(
                f"Expected session_ready, got: {data.get('type')!r}"
            )
            return False
        except Exception as e:
            self.connection_error.emit(f"Handshake failed: {e}")
            return False

    async def _connect_once(self) -> bool:
        try:
            print(f"[WSClient] Connecting to {self.uri}")
            self.websocket = await websockets.connect(self.uri)
            if not await self._handshake():
                await self.websocket.close()
                self.websocket = None
                return False
            self.connected = True
            self.retry_count = 0
            self.retry_delay = self.initial_retry_delay
            return True
        except Exception as e:
            self.connection_error.emit(str(e))
            print(f"[WSClient] Connect failed: {e}")
            return False

    async def _poll_connection(self) -> bool:
        """Retry connect with exponential backoff (capped at 30 s)."""
        while self.running and not self.connected and self.retry_count < self.max_retries:
            self.status_changed.emit(
                f"Retrying connection ({self.retry_count + 1}/{self.max_retries})..."
            )
            if await self._connect_once():
                return True
            self.retry_count += 1
            self.retry_delay = min(
                self.initial_retry_delay * (2 ** (self.retry_count - 1)), 30.0
            )
            await asyncio.sleep(self.retry_delay)
        if not self.connected:
            self.connection_error.emit("Connection failed after maximum retries")
            self.status_changed.emit("disconnected")
            return False
        return True

    async def _recv_loop(self) -> None:
        """Receive TEXT messages (alpha updates, errors) until disconnect."""
        assert self.websocket is not None
        try:
            async for msg in self.websocket:
                if isinstance(msg, bytes):
                    # The realtime server publishes output frames over ZMQ,
                    # not the WS. If a binary frame arrives here, ignore it.
                    continue
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    print(f"[WSClient] Ignoring non-JSON text: {msg!r}")
                    continue
                t = data.get("type")
                if t == "alpha":
                    self.alpha_updated.emit(
                        float(data.get("alpha", 0.0)),
                        float(data.get("w_a", 0.0)),
                        float(data.get("w_b", 0.0)),
                        float(data.get("level", 0.0)),
                    )
                elif t == "send_frame":
                    # Server gating us to ship the next frame. The UI
                    # thread holds the most recent camera frame in a
                    # single-slot buffer; this signal fires the actual
                    # ``send_frame()`` call from the Qt main thread.
                    self._grant_recv_count = (
                        getattr(self, "_grant_recv_count", 0) + 1
                    )
                    if self._grant_recv_count in (1, 5, 20, 100):
                        print(
                            f"[WSClient] grant #{self._grant_recv_count} "
                            "received from server"
                        )
                    self.send_frame_granted.emit()
                elif t == "error":
                    self.error_message.emit(str(data.get("message", "unknown")))
                # Unknown types are silently ignored — forward-compatible.
        except websockets.exceptions.ConnectionClosed:
            print("[WSClient] Server closed the connection")
        except Exception as e:
            if self.running:
                self.connection_error.emit(f"recv loop error: {e}")

    async def _main_loop(self) -> None:
        while self.running:
            ok = await self._poll_connection()
            if not ok:
                return
            try:
                await self._recv_loop()
            finally:
                self.connected = False
                if self.websocket is not None:
                    try:
                        await self.websocket.close()
                    except Exception:
                        pass
                    self.websocket = None
            if not self.running:
                return
            # Otherwise loop back to _poll_connection for reconnect
            self.status_changed.emit("disconnected")
            self.retry_count = 0
            self.retry_delay = self.initial_retry_delay

    def run(self) -> None:
        """QThread entrypoint — owns the asyncio loop for this thread."""
        self.running = True
        if uvloop is not None:
            uvloop.install()
            self.loop = uvloop.new_event_loop()
        else:
            self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main_loop())
        finally:
            try:
                self.loop.close()
            except Exception:
                pass
            self.loop = None

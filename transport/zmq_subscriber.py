"""ZMQ pubsub subscriber for output frames from the realtime server.

The new realtime server publishes JPEG-encoded output frames over a
ZMQ PUB socket on port 5555 (see `web/server_realtime.py` in
sdxl-travel-ablation). This thread subscribes and emits `frame_received`
with a decoded RGB numpy array, ready to feed into the existing
`ProcessedDisplay.update_frame` slot.

The legacy `components.processed_display.ZMQThread` is intentionally
left in place — it handles both raw-bytes and JPEG payloads and is
coupled to `DISPLAY_WIDTH/HEIGHT/SCALE` reshape logic for the old
protocol. This subscriber is JPEG-only and trusts the server's output
resolution.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import zmq
from PySide6.QtCore import QThread, Signal


class ZMQSubscriber(QThread):
    """Subscribe to JPEG output frames from the realtime server."""

    frame_received = Signal(np.ndarray)
    connection_error = Signal(str)

    def __init__(self, host: str, port: int = 5555):
        super().__init__()
        self.host = host
        self.port = port
        self.running = False
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None

    def _open_socket(self) -> bool:
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            address = f"tcp://{self.host}:{self.port}"
            self.socket.connect(address)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)
            self.socket.setsockopt(zmq.LINGER, 0)
            print(f"[ZMQSubscriber] Connected to {address}")
            return True
        except Exception as e:
            self.connection_error.emit(f"ZMQ connect failed: {e}")
            return False

    def run(self) -> None:
        self.running = True
        if not self._open_socket():
            self.running = False
            return
        try:
            while self.running:
                try:
                    data = self.socket.recv()
                except zmq.error.Again:
                    continue
                except zmq.error.ZMQError as e:
                    if not self.running:
                        break
                    self.connection_error.emit(f"ZMQ recv error: {e}")
                    continue
                if not data:
                    continue
                # Trust the realtime server to ship JPEGs only.
                np_buf = np.frombuffer(data, dtype=np.uint8)
                frame_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                self.frame_received.emit(frame_rgb)
        finally:
            self.running = False
            try:
                if self.socket is not None:
                    self.socket.close()
            except Exception:
                pass
            try:
                if self.context is not None:
                    self.context.term()
            except Exception:
                pass
            self.socket = None
            self.context = None
            print("[ZMQSubscriber] Stopped")

    def stop(self) -> None:
        print("[ZMQSubscriber] Stopping")
        self.running = False
        if not self.wait(1500):
            print("[ZMQSubscriber] Thread did not exit cleanly, terminating")
            self.terminate()
            self.wait(500)

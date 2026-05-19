"""Transport layer for the realtime SDXL streaming protocol.

Replaces the legacy WebSocket protocol (next_frame + params + image)
used by `clients/websocket_client.py`. The new protocol is:

  1. Session-config handshake: TEXT frame with the full SessionConfig.
  2. Per-frame BINARY message: <4-byte BE image_len><image JPEG><PCM s16le>.
  3. Output frames arrive out-of-band over ZMQ pubsub on port 5555.
  4. Live-knob updates: TEXT frame {"type":"update","field":...,"value":...}.

The legacy client and `transport/zmq_subscriber.py` semantics still
overlap; the ZMQ subscriber here is a thin re-export of the existing
ZMQThread that `components/processed_display.py` already owns. We do
NOT move that class yet — it is tightly coupled to display config.
"""

from .session_config import SessionConfig, AudioZoomConfig

# Qt-dependent symbols are lazy-imported so `from transport import SessionConfig`
# works in tooling / test contexts without PySide6 installed. They will fail
# loudly on first use, which is the intended behavior.
try:
    from .ws_client import WSClient
    from .zmq_subscriber import ZMQSubscriber
except ImportError as _e:  # PySide6 / zmq not present (e.g. unit tests)
    WSClient = None  # type: ignore[assignment]
    ZMQSubscriber = None  # type: ignore[assignment]
    _IMPORT_ERROR = _e

__all__ = ["SessionConfig", "AudioZoomConfig", "WSClient", "ZMQSubscriber"]

# glitchbox-frontend

Realtime Python client for SDXL streaming diffusion servers (Qt /
PySide6 UI). Captures camera + mic, ships frames over a WebSocket to a
streaming diffusion backend, and renders the returned output frames
that arrive over a ZMQ pubsub channel.

This client speaks the **realtime protocol** used by
[`sdxl-travel-ablation`](https://github.com/) (one-shot
session-config handshake; per-frame `<4-byte BE image_len><image JPEG><PCM s16le mono>`
binary messages; live-knob updates as TEXT). The legacy glitchbox
protocol (next_frame + params JSON + image) is retained as a fallback
under the `GLITCHBOX_REALTIME_V2=0` feature flag.

## Install

```bash
uv sync
```

That's it — creates `.venv` and installs the lean base deps (PySide6,
websockets, sounddevice, pyzmq, opencv-python, …). No torch, no CUDA,
no `source .venv/bin/activate` ever needed: `uv run` targets the
project venv by path.

Optional extras:

```bash
uv sync --extra stt        # speech-to-text (torch + RealtimeSTT + cudnn; NVIDIA GPU)
uv sync --extra depthcam   # RealSense tools under utils/depthcam/
```

## Run

```bash
./start_client.sh          # or plain: uv run main.py
```

The script runs `main.py` via `uv run` (and, only when
`GLITCHBOX_STT_ENABLED=1`, exports `LD_LIBRARY_PATH` for the
stt-extra's cudnn). The realtime server is expected to be reachable
at `ws://${DEFAULT_SERVER_HOST}:${DEFAULT_SERVER_PORT}/api/installations/plantoid16/live`
with the output ZMQ pub socket on
`tcp://${DEFAULT_SERVER_HOST}:${DEFAULT_SERVER_ZMQ_PORT}`.

## Configuration

All runtime config is via environment variables (commonly set in
`.env` next to `main.py`):

| Variable                       | Purpose                                          |
| ------------------------------ | ------------------------------------------------ |
| `DEFAULT_SERVER_HOST`          | Server hostname / IP                             |
| `DEFAULT_SERVER_PORT`          | WebSocket port (FastAPI / uvicorn)               |
| `DEFAULT_SERVER_ZMQ_PORT`      | ZMQ pubsub port for output frames (default 5555) |
| `MIC_DEVICE_INDEX`             | sounddevice / PortAudio input device index       |
| `CAMERA_DEVICE_INDEX`          | `/dev/videoN` index for the input camera         |
| `GLITCHBOX_REALTIME_V2`        | `1` (default) for new protocol, `0` for legacy   |

Additional Qt / camera / FFT tuning lives in `config.py`.

## Architecture

```
transport/                 ← new realtime protocol layer
  session_config.py        SessionConfig + AudioZoomConfig dataclasses
  ws_client.py             WSClient (handshake + binary frames + knob updates)
  zmq_subscriber.py        JPEG-only ZMQ SUB thread for output frames
threads/
  audio_thread.py          Raw PCM s16le mono mic capture, 50 ms chunks
  fft_thread.py            DEPRECATED — kept for V2=0 fallback only
  camera_thread.py         Unchanged
clients/
  websocket_client.py      DEPRECATED — legacy protocol, V2=0 fallback only
```

## Fork lineage

Forked from [`glitchbox/frontend_py`](https://github.com/) @
`9c3b435b765e29004e38fa0a2f98b25a2d75841b` on 2026-05-19 to support the
realtime protocol used by `sdxl-travel-ablation`.

#!/usr/bin/env bash
# Launcher for the glitchbox-frontend realtime client.
#
# - Self-locates: runs from the script's own directory regardless of cwd.
# - Uses `uv run` to pick up the project venv — no `source .venv/bin/activate`
#   anywhere in the flow. `uv sync` once (creates .venv + installs the lean
#   base deps), then this script — or a bare `uv run main.py` — just works.
#   A stale VIRTUAL_ENV exported by a shell/IDE auto-activation is ignored
#   by uv (it targets the project .venv by path), so it's harmless here.
# - The base install has NO torch / CUDA / STT stack. STT is an extra:
#       uv sync --extra stt
#       GLITCHBOX_STT_ENABLED=1 ./start_client.sh
#   Only then do we probe the wheel-bundled nvidia.cudnn libs for
#   LD_LIBRARY_PATH — STT (faster-whisper / ctranslate2) is cudnn's only
#   consumer. With STT off (the default) there's no CUDA dependency at all.
#
# Extra args pass through to main.py, e.g. pick a server preset:
#   ./start_client.sh --preset journey_cn_depthanything
#   ./start_client.sh --host 100.x.y.z --port 8001 --preset vanilla_journey
# (--preset overrides GLITCHBOX_PRESET; default 'vanilla'.)

set -euo pipefail

cd "$(dirname "$0")"

# Unbuffered stdout/stderr so QThread prints (WSClient, AudioThread, …)
# appear immediately in logs instead of sitting in a block buffer when
# stdout is a pipe rather than a TTY.
export PYTHONUNBUFFERED=1

if [[ "${GLITCHBOX_STT_ENABLED:-0}" == "1" ]]; then
    if CUDNN_LIB_DIR="$(uv run python -c 'import os, nvidia.cudnn; print(os.path.join(nvidia.cudnn.__path__[0], "lib"))' 2>/dev/null)"; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CUDNN_LIB_DIR}"
    else
        echo "[start_client] GLITCHBOX_STT_ENABLED=1 but the STT stack is not installed." >&2
        echo "[start_client] Install it with:  uv sync --extra stt" >&2
        exit 1
    fi
fi

exec uv run python main.py "$@"

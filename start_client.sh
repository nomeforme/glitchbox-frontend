#!/usr/bin/env bash
# Launcher for the glitchbox-frontend realtime client.
#
# - Self-locates: runs from the script's own directory regardless of cwd.
# - Uses `uv run` to pick up the project venv (no manual activate needed).
# - Only sets LD_LIBRARY_PATH to the wheel-bundled nvidia.cudnn libs
#   when STT is explicitly enabled — STT is the only consumer of cudnn
#   (faster-whisper / ctranslate2). With STT off (the default), we skip
#   the probe entirely so there's no implicit CUDA dependency at launch.
#
# Enable STT (requires an NVIDIA GPU + driver):
#   GLITCHBOX_STT_ENABLED=1 ./start_client.sh

set -euo pipefail

cd "$(dirname "$0")"

if [[ "${GLITCHBOX_STT_ENABLED:-0}" == "1" ]]; then
    CUDNN_LIB_DIR="$(uv run python -c 'import os, nvidia.cudnn; print(os.path.join(nvidia.cudnn.__path__[0], "lib"))')"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${CUDNN_LIB_DIR}"
fi

exec uv run python main.py

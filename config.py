"""
Configuration settings for the Glitch Machine Engine.
"""

import os

# Display and camera dimensions
DISPLAY_WIDTH = 1024
DISPLAY_HEIGHT = 768
DISPLAY_SCALE = 1.0
CAMERA_DEVICE_INDEX = 0 #0 #42
# None = system default input device. A hardcoded 0 is often NOT a capture
# device (→ PortAudio -9998 invalid-channel-count). Set to a specific input
# index from `python -m sounddevice` if you want a particular mic.
MIC_DEVICE_INDEX = None

# Client-owned AV capture params (sent to the realtime server in the
# handshake — these are facts about THIS machine's mic/camera, not server
# render config). The server's audio chain must match the actual mic rate.
CLIENT_SAMPLE_RATE = 44100
CLIENT_FPS = 20
MAX_CAMERA_INDEX = 50  # Maximum camera index to check (supports virtual cameras like /dev/video42)

# Speech-to-text settings
# STT is OFF by default. Set GLITCHBOX_STT_ENABLED=1 in the environment
# (shell export, NOT .env — config is imported before load_dotenv runs)
# to enable it. When disabled, SpeechToTextThread is never instantiated
# and the UI button is hidden, so no CUDA / RealtimeSTT runtime path is
# touched. The packages are still installed by uv sync — only runtime
# usage is gated.
STT_ENABLED = os.getenv("GLITCHBOX_STT_ENABLED", "0") == "1"
STT_DEVICE = "cuda"  # Device for STT processing: "cpu" or "cuda" (only used when STT_ENABLED)

# Audio settings
NUM_FFT_BINS = 50
FFT_WINDOW_SIZE_MS = 60
SMOOTHING_LENGTH_MS = 1000 #50
EQUALIZER_STRENGTH = 0.10 #0.20
ROLLING_STATS_WINDOW_S = 20

# FFT frequency range settings
FFT_FREQ_START_IDX = 0  # Starting index for frequency range
FFT_FREQ_END_IDX = None  # Ending index for frequency range (None means use all bins)

# UI behavior settings
AUTO_DISABLE_BLACK_FRAME_AFTER_CURATION_UPDATE = True  # Automatically disable black frame mode after successful curation index update
FORCE_MANUAL_RECONNECTION_AFTER_CURATION_UPDATE = True  # Force manual reconnection after curation index update
BLACK_FRAME_DISABLE_TIMEOUT = 0 #75

# Automatic curation index update settings
CURATION_INDEX_AUTO_UPDATE = False  # Enable automatic curation index updates
CURATION_INDEX_UPDATE_TIME = 120  # Update interval in seconds (1 hour = 3600 seconds)
CURATION_INDEX_MAX = 28  # Maximum curation index value (0 to this value)

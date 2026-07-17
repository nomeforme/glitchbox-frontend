import sys
import cv2
import numpy as np
import os
import asyncio
import argparse
import threading
from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QScrollArea, QSpinBox, QFileDialog, QCheckBox, QComboBox
from PySide6.QtCore import Qt, QTimer, Signal, QObject

# Add imports for device detection
import pyaudio
import time
from typing import List, Tuple

from components import CameraDisplay
from components import ProcessedDisplay
from components import ControlPanel
from components import StatusBar
from components.video_display import VideoDisplay
from clients import WebSocketClient
from threads import CameraThread, SpeechToTextThread, FFTAnalyzerThread, VideoThread, VideoAudioThread
from config import MIC_DEVICE_INDEX, AUTO_DISABLE_BLACK_FRAME_AFTER_CURATION_UPDATE, BLACK_FRAME_DISABLE_TIMEOUT, FORCE_MANUAL_RECONNECTION_AFTER_CURATION_UPDATE, CAMERA_DEVICE_INDEX, CURATION_INDEX_AUTO_UPDATE, CURATION_INDEX_UPDATE_TIME, CURATION_INDEX_MAX, MAX_CAMERA_INDEX, STT_ENABLED, CLIENT_SAMPLE_RATE, CLIENT_FPS
from utils.list_cameras import test_camera, get_device_info

# --- Realtime protocol (V2) integration ---------------------------------
# Toggle the V2 path with `GLITCHBOX_REALTIME_V2=0` to fall back to the
# legacy WebSocketClient + FFTAnalyzerThread codepath (still imported
# above). Default is V2 (1).
REALTIME_V2 = os.getenv("GLITCHBOX_REALTIME_V2", "1") == "1"
if REALTIME_V2:
    from transport import WSClient
    from threads.audio_thread import AudioThread
# ------------------------------------------------------------------------

load_dotenv(override=True)

# Default server configuration
DEFAULT_SERVER_HOST = os.getenv("DEFAULT_SERVER_HOST")
DEFAULT_SERVER_PORT = os.getenv("DEFAULT_SERVER_PORT")

class CurationUpdateSignalHandler(QObject):
    """Signal handler for curation index updates"""
    update_completed = Signal(bool, str)  # success, message

def detect_cameras() -> List[Tuple[int, str, str]]:
    """Detect available camera indices with device names and info"""
    available_cameras = []
    for i in range(MAX_CAMERA_INDEX):
        success, message, camera_info = test_camera(i)
        if success:
            # Get device name from sysfs
            device_path = f"/dev/video{i}"
            device_info = get_device_info(device_path)
            device_name = device_info.get('name', f"Camera {i}")

            # Create info string with resolution and FPS
            info = camera_info.get('resolution', 'Unknown')
            if camera_info.get('fps', 0) > 0:
                info += f" @ {camera_info['fps']:.1f}fps"

            available_cameras.append((i, device_name, info))

    return available_cameras

def detect_microphones() -> List[Tuple[int, str]]:
    """Detect available microphone devices"""
    available_mics = []
    try:
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            try:
                device_info = p.get_device_info_by_index(i)
                # Only include devices with input channels
                if device_info['maxInputChannels'] > 0:
                    name = device_info['name']
                    available_mics.append((i, name))
            except Exception:
                pass
        p.terminate()
    except Exception:
        pass
    return available_mics

class MainWindow(QMainWindow):
    def __init__(self, server_host, server_port, preset=None):
        super().__init__()
        self.setWindowTitle("Glitch Machine Engine")
        
        # Store server configuration
        self.server_host = server_host
        self.server_port = server_port
        self.server_ws_uri = f"ws://{server_host}:{server_port}"
        self.server_http_uri = f"http://{server_host}:{server_port}"

        # Realtime preset request: explicit CLI flag (passed here as
        # `preset`) wins, else GLITCHBOX_PRESET env, else None. None means
        # "let the server decide" — the server owns the render config and
        # applies its own configured default preset. The client never needs
        # to know what presets exist; it only forwards a name if the
        # operator explicitly asked for one.
        self.preset = preset or os.getenv("GLITCHBOX_PRESET") or None

        # Initialize device indices from config
        self.camera_device_index = CAMERA_DEVICE_INDEX
        self.audio_device_index = MIC_DEVICE_INDEX
        
        # Video input mode
        self.video_mode = False
        self.video_path = None
        self.video_loop = True
        self.video_fps = None
        
        # Detect available devices
        self.available_cameras = detect_cameras()
        self.available_microphones = detect_microphones()
        
        print(f"[UI] Detected cameras: {[(idx, name, info) for idx, name, info in self.available_cameras]}")
        print(f"[UI] Detected microphones: {[f'{idx}: {name}' for idx, name in self.available_microphones]}")
        
        # Import and set UI behavior config
        self.auto_disable_black_frame_after_curation_update = AUTO_DISABLE_BLACK_FRAME_AFTER_CURATION_UPDATE
        self.force_manual_reconnection_after_curation_update = FORCE_MANUAL_RECONNECTION_AFTER_CURATION_UPDATE
        self.black_frame_disable_timeout = BLACK_FRAME_DISABLE_TIMEOUT

        # Create scroll area as the central widget
        self.scroll_area = QScrollArea()
        self.setCentralWidget(self.scroll_area)
        
        # Configure scroll area properties
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        
        # Enable smooth scrolling and configure for cross-platform compatibility
        self.scroll_area.verticalScrollBar().setSingleStep(20)  # Smooth scrolling step
        self.scroll_area.horizontalScrollBar().setSingleStep(20)
        
        # Set scroll bar styling for better appearance on both Windows and Linux
        scroll_style = """
        QScrollBar:vertical {
            background: #f0f0f0;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #c0c0c0;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        QScrollBar:horizontal {
            background: #f0f0f0;
            height: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background: #c0c0c0;
            border-radius: 6px;
            min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #a0a0a0;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
        }
        """
        self.scroll_area.setStyleSheet(scroll_style)
        
        # Create the main content widget that will be scrollable
        self.main_content_widget = QWidget()
        self.scroll_area.setWidget(self.main_content_widget)
        
        # Create layout for the main content
        layout = QVBoxLayout(self.main_content_widget)
        
        # Set minimum size for the content widget to ensure proper scrolling
        self.main_content_widget.setMinimumSize(800, 600)
        
        # Add status bar at the top
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)
        
        # Video feeds container
        self.feeds_layout = QHBoxLayout()
        
        # Camera feed container
        self.camera_container = QFrame()
        camera_container_layout = QVBoxLayout(self.camera_container)
        
        # Camera feed
        self.camera_display = CameraDisplay()
        camera_container_layout.addWidget(self.camera_display)
        
        # Video feed (initially hidden)
        self.video_display = VideoDisplay()
        self.video_display.setVisible(False)
        camera_container_layout.addWidget(self.video_display)
        
        # Camera label
        self.camera_label = QLabel("Input Camera")
        self.camera_label.setAlignment(Qt.AlignCenter)
        camera_container_layout.addWidget(self.camera_label)
        
        self.feeds_layout.addWidget(self.camera_container)
        
        # Processed feed container
        self.processed_container = QFrame()
        processed_container_layout = QVBoxLayout(self.processed_container)
        
        # Processed feed
        self.processed_display = ProcessedDisplay()
        processed_container_layout.addWidget(self.processed_display)
        
        # Processed label
        self.processed_label = QLabel("Output Stream")
        self.processed_label.setAlignment(Qt.AlignCenter)
        processed_container_layout.addWidget(self.processed_label)
        
        self.feeds_layout.addWidget(self.processed_container)
        
        layout.addLayout(self.feeds_layout)
        
        # Controls section
        self.controls_container = QFrame()
        controls_layout = QVBoxLayout(self.controls_container)
        
        # Device Index Controls - arranged horizontally
        device_controls_layout = QHBoxLayout()
        
        # Curation Index Control
        curation_layout = QHBoxLayout()
        curation_label = QLabel("Curation Index:")
        self.curation_spinbox = QSpinBox()
        self.curation_spinbox.setMinimum(0)
        self.curation_spinbox.setMaximum(CURATION_INDEX_MAX)
        self.curation_spinbox.setValue(0)  # Default value
        self.curation_update_button = QPushButton("Update")
        self.curation_update_button.clicked.connect(self.update_curation_index)
        
        # Connect value change signal to validate the input
        self.curation_spinbox.valueChanged.connect(self.validate_curation_index)
        
        curation_layout.addWidget(curation_label)
        curation_layout.addWidget(self.curation_spinbox)
        curation_layout.addWidget(self.curation_update_button)
        
        device_controls_layout.addLayout(curation_layout)
        
        # Camera Index Control
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera Index:")
        self.camera_spinbox = QSpinBox()
        self.camera_spinbox.setMinimum(0)
        self.camera_spinbox.setMaximum(MAX_CAMERA_INDEX)
        self.camera_spinbox.setValue(self.camera_device_index)
        
        # Set range and tooltip based on detected cameras
        if self.available_cameras:
            camera_list = [f"{idx}: {name} ({info})" for idx, name, info in self.available_cameras]
            self.camera_spinbox.setToolTip("Available cameras:\n" + "\n".join(camera_list))
        else:
            self.camera_spinbox.setToolTip("No cameras detected, but you can still try different indices")
            
        self.camera_update_button = QPushButton("Update")
        self.camera_update_button.clicked.connect(self.update_camera_index)
        
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_spinbox)
        camera_layout.addWidget(self.camera_update_button)
        
        device_controls_layout.addLayout(camera_layout)
        
        # Microphone Index Control
        mic_layout = QHBoxLayout()
        mic_label = QLabel("Microphone Index:")
        self.mic_spinbox = QSpinBox()
        self.mic_spinbox.setMinimum(-1)  # -1 = system default input device
        self.mic_spinbox.setMaximum(50)  # Audio devices can have higher indices
        # audio_device_index may be None (system default) → show as -1.
        self.mic_spinbox.setValue(
            self.audio_device_index if self.audio_device_index is not None else -1
        )
        
        # Set tooltip with detected microphones
        if self.available_microphones:
            mic_list = [f"{idx}: {name[:30]}..." if len(name) > 30 else f"{idx}: {name}" 
                       for idx, name in self.available_microphones]
            self.mic_spinbox.setToolTip("Available microphones:\n" + "\n".join(mic_list))
        else:
            self.mic_spinbox.setToolTip("No microphones detected, but you can still try different indices")
            
        self.mic_update_button = QPushButton("Update")
        self.mic_update_button.clicked.connect(self.update_mic_index)
        
        mic_layout.addWidget(mic_label)
        mic_layout.addWidget(self.mic_spinbox)
        mic_layout.addWidget(self.mic_update_button)
        
        device_controls_layout.addLayout(mic_layout)
        
        # Refresh Devices Control
        refresh_layout = QHBoxLayout()
        refresh_label = QLabel("Device Lists:")
        self.refresh_devices_button = QPushButton("Refresh")
        self.refresh_devices_button.clicked.connect(self.refresh_device_lists)
        self.refresh_devices_button.setToolTip("Re-scan for available cameras and microphones")
        
        refresh_layout.addWidget(refresh_label)
        refresh_layout.addWidget(self.refresh_devices_button)
        
        device_controls_layout.addLayout(refresh_layout)
        
        # Video Input Controls
        video_layout = QHBoxLayout()
        video_label = QLabel("Video Input:")
        
        # Video mode toggle
        self.video_mode_checkbox = QCheckBox("Enable Video Mode")
        self.video_mode_checkbox.setToolTip("Switch from camera/microphone to video file input")
        self.video_mode_checkbox.stateChanged.connect(self.toggle_video_mode)
        
        # Video file selection
        self.video_file_button = QPushButton("Select Video File")
        self.video_file_button.clicked.connect(self.select_video_file)
        self.video_file_button.setEnabled(False)
        self.video_file_button.setToolTip("Select a video file to use as input")
        
        # Video file path display
        self.video_path_label = QLabel("No video file selected")
        self.video_path_label.setStyleSheet("color: gray; font-style: italic;")
        self.video_path_label.setToolTip("Selected video file path")
        
        # Video loop toggle
        self.video_loop_checkbox = QCheckBox("Loop Video")
        self.video_loop_checkbox.setChecked(True)
        self.video_loop_checkbox.setEnabled(False)
        self.video_loop_checkbox.setToolTip("Loop the video when it reaches the end")
        
        video_layout.addWidget(video_label)
        video_layout.addWidget(self.video_mode_checkbox)
        video_layout.addWidget(self.video_file_button)
        video_layout.addWidget(self.video_path_label)
        video_layout.addWidget(self.video_loop_checkbox)
        
        device_controls_layout.addLayout(video_layout)
        
        # Add stretch to push everything to the left
        device_controls_layout.addStretch()
        
        controls_layout.addLayout(device_controls_layout)
        
        # Create horizontal layout for buttons
        buttons_layout = QHBoxLayout()
        
        # Single idempotent connection toggle. One button switches between
        # Connect and Disconnect; "reconnect" is just Connect again (it tears
        # down any running client first), so first-connect, manual reconnect,
        # and the server-initiated auto-reconnect all share one code path.
        # See toggle_connection / connect_to_server / disconnect_from_server.
        self.connect_button = QPushButton("Connect to Server")
        self.connect_button.clicked.connect(self.toggle_connection)
        buttons_layout.addWidget(self.connect_button)
        
        # Start/Stop Camera button
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        # Camera button is now always enabled
        buttons_layout.addWidget(self.start_button)
        
        # STT Toggle button (gated by STT_ENABLED — hidden by default,
        # enable with GLITCHBOX_STT_ENABLED=1 in the environment).
        if STT_ENABLED:
            self.stt_button = QPushButton("Start Speech Recognition")
            self.stt_button.clicked.connect(self.toggle_stt)
            buttons_layout.addWidget(self.stt_button)
        else:
            self.stt_button = None
        
        # FFT Toggle button
        self.fft_button = QPushButton("Start Audio FFT")
        self.fft_button.clicked.connect(self.toggle_fft)
        buttons_layout.addWidget(self.fft_button)

        # Sound Lab window (server policy-mode telemetry visualizer).
        # Built lazily on first click; telemetry only flows when the
        # server session runs audio_alpha_mode="policy".
        self.soundlab_view = None
        self.soundlab_button = QPushButton("Sound Lab")
        self.soundlab_button.clicked.connect(self.toggle_soundlab)
        buttons_layout.addWidget(self.soundlab_button)

        # NOTE: ProjectionMapper has its own toggle button inside
        # ProcessedDisplay ("Projection Mapper" button, see
        # components/processed_display.py:222). Frames are routed to the
        # mapper from ProcessedDisplay.update_frame() — so it works
        # transparently across both V1 (WS frame_received) and V2 (ZMQ)
        # output paths. No additional wiring in main.py needed.

        # Add buttons layout to controls
        controls_layout.addLayout(buttons_layout)
        
        # Pipeline controls
        self.control_panel = ControlPanel()
        self.control_panel.parameter_changed.connect(self.update_parameter)
        controls_layout.addWidget(self.control_panel)
        
        layout.addWidget(self.controls_container)
        
        # Presentation mode buttons
        presentation_layout = QHBoxLayout()
        
        self.toggle_input_button = QPushButton("Hide Input Feed")
        self.toggle_input_button.clicked.connect(self.toggle_input_feed)
        presentation_layout.addWidget(self.toggle_input_button)
        
        self.toggle_controls_button = QPushButton("Hide Controls")
        self.toggle_controls_button.clicked.connect(self.toggle_controls)
        presentation_layout.addWidget(self.toggle_controls_button)
        
        self.toggle_black_frame_button = QPushButton("Enable Black Frame")
        self.toggle_black_frame_button.clicked.connect(self.toggle_black_frame)
        presentation_layout.addWidget(self.toggle_black_frame_button)

        self.toggle_mirror_button = QPushButton("Enable Mirror")
        self.toggle_mirror_button.clicked.connect(self.toggle_mirror)
        presentation_layout.addWidget(self.toggle_mirror_button)
        
        self.toggle_presentation_button = QPushButton("Enter Presentation Mode")
        self.toggle_presentation_button.clicked.connect(self.toggle_presentation_mode)
        presentation_layout.addWidget(self.toggle_presentation_button)
        
        self.toggle_fullscreen_button = QPushButton("Detach Output")
        self.toggle_fullscreen_button.clicked.connect(self.toggle_fullscreen)
        presentation_layout.addWidget(self.toggle_fullscreen_button)
        
        # Add automatic curation update toggle button
        self.toggle_auto_curation_button = QPushButton("Start Auto Curation Updates")
        self.toggle_auto_curation_button.clicked.connect(self.toggle_automatic_curation_updates)
        self.toggle_auto_curation_button.setToolTip(f"Toggle automatic curation index updates (every {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
        presentation_layout.addWidget(self.toggle_auto_curation_button)
        
        # Add force terminate button
        self.force_terminate_button = QPushButton("Exit")
        self.force_terminate_button.setStyleSheet("background-color: #ff4444; color: white;")
        self.force_terminate_button.clicked.connect(self.force_terminate)
        presentation_layout.addWidget(self.force_terminate_button)
        
        layout.addLayout(presentation_layout)
        
        # Initialize threads
        self.ws_client = WebSocketClient(uri=self.server_ws_uri, max_retries=10, initial_retry_delay=1.0)
        # Single source of truth for camera-thread construction + signal
        # wiring (see _create_camera_thread). Every recreation path routes
        # through it so the V2 send gate is never left unwired.
        self._create_camera_thread()

        # Initialize the STT thread with the audio device index (gated by STT_ENABLED).
        # When disabled, self.stt_thread stays None — existing cleanup paths use
        # `hasattr(...) and self.stt_thread is not None` so they short-circuit safely.
        if STT_ENABLED:
            self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
            self.stt_thread.transcription_updated.connect(self.handle_transcription)
        else:
            self.stt_thread = None
            print("[UI] STT disabled (set GLITCHBOX_STT_ENABLED=1 to enable)")
        self.stt_active = False
        
        # Initialize the FFT thread with the audio device index
        self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
        self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
        self.fft_active = False
        
        # Initialize video threads
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.handle_camera_frame)
        self.video_thread.video_finished.connect(self.handle_video_finished)
        
        self.video_audio_thread = VideoAudioThread()
        self.video_audio_thread.fft_data_updated.connect(self.handle_fft_data)
        self.video_audio_thread.audio_finished.connect(self.handle_video_finished)
        
        # Connect video display signals
        self.video_display.signals.frame_ready.connect(self.handle_video_frame)
        self.video_display.signals.fft_data_ready.connect(self.handle_fft_data)
        self.video_display.signals.video_finished.connect(self.handle_video_finished)
        
        # Connect signals
        self.ws_client.frame_received.connect(self.processed_display.update_frame)
        self.ws_client.connection_error.connect(self.handle_connection_error)
        self.ws_client.settings_received.connect(self.handle_settings)
        self.ws_client.status_changed.connect(self.handle_status_change)
        self.ws_client.param_updated.connect(self.handle_param_update)
        # camera_thread.frame_ready is wired in _create_camera_thread().

        # --- Realtime V2 wiring ---------------------------------------
        # Build a SessionConfig from server-side defaults (the realtime
        # server's /api/installations/plantoid16/defaults endpoint),
        # construct a WSClient, and wire the per-frame joiner. The
        # The client is a dumb AV terminal: it forwards an OPTIONAL preset
        # name + its own capture params (sample_rate, fps), and displays
        # whatever frames the server renders. It holds NO render config
        # (LoRA / ControlNet / dimensions / prompts all live server-side).
        # This is what lets the client run on a machine with zero knowledge
        # of the server's setup.
        if REALTIME_V2:
            self.ws_client_v2 = WSClient(
                host=server_host,
                port=int(server_port),
                max_retries=10,
                initial_retry_delay=1.0,
            )
            self.ws_client_v2.configure(
                preset=self.preset,
                sample_rate=CLIENT_SAMPLE_RATE,
                fps=CLIENT_FPS,
            )
            # Render the server-advertised knob schema, THEN classify
            # (enable live / grey-out frozen). _v2_build_controls runs
            # first so apply_capabilities has controls to operate on.
            self.ws_client_v2.capabilities_received.connect(
                self._v2_build_controls
            )
            # V2 handshake-complete also triggers ProcessedDisplay's ZMQ
            # subscriber. Without this, the server emits rendered frames
            # on tcp://*:5555 but the client never subscribes to the
            # pubsub, so the output window stays blank even though the
            # GPU is actively rendering. (Legacy parity: V1 wires this
            # off handle_settings, which the realtime server never
            # triggers — its /api/settings returns 404.)
            self.ws_client_v2.capabilities_received.connect(
                self._v2_on_capabilities
            )
            self.ws_client_v2.connection_error.connect(self.handle_connection_error)
            self.ws_client_v2.status_changed.connect(self.handle_status_change)
            # Read-only prompt-travel context → projection mapper modal.
            self.ws_client_v2.prompt_context_updated.connect(
                self._v2_on_prompt_context
            )
            # Soundlab policy telemetry → Sound Lab window (history
            # accumulates even while the window is hidden, so opening it
            # mid-set shows the recent past, not a blank chart).
            self.ws_client_v2.soundlab_updated.connect(self._v2_handle_soundlab)
            # Live-knob updates flow control_panel → ws_client_v2.update_knob
            self.control_panel.knob_changed.connect(self.ws_client_v2.update_knob)
            # LoRA hot-swap (Load button) → ws_client_v2.swap_lora
            self.control_panel.lora_swap_requested.connect(
                self.ws_client_v2.swap_lora
            )
            # Remember the last LoRA pair the user loaded so a reconnect can
            # restore it — the server rebuilds the session at the preset's
            # default LoRA, so this must be re-fired (see _v2_restore_lora).
            self._lora_selection = None
            self.control_panel.lora_swap_requested.connect(self._v2_remember_lora)
            # Remember the latest value of each live knob the user changes, so
            # a reconnect can restore the session's active settings instead of
            # snapping back to the server's defaults. The control panel is
            # rebuilt from defaults on every handshake (see _v2_build_controls);
            # _v2_restore_knobs replays this dict afterwards.
            self._knob_values: dict = {}
            self.control_panel.knob_changed.connect(self._v2_remember_knob)

            # Audio thread (raw PCM, no client-side FFT). The client OWNS
            # its mic rate (CLIENT_SAMPLE_RATE) and ships it to the server
            # in the handshake (client_av.sample_rate) — the server builds
            # RealtimeFFTAudioAnalyzer with that exact value, so they can't
            # drift.
            self.audio_thread = AudioThread(
                input_device_index=self.audio_device_index,
                sample_rate=CLIENT_SAMPLE_RATE,
                chunk_ms=50,
            )
            self.audio_thread.pcm_chunk_ready.connect(self._v2_handle_pcm_chunk)

            # Joiner state: keep the most recent PCM chunk, pair on each
            # camera frame (camera tick is the dispatch trigger so we
            # send at most one packet per rendered frame).
            self._v2_latest_pcm = b""
            # Glitchbox-faithful gate state. The camera thread emits at
            # webcam hardware rate (~30 fps) regardless of network /
            # server speed; the slot pattern decouples that from WS
            # send rate. Frames pile up into ``_v2_latest_frame``
            # (overwriting older ones), and the server's per-frame
            # ``{"type":"send_frame"}`` grant is what actually fires a
            # ``ws_client_v2.send_frame()`` call.
            #
            # Two-state machine:
            #   * grant arrives, slot empty   → set _v2_grant_held=True
            #   * grant arrives, slot has frame → ship it, clear slot
            #   * frame arrives, _v2_grant_held=True → ship now, clear
            #   * frame arrives, _v2_grant_held=False → just stash in slot
            self._v2_latest_frame: bytes | None = None
            self._v2_grant_held = False
            # camera_thread.frame_ready → _v2_handle_camera_frame is wired in
            # _create_camera_thread() so every recreation path keeps the gate.
            self.ws_client_v2.send_frame_granted.connect(
                self._v2_handle_grant
            )

            # Alpha telemetry → status bar (throttled to ~1 Hz so we
            # don't spam the UI at 20-30 fps).
            self._v2_alpha_throttle = 0
            self.ws_client_v2.alpha_updated.connect(self._v2_handle_alpha)
        # --------------------------------------------------------------
        
        # Track signal connections to prevent duplication
        self.signal_connections_active = True
        
        # Frame processing timer
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.process_frame)
        self.frame_timer.setInterval(33)  # ~30 FPS
        
        # Automatic curation index update timer
        self.curation_auto_timer = QTimer()
        self.curation_auto_timer.timeout.connect(self.perform_automatic_curation_update)
        self.curation_auto_timer.setInterval(CURATION_INDEX_UPDATE_TIME * 1000)  # Convert seconds to milliseconds
        
        # Track reconnection attempts to prevent infinite loops
        self.reconnection_count = 0
        self.max_reconnection_attempts = 5
        
        # State variables
        self.current_frame = None
        self.processing_frame = False
        self.presentation_mode = False
        self.black_frame_enabled = False
        
        # Add separate state tracking for camera and server connection
        self.camera_running = False
        self.server_connected = False
        # User intent for the connection toggle. True from the moment the
        # user clicks Connect until they click Disconnect — independent of
        # transient socket drops, so the button label stays meaningful while
        # the client auto-reconnects in the background.
        self._user_wants_connected = False

        # Create signal handler for curation updates
        self.curation_signal_handler = CurationUpdateSignalHandler()
        self.curation_signal_handler.update_completed.connect(self._handle_curation_update_result)

        # Don't automatically connect to server - user will click Connect button
        # self.get_initial_settings()
        
        # Set initial status message
        self.status_bar.update_processing_status("Ready - Click 'Connect to Server' to connect")
        
        # Log automatic curation update configuration
        if CURATION_INDEX_AUTO_UPDATE:
            print(f"[UI] Automatic curation updates enabled (interval: {CURATION_INDEX_UPDATE_TIME} seconds)")
            print(f"[UI] Curation index range: 0 to {CURATION_INDEX_MAX}")
            print(f"[UI] Timer will start automatically when connected to server")
        else:
            print("[UI] Automatic curation updates disabled in config")
        print(f"[UI] Curation index range: 0 to {CURATION_INDEX_MAX}")

    def _refresh_connection_button(self):
        """Single source of truth for the connection toggle button.

        The label tracks user *intent* (`_user_wants_connected`), not the
        transient socket state, so the button stays meaningful while the
        client is mid-(re)connect: it reads "Disconnect from Server" the
        whole time you want to be connected — letting you abort a stuck
        reconnect — and "Connect to Server" once you've disconnected. Always
        enabled; that's what makes the toggle idempotent.
        """
        if self._user_wants_connected:
            self.connect_button.setText("Disconnect from Server")
        else:
            self.connect_button.setText("Connect to Server")
        self.connect_button.setEnabled(True)

    def toggle_connection(self):
        """Idempotent Connect/Disconnect toggle — the single button's slot."""
        if self._user_wants_connected:
            self.disconnect_from_server()
        else:
            self.connect_to_server()

    def _on_connection_terminated(self, reason: str):
        """The client gave up (max retries) or failed to start.

        Returns the UI to the disconnected state so the toggle reads
        "Connect to Server" and the next click starts a fresh attempt.
        """
        self._user_wants_connected = False
        self.server_connected = False
        self.status_bar.update_connection_status(False)
        self._refresh_connection_button()
        self.status_bar.update_processing_status(reason)

    def _start_network_clients(self):
        """Start the active protocol's WebSocket client (idempotent).

        The single connect primitive shared by first-connect, manual
        reconnect, and the curation-triggered reconnect. Any client still
        running is stopped first, so every path is identical.

        Only the client for the active protocol is started: under
        REALTIME_V2 that's the realtime ``WSClient``. The legacy client is
        left idle — starting it against the realtime server just spams
        ``/api/settings`` with 404s and races the V2 status signals.
        """
        # Clean slate — this is what makes connect and reconnect one path.
        self._stop_network_clients()

        # Fresh connection: let the one-shot V2 gate diagnostics print again.
        self._v2_cam_disconnect_logged = False
        self._v2_first_cam_frame_logged = False
        self._v2_first_send_logged = False

        try:
            if REALTIME_V2 and hasattr(self, "ws_client_v2"):
                print("[UI] Starting V2 WSClient (realtime protocol)")
                self.ws_client_v2.start()
            else:
                print(f"[UI] Starting legacy WebSocket client: {self.ws_client.uri}")
                self.ws_client.start()
        except Exception as e:
            print(f"[UI] Error starting WebSocket client: {e}")
            self._on_connection_terminated(f"Failed to start connection: {e}")

    def _stop_network_clients(self):
        """Stop whichever WebSocket client threads are running (idempotent).

        Stops BOTH clients if alive. Earlier builds only stopped the legacy
        client, so the realtime thread leaked across disconnect/reconnect and
        kept auto-retrying behind the UI's back.
        """
        v2 = getattr(self, "ws_client_v2", None)
        if v2 is not None:
            try:
                if v2.isRunning():
                    v2.stop()
            except Exception as e:
                print(f"[UI] Error stopping V2 client: {e}")
        v1 = getattr(self, "ws_client", None)
        if v1 is not None:
            try:
                v1.close()
                if v1.isRunning():
                    v1.stop()
            except Exception as e:
                print(f"[UI] Error stopping legacy client: {e}")

    def connect_to_server(self):
        """Connect to the server (also serves as reconnect).

        Idempotent: records that the user wants to be connected, then
        (re)starts the client from a clean state. Safe to call when already
        connected — it simply reconnects.
        """
        print("[UI] Connect requested")
        self._user_wants_connected = True
        self._refresh_connection_button()
        self.status_bar.update_processing_status("Connecting to server...")
        self._start_network_clients()

    def disconnect_from_server(self):
        """Disconnect from the server and tear down streaming (idempotent).

        Safe to call when already disconnected — it just settles the UI into
        the disconnected state. Stops BOTH client threads so nothing keeps
        auto-reconnecting after the user explicitly disconnects.
        """
        print("[UI] Disconnect requested")
        self._user_wants_connected = False
        self.server_connected = False
        self._refresh_connection_button()
        self.status_bar.update_connection_status(False)
        self.status_bar.update_processing_status("Disconnecting from server...")

        # Stop frame processing + output display immediately.
        self.frame_timer.stop()
        self.processing_frame = False
        self.processed_display.clear_zmq_queue()
        self.processed_display.stop_stream()
        self.processed_display.clear_display()

        # Stop the automatic curation update timer.
        if self.curation_auto_timer.isActive():
            self.curation_auto_timer.stop()
            if hasattr(self, 'toggle_auto_curation_button'):
                self.toggle_auto_curation_button.setText("Start Auto Curation Updates")

        # Tell the server to stop streaming (legacy no-op under V2), then
        # stop the client thread(s). _stop_network_clients() is bounded and
        # terminate-free now that close() interrupts the socket/backoff.
        if self.camera_running:
            try:
                self.ws_client.stop_camera()
            except Exception as e:
                print(f"[UI] Error stopping camera streaming: {e}")
        self._stop_network_clients()

        # Keep the local camera preview alive if the camera is still on.
        if self.camera_running:
            if not self.frame_timer.isActive():
                self.frame_timer.start()
            self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")
        else:
            self.status_bar.update_processing_status("Disconnected from server")
        print("[UI] Server disconnection completed")

    def handle_settings(self, settings):
        """Handle received pipeline settings"""
        self.control_panel.setup_pipeline_options(settings)
        self.server_connected = True
        
        # Connected — reflect it on the single toggle button.
        self._user_wants_connected = True
        self._refresh_connection_button()
        self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
        
        # Reset reconnection count on successful connection
        self.reconnection_count = 0
        
        # Ensure ProcessedDisplay streaming is started (important for reconnection)
        print("[UI] Starting ProcessedDisplay streaming...")
        try:
            # Stop any existing streams first to prevent duplication
            self.processed_display.stop_stream()
            
            # Add delay to ensure cleanup is complete
            QTimer.singleShot(100, lambda: self._start_fresh_stream())
            print("[UI] ProcessedDisplay streaming restart scheduled")
        except Exception as e:
            print(f"[UI] Error starting ProcessedDisplay streaming: {e}")
        
        # If camera is already running, start streaming automatically
        if self.camera_running:
            self.ws_client.start_camera()
            self.status_bar.update_processing_status("Streaming frames to server...")
        
        # Restart streaming components
        self._restart_streaming_components()
        
        # Update curation index from server settings
        current_curation_index = settings.get('current_curation_index', 0)
        self.curation_spinbox.setValue(current_curation_index)
        print(f"[UI] Set curation index to {current_curation_index} from server settings")
        
        # Start automatic curation update timer if enabled
        if CURATION_INDEX_AUTO_UPDATE:
            print(f"[UI] Starting automatic curation update timer (interval: {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
            self.curation_auto_timer.start()
            print(f"[UI] Automatic curation update timer is now active")
            # Update button text to reflect active state
            if hasattr(self, 'toggle_auto_curation_button'):
                self.toggle_auto_curation_button.setText("Stop Auto Curation Updates")
        else:
            print("[UI] Automatic curation updates disabled in config")
            # Update button text to reflect inactive state
            if hasattr(self, 'toggle_auto_curation_button'):
                self.toggle_auto_curation_button.setText("Start Auto Curation Updates")

    def handle_param_update(self, params: dict):
        """Handle param updates pushed from server (e.g., from gRPC)"""
        print(f"[UI] Received param update from server: {list(params.keys())}")
        for param_id, value in params.items():
            # Update control panel UI
            self.control_panel.update_control(param_id, value)
        self.status_bar.update_processing_status(f"Params updated from server: {', '.join(params.keys())}")

    def handle_status_change(self, status: str):
        """Handle WebSocket status changes."""
        if status.startswith("Retrying connection") or status.startswith("Retrying to fetch"):
            self.status_bar.update_processing_status(
                f"Reconnecting to {self.server_host}:{self.server_port}..."
            )
        elif status == "connected":
            self.server_connected = True
            self.status_bar.update_connection_status(True)
            self.status_bar.update_processing_status(
                f"Connected to server: {self.server_host}:{self.server_port}"
            )
            # Stream is (re)started by handle_settings() (V1) /
            # _v2_on_capabilities() (V2); don't start it here (double-start
            # inflates the FPS counter).
            self._user_wants_connected = True
            self._refresh_connection_button()
        elif status == "disconnected":
            # A drop. If the user still wants to be connected, the client
            # thread is auto-retrying in the background — keep the toggle on
            # "Disconnect" and just reflect the transient state. We only fall
            # back to the "Connect" state on an explicit disconnect or when
            # the client gives up (see handle_connection_error).
            self.server_connected = False
            self.status_bar.update_connection_status(False)
            self.processed_display.stop_stream()
            if self.curation_auto_timer.isActive():
                self.curation_auto_timer.stop()
                if hasattr(self, 'toggle_auto_curation_button'):
                    self.toggle_auto_curation_button.setText("Start Auto Curation Updates")
            self._refresh_connection_button()
            if self._user_wants_connected:
                self.status_bar.update_processing_status(
                    f"Connection lost — reconnecting to {self.server_host}:{self.server_port}..."
                )
            elif self.camera_running:
                self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")
            else:
                self.status_bar.update_processing_status("Disconnected from server")
        elif status == "ready":
            # Emitted when camera streaming is stopped on the server side.
            if self.camera_running and self.server_connected:
                self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
            elif self.camera_running:
                self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")
        elif status == "Connection failed after maximum retries":
            self._on_connection_terminated("Connection failed - click Connect to retry")

    def handle_camera_frame(self, frame):
        """Handle new frame from camera"""
        self.current_frame = frame
        self.camera_display.update_frame(frame)

    # --- Realtime V2 helpers --------------------------------------------
    def _v2_build_controls(self, caps: dict):
        """Render the server's knob schema, then classify live/frozen.

        The server advertises the realtime-adjustable knobs (id, type,
        range, current value) in ``caps['controls']`` — same shape the
        legacy /api/settings InputParams used. We render them generically
        (the client never learns what a knob means) and then let
        apply_capabilities enable the live ones + wire knob_changed →
        ws_client_v2.update_knob → server.update(field, value).
        """
        controls = caps.get("controls", {})
        # Sound Lab is the audio control center: audio-group knobs render
        # there; everything else stays in the main control panel.
        audio_schema = {k: v for k, v in controls.items()
                        if v.get("group") == "audio"}
        rest = {k: v for k, v in controls.items()
                if v.get("group") != "audio"}
        if controls:
            self.control_panel.setup_pipeline_options(
                {"input_params": {"properties": rest}}
            )
        self.control_panel.apply_capabilities(caps)
        view = self._ensure_soundlab_view()
        view.build_audio_controls(audio_schema)
        view.set_graph_presets(caps.get("soundlab_graphs") or {})
        # Re-apply the client's last live-knob values so a reconnect resumes
        # where the session left off instead of snapping back to defaults.
        self._v2_restore_knobs(set(caps.get("live_adjustable", [])))
        # Same for the last-loaded LoRA pair (deferred Load, not a live knob).
        self._v2_restore_lora()
        # Mic presence is a CLIENT fact (the mic lives here; the server
        # may be a different machine and only ever sees our PCM stream).
        # audio+policy are the server defaults, which suit a mic-bearing
        # client; a MIC-LESS client decides for itself and applies the
        # `ramp` graph preset so the deck still moves without audio.
        if (not self.available_microphones
                and "soundlab_graph" not in self._knob_values
                and getattr(self, "ws_client_v2", None) is not None):
            import json as _json
            ramp_g = (caps.get("soundlab_graphs") or {}).get("ramp")
            if ramp_g:
                gjson = _json.dumps(ramp_g)
                view.nodes_editor.load_graph(ramp_g)   # reflect in editor
                self.ws_client_v2.update_knob("soundlab_graph", gjson)
                self._knob_values["soundlab_graph"] = gjson
                print("[UI/V2] No microphone detected → applied `ramp` "
                      "graph preset (client-side decision)")

    def _v2_remember_knob(self, field, value):
        """Track the latest value of each live knob (see _v2_restore_knobs)."""
        self._knob_values[field] = value

    def _v2_remember_lora(self, slug_a, slug_b, weight_a, weight_b):
        """Remember the last LoRA pair the user loaded (see _v2_restore_lora)."""
        self._lora_selection = (slug_a, slug_b, float(weight_a), float(weight_b))

    def _v2_restore_lora(self):
        """Re-apply the client's last LoRA selection after a (re)connect.

        Restores the A/B dropdowns + fuse weights and re-fires the swap on
        the server, because a reconnect rebuilds the session at the preset's
        default LoRA. Heavy — a swap stalls the server a few seconds — so it
        only fires when the user actually loaded a pair this session AND both
        slugs still exist in the rebuilt dropdowns (e.g. not after a preset
        switch). No-op on first connect (nothing remembered yet).
        """
        if not self._lora_selection:
            return
        slug_a, slug_b, weight_a, weight_b = self._lora_selection
        if not self.control_panel.set_lora_selection(slug_a, slug_b, weight_a, weight_b):
            print(
                f"[UI/V2] Skipping LoRA restore — slug(s) not in rebuilt list: "
                f"{slug_a!r}, {slug_b!r}"
            )
            return
        if getattr(self, "ws_client_v2", None) is not None:
            self.ws_client_v2.swap_lora(slug_a, slug_b, weight_a, weight_b)
            print(
                f"[UI/V2] Restored LoRA after reconnect: {slug_a} + {slug_b} "
                f"(w_a={weight_a}, w_b={weight_b})"
            )

    def _v2_restore_knobs(self, live_fields: set):
        """Replay the client's last live-knob values after a (re)connect.

        The control panel is rebuilt from the server's advertised defaults on
        every handshake, which would otherwise wipe adjustments made during
        the session. We replay the remembered live values — updating both the
        widget and the server — so reconnecting resumes where the session left
        off. Only fields that still exist and are live-adjustable under the new
        capabilities are restored (stale or frozen fields are skipped).
        """
        if not self._knob_values:
            return
        restored = []
        lab = self.soundlab_view   # may be None before first handshake
        lab_controls = getattr(lab, "audio_controls", {}) if lab else {}
        # knobs with dedicated Sound Lab surfaces (lever / graph): no
        # widget lookup — just replay the value to the server
        passthrough = {"soundlab_mix", "soundlab_graph", "manual_alpha",
                       "audio_alpha_mode"}
        # Snapshot: update_control() re-enters _v2_remember_knob via the
        # widget's own signal in the live window, mutating _knob_values.
        for field, value in list(self._knob_values.items()):
            in_panel = field in self.control_panel.controls
            in_lab = field in lab_controls
            if not (in_panel or in_lab or field in passthrough):
                continue
            if live_fields and field not in live_fields:
                continue
            if in_panel:
                self.control_panel.update_control(field, value)
            elif in_lab:
                lab.update_control(field, value)
            if getattr(self, "ws_client_v2", None) is not None:
                self.ws_client_v2.update_knob(field, value)    # push to server
            restored.append(field)
        if restored:
            print(f"[UI/V2] Restored {len(restored)} client knob(s) after reconnect: {restored}")

    def _v2_handle_pcm_chunk(self, pcm_bytes: bytes):
        """ACCUMULATE PCM for the next camera frame (soundlab finding:
        keep-latest dropped half the audio at <20 fps render rates — the
        server's audio clock ran at 0.5x and every time constant smeared).
        The server steps its audio chain once per 50 ms sub-chunk, so
        sending everything keeps the chain on the true clock. Cap ~1 s
        so a stall can't balloon a frame."""
        if len(self._v2_latest_pcm) < 88200:   # 1 s @ 44.1 kHz int16
            self._v2_latest_pcm += pcm_bytes
        else:
            self._v2_latest_pcm = pcm_bytes

    def _v2_handle_camera_frame(self, frame):
        """Stash the latest camera frame (overwriting older ones).

        Glitchbox-faithful: the camera thread runs at hardware rate
        (~30 fps) regardless of GPU throughput. We never directly fire
        ``send_frame()`` here — that would re-introduce the unbounded
        TCP buffer fill the request-grant gate is designed to prevent.
        Instead we update a single-slot buffer; the server's per-frame
        ``send_frame`` grant (see ``_v2_handle_grant``) is what
        actually ships a frame.
        """
        if not REALTIME_V2:
            return
        if not getattr(self, "ws_client_v2", None) or not self.ws_client_v2.connected:
            # One-time print so we can see if frames are coming in but
            # the gate is blocking. After connect this should never
            # short-circuit.
            if not getattr(self, "_v2_cam_disconnect_logged", False):
                print(
                    f"[V2 GATE] camera_frame arriving but ws not connected — "
                    f"ws_client_v2={bool(getattr(self, 'ws_client_v2', None))}, "
                    f"connected={getattr(self.ws_client_v2, 'connected', None) if getattr(self, 'ws_client_v2', None) else 'n/a'}"
                )
                self._v2_cam_disconnect_logged = True
            return
        if not getattr(self, "_v2_first_cam_frame_logged", False):
            print(
                f"[V2 GATE] first camera frame received post-connect; "
                f"_v2_grant_held={self._v2_grant_held}"
            )
            self._v2_first_cam_frame_logged = True
        try:
            # camera_thread emits RGB; encode as JPEG (BGR for cv2).
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr)
            if not ok:
                return
            jpeg = buf.tobytes()
            if self._v2_grant_held:
                # Server already granted but we had no frame to send.
                # Ship this one immediately and consume the grant.
                self._v2_grant_held = False
                self._v2_latest_frame = None
                self.ws_client_v2.send_frame(jpeg, self._v2_latest_pcm)
                self._v2_latest_pcm = b""   # accumulator consumed
                if not getattr(self, "_v2_first_send_logged", False):
                    print("[V2 GATE] first frame shipped via held grant")
                    self._v2_first_send_logged = True
            else:
                # No grant outstanding — overwrite the slot. Older
                # unsent frames are dropped on the floor (intentional;
                # glitchbox's ``wsClient.current_frame = frame`` does
                # the same).
                self._v2_latest_frame = jpeg
        except Exception as e:
            print(f"[V2] send_frame error: {e}")

    def _v2_handle_grant(self):
        """Server granted the next frame. Ship the slot, or hold the grant.

        Equivalent of the legacy client's
        ``websocket_client.py:256-259`` handler that reacted to
        ``{"status":"send_frame"}`` by calling ``send_frame()``.
        """
        if not REALTIME_V2:
            return
        if not getattr(self, "ws_client_v2", None) or not self.ws_client_v2.connected:
            return
        # Track grant arrivals so we can see if the recv loop is reading
        # them. Count rather than print-per-grant to avoid log spam at
        # GPU throughput.
        self._v2_grant_count = getattr(self, "_v2_grant_count", 0) + 1
        if self._v2_grant_count in (1, 5, 20, 100):
            print(
                f"[V2 GATE] grant #{self._v2_grant_count} "
                f"(latest_frame={'set' if self._v2_latest_frame else 'None'})"
            )
        if self._v2_latest_frame is not None:
            jpeg = self._v2_latest_frame
            self._v2_latest_frame = None
            self._v2_grant_held = False
            try:
                self.ws_client_v2.send_frame(jpeg, self._v2_latest_pcm)
                self._v2_latest_pcm = b""   # accumulator consumed
            except Exception as e:
                print(f"[V2] send_frame (grant) error: {e}")
        else:
            # No frame in slot yet — the next ``_v2_handle_camera_frame``
            # will ship immediately and consume this grant.
            self._v2_grant_held = True

    def _v2_on_prompt_context(self, from_a, from_b, to_a, to_b):
        """Forward the live prompt-travel context (4 bilinear corners) to the
        projection-mapper modal's read-only display, if it's open."""
        pm = getattr(self.processed_display, "projection_mapper", None)
        if pm is not None and hasattr(pm, "update_prompt_context"):
            pm.update_prompt_context(from_a, from_b, to_a, to_b)

    def _v2_on_capabilities(self, caps):
        """V2 handshake-complete handler — kicks the ZMQ subscriber so
        rendered frames from the server's tcp://*:5555 publisher land in
        ProcessedDisplay.update_frame. Mirrors the parts of the legacy
        handle_settings() that aren't already covered by
        ControlPanel.apply_capabilities."""
        self.server_connected = True
        self._user_wants_connected = True
        self._refresh_connection_button()
        self.status_bar.update_processing_status(
            f"V2 connected: {self.server_host}:{self.server_port}"
        )
        self.reconnection_count = 0
        # Fresh-connection reset of the request/grant slot. Any stale
        # grant held over from a prior session would have caused the
        # first camera frame to ship without a grant from the new
        # session; clearing both halves of the slot keeps the gate
        # honest after each handshake.
        self._v2_latest_frame = None
        self._v2_grant_held = False
        # Kick ProcessedDisplay's ZMQ subscriber. The legacy StreamThread
        # spawned alongside will harmlessly 404 against the realtime
        # server's nonexistent /api/stream/{user_id} endpoint; only the
        # ZMQ subscriber matters for V2 output.
        print("[UI/V2] Capabilities received — starting ProcessedDisplay ZMQ subscriber")
        try:
            self.processed_display.stop_stream()
            QTimer.singleShot(100, self._start_fresh_stream)
        except Exception as e:
            print(f"[UI/V2] Error starting fresh stream: {e}")

    def _v2_handle_alpha(self, alpha: float, w_a: float, w_b: float, level: float = 0.0):
        """Render the audio meter (every frame, for clap responsiveness)
        and the α/blend readout (throttled ~1 Hz)."""
        # Live audio level meter — update every frame so transients show.
        self.status_bar.update_audio_level(level)
        # α / weights text — throttled so it doesn't thrash at 20-30 fps.
        self._v2_alpha_throttle += 1
        if self._v2_alpha_throttle < 20:
            return
        self._v2_alpha_throttle = 0
        self.status_bar.update_processing_status(
            f"streaming · α={alpha:.2f}  w_a={w_a:.2f}  w_b={w_b:.2f}"
        )
    # --------------------------------------------------------------------

    def handle_video_frame(self, frame):
        """Handle new frame from video"""
        from PySide6.QtGui import QImage, QPixmap
        from config import DISPLAY_WIDTH, DISPLAY_HEIGHT
        
        self.current_frame = frame
        
        # Convert numpy array to QPixmap (same as camera display)
        if frame is not None:
            height, width = frame.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image).scaled(
                DISPLAY_WIDTH, DISPLAY_HEIGHT, Qt.KeepAspectRatio)
            self.video_display.video_label.setPixmap(pixmap)

    def update_parameter(self, param_id: str, value):
        """Update parameter in WebSocket client"""
        self.ws_client.update_settings({param_id: value})

    def handle_fft_data(self, fft_data):
        """Handle FFT data from the FFT analyzer thread"""
        if fft_data and isinstance(fft_data, dict):
            # Update the acid_settings in the WebSocket client
            self.ws_client.update_settings({"acid_settings": fft_data})
            # Update the UI to show FFT is active
            if "binned_fft" in fft_data:
                bins = fft_data["binned_fft"]
                if isinstance(bins, list) and len(bins) > 0:
                    avg_energy = sum(bins) / len(bins)
                    self.status_bar.update_processing_status(f"FFT Audio Energy: {avg_energy:.2f}")

    def toggle_camera(self):
        """Start/Stop camera (local display only)"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()





    def process_frame(self):
        """Process current frame through WebSocket"""
        if self.current_frame is not None and not self.processing_frame:
            self.processing_frame = True
            # Store the current frame in the websocket client
            self.ws_client.current_frame = self.current_frame
            # Reset flag after setting the frame
            self.processing_frame = False

    def handle_connection_error(self, error_msg: str):
        """Handle connection errors (transient retry failures and terminal)."""
        self.server_connected = False
        self.status_bar.update_connection_status(False)

        # Terminal: the client exhausted its retries and its thread has
        # stopped. Return to the disconnected state so the toggle reads
        # "Connect". Every other error is transient — the client is still
        # retrying — so keep the toggle on "Disconnect".
        if error_msg == "Connection failed after maximum retries":
            self._on_connection_terminated("Connection failed - click Connect to retry")
            return

        self._refresh_connection_button()
        if self._user_wants_connected:
            self.status_bar.update_processing_status(f"Connection error (retrying): {error_msg}")
        elif self.camera_running:
            self.status_bar.update_processing_status("Camera running (not streaming - connection error)")
        else:
            self.status_bar.update_processing_status(f"Connection error: {error_msg}")

    def reconnect_to_server(self):
        """Reconnect = connect again.

        Kept as a named entry point for the curation-update flow
        (FORCE_MANUAL_RECONNECTION_AFTER_CURATION_UPDATE and the post-update
        connection-health check still call this). connect_to_server() stops
        any running client before starting, so reconnect and first-connect
        run the identical path — no bespoke teardown/recreate dance, and it
        targets the active protocol's client (V2 by default) rather than
        only the legacy one.
        """
        print("[UI] Reconnect requested")
        self.connect_to_server()

    def _restart_streaming_components(self):
        """Restart all streaming components for reconnection"""
        print("[UI] Restarting streaming components...")
        
        # The ProcessedDisplay will automatically restart its streaming when start_stream is called
        # This happens in handle_settings when the server connection is established
        # For now, just ensure it's ready to receive new streams
        
        # Restart frame timer if camera is running (create fresh timer instance)
        if self.camera_running:
            print("[UI] Recreating and starting frame timer for camera...")
            # Ensure old timer is completely stopped
            if hasattr(self, 'frame_timer'):
                self.frame_timer.stop()
                
            # Create a completely new timer instance to prevent duplication
            self.frame_timer = QTimer()
            self.frame_timer.timeout.connect(self.process_frame)
            self.frame_timer.setInterval(33)  # ~30 FPS
            self.frame_timer.start()
            print("[UI] Fresh frame timer created and started")
        
        print("[UI] Streaming components restart completed")

    def _stop_all_threads(self):
        """Stop all active threads cleanly"""
        print("[UI] Stopping all threads for reconnection...")
        
        # Stop frame timer
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()
        self.processing_frame = False
        
        # Reset state variables
        self.camera_running = False
        self.server_connected = False
        
        # Stop STT thread if running
        if hasattr(self, 'stt_thread') and self.stt_thread is not None:
            print("[UI] Stopping STT thread...")
            self.stt_thread.stop()
            if not self.stt_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating STT thread")
                self.stt_thread.terminate()
        
        # Stop FFT thread if running
        if hasattr(self, 'fft_thread') and self.fft_thread is not None:
            print("[UI] Stopping FFT thread...")
            self.fft_thread.stop()
            if not self.fft_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating FFT thread")
                self.fft_thread.terminate()
        
        # Stop video threads if running
        if hasattr(self, 'video_thread') and self.video_thread is not None:
            print("[UI] Stopping video thread...")
            self.video_thread.stop()
            if not self.video_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating video thread")
                self.video_thread.terminate()
                
        if hasattr(self, 'video_audio_thread') and self.video_audio_thread is not None:
            print("[UI] Stopping video audio thread...")
            self.video_audio_thread.stop()
            if not self.video_audio_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating video audio thread")
                self.video_audio_thread.terminate()
        
        # Stop camera thread
        if hasattr(self, 'camera_thread') and self.camera_thread is not None:
            print("[UI] Stopping camera thread...")
            self.camera_thread.stop()
            if not self.camera_thread.wait(2000):  # 2 second timeout
                print("[UI] Force terminating camera thread")
                self.camera_thread.terminate()
        
        # Stop WebSocket client
        if hasattr(self, 'ws_client') and self.ws_client is not None:
            print("[UI] Stopping WebSocket client...")
            # Force stop flags
            self.ws_client.running = False
            self.ws_client.processing = False
            
            # Stop the stream immediately
            if hasattr(self, 'processed_display'):
                self.processed_display.stop_stream()
            
            # Stop WebSocket thread
            self.ws_client.stop()
            if not self.ws_client.wait(3000):  # 3 second timeout
                print("[UI] Force terminating WebSocket thread")
                self.ws_client.terminate()
        
        # Recreate threads (except WebSocket which is recreated in reconnect_to_server)
        print("[UI] Recreating threads...")
        # Fully-wired recreate (incl. the V2 send gate) — see
        # _create_camera_thread. Wiring only handle_camera_frame here would
        # leave the local preview alive but stop feeding frames to the server.
        self._create_camera_thread()
        
        # Recreate STT thread (gated by STT_ENABLED)
        if STT_ENABLED:
            self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
            self.stt_thread.transcription_updated.connect(self.handle_transcription)
        else:
            self.stt_thread = None
        
        # Recreate FFT thread
        self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
        self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
        
        # Recreate video threads
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.handle_camera_frame)
        self.video_thread.video_finished.connect(self.handle_video_finished)
        
        self.video_audio_thread = VideoAudioThread()
        self.video_audio_thread.fft_data_updated.connect(self.handle_fft_data)
        self.video_audio_thread.audio_finished.connect(self.handle_video_finished)
        
        # Reset UI button states for disconnected state
        self.start_button.setText("Start Camera")
        self._user_wants_connected = False
        self._refresh_connection_button()
        self.stt_active = False
        if self.stt_button is not None:
            self.stt_button.setText("Start Speech Recognition")
        self.fft_active = False
        self.fft_button.setText("Start Audio FFT")
        
        print("[UI] All threads stopped and recreated successfully")

    def handle_transcription(self, text):
        """Handle transcribed text from STT and update the prompt"""
        if text and len(text.strip()) > 0:
            # Get the current client prompt prefix from WebSocket client parameters
            current_prefix = self.ws_client.params.get('client_prompt_prefix', "")
            
            # Prepend the client prompt prefix if it exists
            if current_prefix and len(current_prefix.strip()) > 0:
                full_prompt = f"{current_prefix} {text}"
                print(f"[UI] STT prompt with prefix: '{current_prefix}' + '{text}' = '{full_prompt}'")
            else:
                full_prompt = text
                print(f"[UI] STT prompt without prefix: '{text}'")
            
            # Update the prompt in the WebSocket client
            self.ws_client.update_prompt(full_prompt)
            # Update the prompt field in the control panel
            self.control_panel.update_control('prompt', full_prompt)
            # Update the UI to show the current prompt
            self.status_bar.update_processing_status(f"Prompt: {full_prompt}")

    def toggle_stt(self):
        """Toggle speech-to-text processing"""
        if not STT_ENABLED or self.stt_thread is None:
            print("[UI] STT is disabled — set GLITCHBOX_STT_ENABLED=1 to enable")
            self.status_bar.update_processing_status("Speech recognition disabled (GLITCHBOX_STT_ENABLED=0)")
            return
        if not self.stt_active:
            # Start STT
            self.stt_thread.start()
            self.stt_active = True
            self.stt_button.setText("Stop Speech Recognition")
            self.status_bar.update_processing_status("Speech recognition active")
        else:
            # Stop STT
            self.stt_thread.stop()
            self.stt_active = False
            self.stt_button.setText("Start Speech Recognition")
            self.status_bar.update_processing_status("Speech recognition stopped")

            # Recreate the STT thread for next use (QThread cannot be restarted)
            self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
            self.stt_thread.transcription_updated.connect(self.handle_transcription)

    def _ensure_soundlab_view(self):
        """Create the Sound Lab window on demand (hidden until toggled)."""
        if self.soundlab_view is None:
            from components.soundlab_view import SoundlabView
            self.soundlab_view = SoundlabView()
            # manual lever -> live knob (drives α when mode="manual";
            # the server auto-logs the human trace + policy prediction)
            if getattr(self, "ws_client_v2", None) is not None:
                self.soundlab_view.manual_changed.connect(
                    lambda v: self.ws_client_v2.update_knob(
                        "manual_alpha", v))
                # nodal editor patch -> THE deck driver (policy mode)
                self.soundlab_view.graph_applied.connect(
                    lambda gjson: self.ws_client_v2.update_knob(
                        "soundlab_graph", gjson))
                # Sound Lab is the audio control center: its generic
                # audio-group knobs flow like control-panel knobs
                self.soundlab_view.knob_changed.connect(
                    self.ws_client_v2.update_knob)
                self.soundlab_view.knob_changed.connect(
                    self._v2_remember_knob)
                # remember graph/lever too so reconnects restore them
                self.soundlab_view.graph_applied.connect(
                    lambda gjson: self._v2_remember_knob(
                        "soundlab_graph", gjson))
                self.soundlab_view.manual_changed.connect(
                    lambda v: self._v2_remember_knob("manual_alpha", v))
        return self.soundlab_view

    def _v2_handle_soundlab(self, sl: dict):
        """Per-frame soundlab telemetry → Sound Lab window (built lazily;
        accumulates history even while hidden)."""
        self._ensure_soundlab_view().update_telemetry(sl)

    def toggle_soundlab(self):
        """Show/hide the Sound Lab policy visualizer window."""
        view = self._ensure_soundlab_view()
        if view.isVisible():
            view.hide()
        else:
            view.show()
            view.raise_()

    def toggle_fft(self):
        """Toggle FFT audio analysis"""
        if not self.fft_active:
            # Start FFT
            if self.video_mode and self.camera_running:
                # In video mode, FFT is handled by video_audio_thread
                self.fft_active = True
                self.fft_button.setText("Stop Audio FFT")
                self.status_bar.update_processing_status("FFT audio analysis active (video mode)")
            else:
                # Start regular FFT thread
                self.fft_thread.start()
                self.fft_active = True
                self.fft_button.setText("Stop Audio FFT")
                self.status_bar.update_processing_status("FFT audio analysis active")
        else:
            # Stop FFT
            if self.video_mode and self.camera_running:
                # In video mode, FFT is handled by video_audio_thread
                self.fft_active = False
                self.fft_button.setText("Start Audio FFT")
                self.status_bar.update_processing_status("FFT audio analysis stopped")
            else:
                # Stop regular FFT thread
                self.fft_thread.stop()
                self.fft_active = False
                self.fft_button.setText("Start Audio FFT")
                self.status_bar.update_processing_status("FFT audio analysis stopped")
                
                # Recreate the FFT thread for next use (QThread cannot be restarted)
                self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
                self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
            
    def toggle_input_feed(self):
        """Toggle visibility of the input camera feed"""
        if self.camera_container.isVisible():
            self.camera_container.hide()
            self.toggle_input_button.setText("Show Input Feed")
            # Adjust the processed display to take full width
            self.processed_container.setMinimumWidth(self.width() - 40)
            # Resize window to be more compact
            new_width = max(self.width() // 2, 640)  # Don't go smaller than 640px
            self.resize(new_width, self.height())
        else:
            self.camera_container.show()
            self.toggle_input_button.setText("Hide Input Feed")
            # Reset the processed display width
            self.processed_container.setMinimumWidth(0)
            # Restore window width
            self.resize(self.width() * 2, self.height())
            
    def toggle_controls(self):
        """Toggle visibility of the controls panel"""
        if self.controls_container.isVisible():
            self.controls_container.hide()
            self.toggle_controls_button.setText("Show Controls")
            # Adjust window height
            new_height = self.height() - self.controls_container.height()
            self.resize(self.width(), new_height)
        else:
            self.controls_container.show()
            self.toggle_controls_button.setText("Hide Controls")
            # Restore window height
            new_height = self.height() + self.controls_container.height()
            self.resize(self.width(), new_height)
            
    def toggle_presentation_mode(self):
        """Toggle presentation mode (hide input feed and controls)"""
        if not self.presentation_mode:
            # Store original window size and state for restoration
            self.original_size = self.size()
            self.original_pos = self.pos()
            self.original_window_state = self.windowState()
            
            # Enter presentation mode
            self.camera_container.hide()
            self.controls_container.hide()
            self.toggle_input_button.setText("Show Input Feed")
            self.toggle_controls_button.setText("Show Controls")
            # Keep black frame button text as-is since it's independent of presentation mode
            self.toggle_presentation_button.setText("Exit Presentation Mode")
            
            # Hide status bar and connection label for cleaner look
            self.status_bar.hide()
            
            # Hide scroll bars in presentation mode
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            
            # Remove all margins and spacing from layouts
            self.feeds_layout.setContentsMargins(0, 0, 0, 0)
            self.feeds_layout.setSpacing(0)
            self.main_content_widget.layout().setContentsMargins(0, 0, 0, 0)
            self.main_content_widget.layout().setSpacing(0)
            
            # Make the processed display fill the entire window
            self.processed_container.setContentsMargins(0, 0, 0, 0)
            self.processed_display.setContentsMargins(0, 0, 0, 0)
            self.processed_label.hide()  # Hide the label in presentation mode
            
            # Go fullscreen
            self.showFullScreen()
            
        else:
            # Exit presentation mode
            self.camera_container.show()
            self.controls_container.show()
            self.toggle_input_button.setText("Hide Input Feed")
            self.toggle_controls_button.setText("Hide Controls")
            # Keep black frame button text as-is since it's independent of presentation mode
            self.toggle_presentation_button.setText("Enter Presentation Mode")
            
            # Show status bar and connection label
            self.status_bar.show()
            self.processed_label.show()
            
            # Restore scroll bars
            self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            
            # Restore original margins and spacing
            self.feeds_layout.setContentsMargins(9, 9, 9, 9)
            self.feeds_layout.setSpacing(6)
            self.main_content_widget.layout().setContentsMargins(9, 9, 9, 9)
            self.main_content_widget.layout().setSpacing(6)
            
            # Restore processed display margins
            self.processed_container.setContentsMargins(9, 9, 9, 9)
            self.processed_display.setContentsMargins(9, 9, 9, 9)
            
            # Restore original window state
            if hasattr(self, 'original_window_state'):
                if self.original_window_state == Qt.WindowFullScreen:
                    self.showFullScreen()
                else:
                    self.showNormal()
                    self.resize(self.original_size)
                    if hasattr(self, 'original_pos'):
                        self.move(self.original_pos)
            
        self.presentation_mode = not self.presentation_mode

    def toggle_fullscreen(self):
        """Toggle fullscreen mode for the processed display"""
        if hasattr(self, 'processed_display'):
            self.processed_display.toggle_fullscreen()
            if self.processed_display.is_fullscreen:
                self.toggle_fullscreen_button.setText("Close Output Window")
            else:
                self.toggle_fullscreen_button.setText("Detach Output")

    def toggle_black_frame(self):
        """Toggle black frame mode for the processed display"""
        self.black_frame_enabled = not self.black_frame_enabled
        
        # Update the processed display
        self.processed_display.set_black_frame_mode(self.black_frame_enabled)
        
        # Update button text
        if self.black_frame_enabled:
            self.toggle_black_frame_button.setText("Disable Black Frame")
            self.status_bar.update_processing_status("Black frame mode enabled")
        else:
            self.toggle_black_frame_button.setText("Enable Black Frame")
            self.status_bar.update_processing_status("Black frame mode disabled")

    def toggle_mirror(self):
        """Toggle mirror mode for the processed display"""
        if hasattr(self, 'processed_display'):
            self.processed_display.toggle_mirror()
            if self.processed_display.is_mirrored:
                self.toggle_mirror_button.setText("Disable Mirror")
            else:
                self.toggle_mirror_button.setText("Enable Mirror")

    def scroll_to_top(self):
        """Scroll to the top of the content"""
        self.scroll_area.verticalScrollBar().setValue(0)
        
    def scroll_to_bottom(self):
        """Scroll to the bottom of the content"""
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
        
    def scroll_to_controls(self):
        """Scroll to the controls section"""
        if hasattr(self, 'controls_container'):
            # Get the position of the controls container
            controls_pos = self.controls_container.mapTo(self.main_content_widget, self.controls_container.rect().topLeft())
            # Scroll to show the controls
            self.scroll_area.ensureVisible(controls_pos.x(), controls_pos.y(), 0, 0)
            
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for scrolling"""
        # Handle common scroll shortcuts that work on both Windows and Linux
        if event.key() == Qt.Key_Home:
            self.scroll_to_top()
        elif event.key() == Qt.Key_End:
            self.scroll_to_bottom()
        elif event.key() == Qt.Key_PageUp:
            # Scroll up by one page
            current_value = self.scroll_area.verticalScrollBar().value()
            page_step = self.scroll_area.verticalScrollBar().pageStep()
            self.scroll_area.verticalScrollBar().setValue(current_value - page_step)
        elif event.key() == Qt.Key_PageDown:
            # Scroll down by one page
            current_value = self.scroll_area.verticalScrollBar().value()
            page_step = self.scroll_area.verticalScrollBar().pageStep()
            self.scroll_area.verticalScrollBar().setValue(current_value + page_step)
        else:
            # Pass other key events to the parent
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Handle window close event - cleanup all threads in proper order"""
        print("[UI] Closing window - cleaning up...")
        
        # Accept the close event immediately to prevent freezing
        event.accept()
        
        try:
            # Stop all active processes immediately
            self.frame_timer.stop()
            
            # Stop automatic curation update timer
            if hasattr(self, 'curation_auto_timer'):
                self.curation_auto_timer.stop()
            
            # Reset state variables
            self.camera_running = False
            self.server_connected = False
            self.processing_frame = False
            
            # Clear displays immediately
            if hasattr(self, 'camera_display'):
                self.camera_display.clear_display()
            if hasattr(self, 'video_display'):
                print("[UI] Cleaning up video display...")
                self.video_display.cleanup()
            if hasattr(self, 'processed_display'):
                print("[UI] Cleaning up processed display...")
                # Clear ZMQ queue first to prevent blocking
                self.processed_display.clear_zmq_queue()
                self.processed_display.cleanup()
            
            # Force terminate all threads immediately to prevent core dumps
            threads_to_terminate = []
            
            if hasattr(self, 'ws_client') and self.ws_client is not None:
                self.ws_client.running = False
                self.ws_client.processing = False
                self.ws_client.close()
                threads_to_terminate.append(('WebSocket', self.ws_client))

            # V2 realtime client + its raw-PCM audio thread (these used to
            # leak on close — only the legacy client was being stopped).
            if hasattr(self, 'ws_client_v2') and self.ws_client_v2 is not None:
                self.ws_client_v2.close()
                threads_to_terminate.append(('WSClientV2', self.ws_client_v2))

            if hasattr(self, 'audio_thread') and self.audio_thread is not None:
                self.audio_thread.stop()
                threads_to_terminate.append(('Audio', self.audio_thread))

            if hasattr(self, 'stt_thread') and self.stt_thread is not None:
                self.stt_thread.stop()
                threads_to_terminate.append(('STT', self.stt_thread))
            
            if hasattr(self, 'fft_thread') and self.fft_thread is not None:
                self.fft_thread.stop()
                threads_to_terminate.append(('FFT', self.fft_thread))
                
            if hasattr(self, 'video_thread') and self.video_thread is not None:
                self.video_thread.stop()
                threads_to_terminate.append(('Video', self.video_thread))
                
            if hasattr(self, 'video_audio_thread') and self.video_audio_thread is not None:
                self.video_audio_thread.stop()
                threads_to_terminate.append(('VideoAudio', self.video_audio_thread))
                
            if hasattr(self, 'camera_thread') and self.camera_thread is not None:
                self.camera_thread.stop()
                threads_to_terminate.append(('Camera', self.camera_thread))
            
            # Give threads a brief moment to stop gracefully, then force terminate
            def force_terminate_remaining():
                for name, thread in threads_to_terminate:
                    if thread and thread.isRunning():
                        print(f"[UI] Force terminating {name} thread...")
                        thread.terminate()
                
                # Final cleanup after all threads are terminated
                def final_cleanup():
                    print("[UI] All threads terminated - final cleanup...")
                    QApplication.quit()
                
                QTimer.singleShot(200, final_cleanup)  # Final cleanup after 200ms
            
            QTimer.singleShot(300, force_terminate_remaining)  # Force terminate after 300ms
            
            print("[UI] Cleanup signals sent - threads will be force terminated if needed")
            
        except Exception as e:
            print(f"[UI] Error during cleanup: {e}")
            # Force immediate exit on error
            def emergency_exit():
                import os
                print("[UI] Emergency exit due to cleanup error")
                try:
                    os._exit(1)
                except:
                    import sys
                    sys.exit(1)
            
            QTimer.singleShot(100, emergency_exit)
            
        # Use QTimer to force termination if app doesn't quit quickly
        def force_exit():
            import os
            import sys
            print("[UI] Force terminating application")
            try:
                os._exit(0)
            except:
                sys.exit(0)
        
        QTimer.singleShot(2000, force_exit)  # Force exit after 2 seconds

    def update_curation_index(self):
        """Update the curation index on the server (manual button press)"""
        index = self.curation_spinbox.value()
        self._perform_curation_index_update(index, is_automatic=False)

    def _perform_curation_index_update(self, index, is_automatic=False):
        """Internal method to perform curation index update"""
        # Enable black frame mode before updating curation index
        if not self.black_frame_enabled:
            print(f"[UI] Enabling black frame mode for curation index update")
            self.black_frame_enabled = True
            self.processed_display.set_black_frame_mode(True)
            self.toggle_black_frame_button.setText("Disable Black Frame")
        
        # Clear ZMQ queue and display black frame immediately
        print(f"[UI] Clearing ZMQ queue and displaying black frame for curation index update")
        self.processed_display.clear_zmq_queue()
        
        # Handle button state only for manual updates
        if not is_automatic:
            # Disable the button during update
            self.curation_update_button.setEnabled(False)
            self.curation_update_button.setText("Updating...")
        
        update_type = "Automatic" if is_automatic else "Manual"
        self.status_bar.update_processing_status(f"{update_type} curation index update to {index}...")
        print(f"[UI] {update_type.lower()} curation index update: {index}")

        
        # Use QTimer to perform the update without blocking the UI
        def perform_update():
            try:
                # This will be called in the main thread but we'll use the WebSocket client's
                # built-in async handling which should be non-blocking
                print(f"[UI] Sending curation index update request for index {index}")
                
                # Store the update parameters for the success/failure callbacks
                self._pending_curation_index = index
                self._pending_curation_is_automatic = is_automatic
                
                # Call the WebSocket client's update method (this should be non-blocking)
                # We'll assume the WebSocket client handles this asynchronously
                import asyncio
                import threading
                
                def run_async_update():
                    success = False
                    message = "Unknown error"
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success, message = loop.run_until_complete(
                            self.ws_client.update_curation_index(index)
                        )
                        loop.close()
                        print(f"[UI] Async update completed: success={success}, message={message}")
                        
                    except Exception as e:
                        success = False
                        message = str(e)
                        print(f"[UI] Async update failed with exception: {e}")
                    finally:
                        # Use Qt signal to communicate back to main thread (this works from any thread)
                        print(f"[UI] Emitting signal with success={success}, message={message}")
                        self.curation_signal_handler.update_completed.emit(success, message)
                
                # Run the async operation in a separate thread
                thread = threading.Thread(target=run_async_update)
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                print(f"[UI] Error in perform_update: {e}")
                # Use signal for error case too
                self.curation_signal_handler.update_completed.emit(False, str(e))
        
        # Use QTimer with 0 delay to run the update in the next event loop iteration
        QTimer.singleShot(50, perform_update)  # Small delay to ensure UI updates are processed
    
    def _handle_curation_update_result(self, success, message):
        """Handle the result of curation index update (called from main thread)"""
        try:
            print(f"[UI] Handling curation update result: success={success}, message={message}")
            index = getattr(self, '_pending_curation_index', 'unknown')
            is_automatic = getattr(self, '_pending_curation_is_automatic', False)
            
            update_type = "Automatic" if is_automatic else "Manual"
            
            if success:
                self.status_bar.update_processing_status(f"{update_type} curation index updated to {index}: {message}")
                print(f"[UI] Successfully updated curation index to {index} ({update_type.lower()})")
                
                # Automatically disable black frame mode on successful update (if configured)
                if self.black_frame_enabled and self.auto_disable_black_frame_after_curation_update:
                    print(f"[UI] Automatically disabling black frame mode after successful curation index update")
                    print(f"[UI] Waiting {self.black_frame_disable_timeout} seconds...")
                    time.sleep(self.black_frame_disable_timeout)
                    self.black_frame_enabled = False
                    self.processed_display.set_black_frame_mode(False)
                    self.toggle_black_frame_button.setText("Enable Black Frame")
            else:
                self.status_bar.update_processing_status(f"Failed to update curation index ({update_type.lower()}): {message}")
                print(f"[UI] Failed to update curation index ({update_type.lower()}): {message}")
                
        except Exception as e:
            self.status_bar.update_processing_status(f"Error updating curation index: {str(e)}")
            print(f"[UI] Error in _handle_curation_update_result: {e}")
        finally:
            # Re-enable the button only for manual updates
            try:
                if not getattr(self, '_pending_curation_is_automatic', False):
                    print(f"[UI] Re-enabling curation update button")
                    self.curation_update_button.setEnabled(True)
                    self.curation_update_button.setText("Update")
                
                # Clean up the pending data
                if hasattr(self, '_pending_curation_index'):
                    delattr(self, '_pending_curation_index')
                if hasattr(self, '_pending_curation_is_automatic'):
                    delattr(self, '_pending_curation_is_automatic')
                print(f"[UI] Cleaned up pending curation data")
                
                # Check connection state after curation update and trigger reconnection if needed
                if self.force_manual_reconnection_after_curation_update:
                    print(f"[UI] Force Reconnecting after curation update")
                    # Add longer delay to allow complete cleanup before reconnection
                    QTimer.singleShot(2000, self.reconnect_to_server)  # 2 second delay
                else:
                    self._check_connection_after_curation_update()
                
            except Exception as e:
                print(f"[UI] Error cleaning up: {e}")

    def perform_automatic_curation_update(self):
        """Perform automatic curation index update (called by timer)"""
        print("[UI] Automatic curation update timer triggered")
        
        if not CURATION_INDEX_AUTO_UPDATE:
            print("[UI] Skipping automatic curation update - disabled in config")
            return
            
        if not self.server_connected:
            print("[UI] Skipping automatic curation update - not connected to server")
            return
            
        print("[UI] Performing automatic curation index update...")
        
        # Get current curation index and increment it
        current_index = self.curation_spinbox.value()
        
        # Calculate next index (cycle through available indices)
        next_index = (current_index + 1) % (CURATION_INDEX_MAX + 1)
        
        # Update the spinbox value
        self.curation_spinbox.setValue(next_index)
        
        # Validate that the value is within range
        if next_index > CURATION_INDEX_MAX:
            print(f"[UI] Warning: Curation index {next_index} exceeds maximum {CURATION_INDEX_MAX}, clamping to maximum")
            next_index = CURATION_INDEX_MAX
            self.curation_spinbox.setValue(next_index)
        
        # Simulate the manual update process
        print(f"[UI] Automatic curation update: {current_index} -> {next_index}")
        self.status_bar.update_processing_status(f"Automatic curation update: {current_index} -> {next_index}")
        
        # Perform the actual update (similar to manual update but without button interference)
        self._perform_curation_index_update(next_index, is_automatic=True)

    def validate_curation_index(self, value):
        """Validate that the curation index is within the configured range"""
        if value > CURATION_INDEX_MAX:
            print(f"[UI] Warning: Manual curation index {value} exceeds maximum {CURATION_INDEX_MAX}, clamping to maximum")
            self.curation_spinbox.setValue(CURATION_INDEX_MAX)
        elif value < 0:
            print(f"[UI] Warning: Manual curation index {value} is negative, setting to 0")
            self.curation_spinbox.setValue(0)

    def toggle_automatic_curation_updates(self):
        """Toggle automatic curation index updates"""
        if self.curation_auto_timer.isActive():
            # Stop automatic updates
            self.curation_auto_timer.stop()
            self.toggle_auto_curation_button.setText("Start Auto Curation Updates")
            self.status_bar.update_processing_status("Automatic curation updates stopped")
            print("[UI] Automatic curation updates stopped")
        else:
            # Start automatic updates (only if connected to server)
            if self.server_connected:
                self.curation_auto_timer.start()
                self.toggle_auto_curation_button.setText("Stop Auto Curation Updates")
                self.status_bar.update_processing_status(f"Automatic curation updates started (every {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
                print(f"[UI] Automatic curation updates started (interval: {CURATION_INDEX_UPDATE_TIME} seconds, range 0-{CURATION_INDEX_MAX})")
            else:
                self.status_bar.update_processing_status("Cannot start automatic updates - not connected to server")
                print("[UI] Cannot start automatic updates - not connected to server")

    def _create_camera_thread(self):
        """Create and fully wire a fresh CameraThread (single source of truth).

        Every path that (re)creates the camera thread — initial setup, the
        reconnect teardown, and update_camera_index() — must route through
        here so the SAME signals are wired each time.

        Critically, under REALTIME_V2 the camera frame is the send gate's
        dispatch trigger (_v2_handle_camera_frame); a recreation path that
        wires only handle_camera_frame leaves the local preview working but
        silently stops feeding frames to the server, which presents as a dead
        connection after a reconnect or a live camera-index change.
        """
        self.camera_thread = CameraThread()
        self.camera_thread.device_index = self.camera_device_index
        self.camera_thread.frame_ready.connect(self.handle_camera_frame)
        if REALTIME_V2:
            self.camera_thread.frame_ready.connect(self._v2_handle_camera_frame)
        return self.camera_thread

    def update_camera_index(self):
        """Update the camera device index"""
        new_index = self.camera_spinbox.value()
        
        if new_index == self.camera_device_index:
            self.status_bar.update_processing_status(f"Camera already using index {new_index}")
            return
            
        print(f"[UI] Updating camera index from {self.camera_device_index} to {new_index}")
        self.camera_update_button.setEnabled(False)
        self.camera_update_button.setText("Updating...")
        self.status_bar.update_processing_status(f"Updating camera to index {new_index}...")
        
        # Store the old state
        was_camera_running = self.camera_running
        
        try:
            # Stop camera if it's running. Deliberately NOT calling
            # self.stop_camera() here: its cleanup is deferred via
            # QTimer.singleShot(50, perform_camera_cleanup), which closes
            # over `self.camera_thread` by reference, not by value. This
            # method synchronously stops the OLD thread, recreates, and
            # restarts a NEW one just below — reliably taking well over
            # 50ms (V4L2 device open + the wait(1000) call) — so that
            # deferred callback used to fire AFTER self.camera_thread
            # already pointed at the brand-new thread, calling
            # .stop()/.terminate() on it moments after it started.
            # Terminating a QThread while it's inside cv2.VideoCapture's
            # blocking open() call is what produced a
            # "FATAL: exception not rethrown" crash. Only the immediate
            # UI-state reset is needed here; the rest of this method
            # already does its own correct synchronous stop/recreate/
            # restart of camera_thread.
            #
            # Also deliberately NOT calling self.processed_display.
            # clear_display() here — it calls stop_stream() under the
            # hood, tearing down the ZMQ subscriber that receives
            # rendered output from the server. That subscriber is only
            # ever restarted by a fresh websocket handshake
            # (_v2_on_capabilities -> _start_fresh_stream), which a
            # camera-index change does NOT trigger — the server-side
            # RealtimeSession and its output stream are completely
            # independent of which local camera is capturing input.
            # Clearing it here silently kills the output display until
            # the user manually disconnects/reconnects.
            if was_camera_running:
                self.frame_timer.stop()
                self.processing_frame = False
                self.camera_running = False
                self.camera_display.clear_display()

            # Update the index
            self.camera_device_index = new_index
            
            # Recreate camera thread with new index
            if hasattr(self, 'camera_thread'):
                self.camera_thread.stop()
                self.camera_thread.wait(1000)
                if self.camera_thread.isRunning():
                    self.camera_thread.terminate()
            
            # Recreate a FULLY-wired thread (this also re-attaches the V2
            # send gate — see _create_camera_thread).
            self._create_camera_thread()

            # Restart camera if it was running
            if was_camera_running:
                self.start_camera()
                self.status_bar.update_processing_status(f"Camera updated to index {new_index} and restarted")
            else:
                self.status_bar.update_processing_status(f"Camera updated to index {new_index}")
                
            print(f"[UI] Successfully updated camera to index {new_index}")
            
        except Exception as e:
            print(f"[UI] Error updating camera index: {e}")
            self.status_bar.update_processing_status(f"Error updating camera: {e}")
        finally:
            self.camera_update_button.setEnabled(True)
            self.camera_update_button.setText("Update Camera")

    def update_mic_index(self):
        """Update the microphone device index"""
        new_index = self.mic_spinbox.value()
        # -1 in the spinbox means "system default input device" (None).
        if new_index < 0:
            new_index = None

        if new_index == self.audio_device_index:
            self.status_bar.update_processing_status(f"Microphone already using index {new_index}")
            return
            
        print(f"[UI] Updating microphone index from {self.audio_device_index} to {new_index}")
        self.mic_update_button.setEnabled(False)
        self.mic_update_button.setText("Updating...")
        self.status_bar.update_processing_status(f"Updating microphone to index {new_index}...")
        
        # Store the old states
        was_stt_running = self.stt_active
        was_fft_running = self.fft_active
        
        try:
            # Stop audio threads if they're running
            if was_stt_running:
                self.toggle_stt()  # This will stop and recreate the thread
            if was_fft_running:
                self.toggle_fft()  # This will stop and recreate the thread
            
            # Update the index
            self.audio_device_index = new_index
            
            # Recreate audio threads with new index (STT gated by STT_ENABLED)
            if STT_ENABLED:
                self.stt_thread = SpeechToTextThread(input_device_index=self.audio_device_index)
                self.stt_thread.transcription_updated.connect(self.handle_transcription)
            else:
                self.stt_thread = None

            self.fft_thread = FFTAnalyzerThread(input_device_index=self.audio_device_index)
            self.fft_thread.fft_data_updated.connect(self.handle_fft_data)
            
            # Restart audio threads if they were running
            if was_stt_running:
                self.toggle_stt()  # This will start the new thread
            if was_fft_running:
                self.toggle_fft()  # This will start the new thread
                
            if was_stt_running or was_fft_running:
                self.status_bar.update_processing_status(f"Microphone updated to index {new_index} and audio processing restarted")
            else:
                self.status_bar.update_processing_status(f"Microphone updated to index {new_index}")
                
            print(f"[UI] Successfully updated microphone to index {new_index}")
            
        except Exception as e:
            print(f"[UI] Error updating microphone index: {e}")
            self.status_bar.update_processing_status(f"Error updating microphone: {e}")
        finally:
            self.mic_update_button.setEnabled(True)
            self.mic_update_button.setText("Update Microphone")

    def refresh_device_lists(self):
        """Refresh the lists of available cameras and microphones"""
        print("[UI] Refreshing device lists...")
        self.refresh_devices_button.setEnabled(False)
        self.refresh_devices_button.setText("Refreshing...")
        
        try:
            self.available_cameras = detect_cameras()
            self.available_microphones = detect_microphones()
            
            print(f"[UI] Detected cameras: {[(idx, name, info) for idx, name, info in self.available_cameras]}")
            print(f"[UI] Detected microphones: {[f'{idx}: {name}' for idx, name in self.available_microphones]}")
            
            # Update camera spinbox tooltip
            if self.available_cameras:
                camera_list = [f"{idx}: {name} ({info})" for idx, name, info in self.available_cameras]
                self.camera_spinbox.setToolTip("Available cameras:\n" + "\n".join(camera_list))
            else:
                self.camera_spinbox.setToolTip("No cameras detected, but you can still try different indices")
            
            # Update microphone spinbox tooltip
            if self.available_microphones:
                mic_list = [f"{idx}: {name[:30]}..." if len(name) > 30 else f"{idx}: {name}" 
                           for idx, name in self.available_microphones]
                self.mic_spinbox.setToolTip("Available microphones:\n" + "\n".join(mic_list))
            else:
                self.mic_spinbox.setToolTip("No microphones detected, but you can still try different indices")
            
            self.status_bar.update_processing_status(f"Device refresh complete: {len(self.available_cameras)} cameras, {len(self.available_microphones)} microphones")
            print("[UI] Device lists refreshed successfully")
            
        except Exception as e:
            print(f"[UI] Error refreshing device lists: {e}")
            self.status_bar.update_processing_status(f"Error refreshing devices: {e}")
        finally:
            self.refresh_devices_button.setEnabled(True)
            self.refresh_devices_button.setText("Refresh Device Lists")

    def _perform_aggressive_cleanup_before_reconnection(self):
        """Perform thorough cleanup before reconnection to prevent resource accumulation"""
        print("[UI] Performing aggressive cleanup before reconnection...")
        
        try:
            # CRITICAL: Stop frame timer to prevent duplication
            if hasattr(self, 'frame_timer'):
                self.frame_timer.stop()
                self.processing_frame = False
                
            # Reset FPS counter in aggressive cleanup
            if hasattr(self, 'status_bar') and hasattr(self.status_bar, 'frame_times'):
                self.status_bar.frame_times = []
            
            # Stop and forcefully cleanup processed display with longer timeout
            if hasattr(self, 'processed_display'):
                self.processed_display.clear_zmq_queue()
                self.processed_display.stop_stream()
                
                # Force cleanup of ZMQ and Stream threads
                if hasattr(self.processed_display, 'zmq_thread') and self.processed_display.zmq_thread:
                    self.processed_display.zmq_thread.running = False
                    if self.processed_display.zmq_thread.isRunning():
                        self.processed_display.zmq_thread.terminate()
                        self.processed_display.zmq_thread.wait(1000)
                    self.processed_display.zmq_thread = None
                    
                if hasattr(self.processed_display, 'stream_thread') and self.processed_display.stream_thread:
                    self.processed_display.stream_thread.running = False
                    if self.processed_display.stream_thread.isRunning():
                        self.processed_display.stream_thread.terminate()
                        self.processed_display.stream_thread.wait(1000)
                    self.processed_display.stream_thread = None
                
            # Ensure WebSocket is completely dead before proceeding
            if hasattr(self, 'ws_client') and self.ws_client is not None:
                # CRITICAL: Disconnect all signals to prevent duplication
                if hasattr(self, 'signal_connections_active') and self.signal_connections_active:
                    try:
                        self.ws_client.frame_received.disconnect()
                        self.ws_client.connection_error.disconnect()
                        self.ws_client.settings_received.disconnect()
                        self.ws_client.status_changed.disconnect()
                    except Exception as e:
                        print(f"[UI] Error disconnecting signals (non-critical): {e}")
                    self.signal_connections_active = False
                
                self.ws_client.running = False
                self.ws_client.processing = False
                self.ws_client.connection_successful = False
                
                # Force terminate WebSocket thread if it's still running
                if self.ws_client.isRunning():
                    self.ws_client.terminate()
                    self.ws_client.wait(1000)  # Wait up to 1 second
                    
            # Force garbage collection to clean up any lingering objects
            import gc
            gc.collect()
                    
            print("[UI] Aggressive cleanup completed")
            
        except Exception as e:
            print(f"[UI] Error during aggressive cleanup: {e}")

    def _check_connection_after_curation_update(self):
        """Check connection state after curation update and trigger reconnection if needed"""
        def check_connection():
            try:
                print("[UI] Checking connection state after curation update...")
                
                # Check multiple indicators of connection state
                server_connected = self.server_connected
                ws_connection_successful = hasattr(self.ws_client, 'connection_successful') and self.ws_client.connection_successful
                ws_running = hasattr(self.ws_client, 'running') and self.ws_client.running
                
                print(f"[UI] Connection state - server_connected: {server_connected}, ws_connection_successful: {ws_connection_successful}, ws_running: {ws_running}")
                
                # If any connection indicator shows disconnected state, trigger reconnection
                if not server_connected or not ws_connection_successful or not ws_running:
                    # Check if we've exceeded maximum reconnection attempts
                    if self.reconnection_count >= self.max_reconnection_attempts:
                        print(f"[UI] Maximum reconnection attempts ({self.max_reconnection_attempts}) reached - stopping automatic reconnection")
                        self.status_bar.update_processing_status(f"Connection lost - maximum reconnection attempts ({self.max_reconnection_attempts}) reached")
                        return
                        
                    print(f"[UI] Connection appears to be disconnected after curation update - triggering automatic reconnection (attempt {self.reconnection_count + 1}/{self.max_reconnection_attempts})")
                    self.status_bar.update_processing_status(f"Connection lost after curation update - automatically reconnecting (attempt {self.reconnection_count + 1}/{self.max_reconnection_attempts})...")
                    
                    self.reconnection_count += 1
                    
                    # Use QTimer to call reconnect_to_server in the next event loop iteration
                    # This ensures we don't interfere with any ongoing cleanup
                    QTimer.singleShot(100, self.reconnect_to_server)
                else:
                    print("[UI] Connection state appears healthy after curation update")
                    # Reset reconnection count on successful connection
                    self.reconnection_count = 0
                    
            except Exception as e:
                print(f"[UI] Error checking connection state after curation update: {e}")
        
        # Use QTimer to delay the check longer, allowing the connection state to fully stabilize
        QTimer.singleShot(1500, check_connection)  # 1.5 second delay

    def _start_fresh_stream(self):
        """Start a fresh stream with proper cleanup to prevent duplication"""
        try:
            if hasattr(self, 'ws_client') and self.ws_client and hasattr(self, 'processed_display'):
                self.processed_display.start_stream(self.ws_client.user_id, self.server_http_uri)
                print("[UI] Fresh ProcessedDisplay streaming started successfully")
            else:
                print("[UI] Cannot start fresh stream - WebSocket client not available")
        except Exception as e:
            print(f"[UI] Error starting fresh stream: {e}")

    def force_terminate(self):
        """Force terminate the application with proper cleanup"""
        print("[UI] Force terminating application...")
        try:
            # First, perform proper cleanup
            self._perform_immediate_cleanup()
            
            # Give cleanup a moment to complete
            import time
            time.sleep(0.5)
            
            # Then close the application gracefully
            self.close()  # This will trigger closeEvent
            
        except Exception as e:
            print(f"[UI] Error during cleanup, forcing exit: {e}")
            # If cleanup fails, use force termination as last resort
            try:
                pid = os.getpid()
                os.kill(pid, 9)  # SIGKILL signal
            except:
                sys.exit(1)
                
    def _perform_immediate_cleanup(self):
        """Perform immediate cleanup before termination"""
        print("[UI] Performing immediate cleanup...")
        
        # Stop all active processes
        if hasattr(self, 'frame_timer'):
            self.frame_timer.stop()
            
        # Stop camera/video
        if self.camera_running:
            self.stop_camera()
            
        # Clean up video display specifically
        if hasattr(self, 'video_display'):
            print("[UI] Force cleaning up video display...")
            self.video_display.cleanup(force=True)
            
        # Stop other threads
        if hasattr(self, 'stt_thread') and self.stt_thread is not None:
            self.stt_thread.stop()
            
        if hasattr(self, 'fft_thread') and self.fft_thread is not None:
            self.fft_thread.stop()
            
        # Close WebSocket connection
        if hasattr(self, 'ws_client') and self.ws_client is not None:
            self.ws_client.close()

    def toggle_video_mode(self, state):
        """Toggle between camera/microphone and video input modes"""
        self.video_mode = self.video_mode_checkbox.isChecked()  # Use the checkbox's isChecked method
        
        if self.video_mode:
            # Enable video controls
            self.video_file_button.setEnabled(True)
            self.video_loop_checkbox.setEnabled(True)
            
            # Disable camera/microphone controls
            self.camera_spinbox.setEnabled(False)
            self.camera_update_button.setEnabled(False)
            self.mic_spinbox.setEnabled(False)
            self.mic_update_button.setEnabled(False)
            
            # Switch display components
            self.camera_display.setVisible(False)
            self.video_display.setVisible(True)
            
            # Update labels and button text
            self.camera_label.setText("Video Input")
            if not self.camera_running:
                self.start_button.setText("Start Video")
            self.status_bar.update_processing_status("Video mode enabled - select a video file")
            
            # Stop camera if running
            if self.camera_running:
                self.stop_camera()
                
        else:
            # Disable video controls
            self.video_file_button.setEnabled(False)
            self.video_loop_checkbox.setEnabled(False)
            
            # Enable camera/microphone controls
            self.camera_spinbox.setEnabled(True)
            self.camera_update_button.setEnabled(True)
            self.mic_spinbox.setEnabled(True)
            self.mic_update_button.setEnabled(True)
            
            # Switch display components
            self.video_display.setVisible(False)
            self.camera_display.setVisible(True)
            
            # Update labels and button text
            self.camera_label.setText("Input Camera")
            if not self.camera_running:
                self.start_button.setText("Start Camera")
            self.status_bar.update_processing_status("Camera mode enabled")
            
            # Stop video if running
            if self.camera_running:
                self.stop_camera()

    def select_video_file(self):
        """Open file dialog to select a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_path_label.setText(os.path.basename(file_path))
            self.video_path_label.setStyleSheet("color: black; font-style: normal;")
            self.video_path_label.setToolTip(file_path)
            
            # Update video display component
            self.video_display.set_video_path(file_path)
            self.video_display.set_loop_enabled(self.video_loop_checkbox.isChecked())
            
            # Update old video threads (keep for compatibility if needed)
            self.video_thread.set_video_path(file_path)
            self.video_thread.set_audio_enabled(True)  # Enable audio in video thread
            self.video_audio_thread.set_video_path(file_path)
            
            # Set loop settings
            loop_enabled = self.video_loop_checkbox.isChecked()
            self.video_thread.set_loop(loop_enabled)
            self.video_audio_thread.set_loop(loop_enabled)
            
            # Get video info
            video_info = self.video_thread.get_video_info()
            if video_info:
                info_text = f"{video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps ({video_info['duration']:.1f}s)"
                self.status_bar.update_processing_status(f"Video loaded: {info_text}")
            else:
                self.status_bar.update_processing_status("Video loaded but could not get info")

    def handle_video_finished(self):
        """Handle video playback completion"""
        if not self.video_loop_checkbox.isChecked():
            self.status_bar.update_processing_status("Video playback finished")
            if self.camera_running:
                self.stop_camera()



    def start_camera(self):
        """Start camera/video and local display"""
        if self.video_mode and self.video_path:
            # Start video mode using video display component
            if self.video_display.start_playback():
                self.frame_timer.start()
                self.camera_running = True
                self.start_button.setText("Stop Video")
                
                # Automatically enable FFT in video mode
                if not self.fft_active:
                    self.fft_active = True
                    self.fft_button.setText("Stop Audio FFT")
                    self.status_bar.update_processing_status("Video running with FFT - streaming to server")
                else:
                    self.status_bar.update_processing_status("Video running - streaming to server")
                
                # Update status based on connection state
                if self.server_connected:
                    # Start server streaming automatically
                    self.ws_client.start_camera()
                    if self.fft_active:
                        self.status_bar.update_processing_status("Video running with FFT - streaming to server")
                    else:
                        self.status_bar.update_processing_status("Video running - streaming to server")
                else:
                    if self.fft_active:
                        self.status_bar.update_processing_status("Video running with FFT (not streaming - disconnected)")
                    else:
                        self.status_bar.update_processing_status("Video running (not streaming - disconnected)")
            else:
                self.status_bar.update_processing_status("Failed to start video playback")
        else:
            # Start camera mode (original behavior)
            self.camera_thread.start()
            # V2 requires the raw-PCM AudioThread running in parallel —
            # the camera frame is the dispatch trigger but each frame
            # carries the most-recent PCM chunk as its audio tail. If
            # AudioThread isn't started, _v2_latest_pcm stays b"" and
            # every frame ships silent audio.
            if REALTIME_V2:
                self.audio_thread.start()
            self.frame_timer.start()
            self.camera_running = True
            self.start_button.setText("Stop Camera")
            
            # Update status based on connection state
            if self.server_connected:
                # Start server streaming automatically
                self.ws_client.start_camera()
                self.status_bar.update_processing_status("Camera running - streaming to server")
            else:
                self.status_bar.update_processing_status("Camera running (not streaming - disconnected)")

    def stop_camera(self):
        """Stop camera/video and processing"""
        print("[UI] Stopping camera/video")
        # Stop frame timer first
        self.frame_timer.stop()
        self.processing_frame = False
        
        # Update state immediately
        self.camera_running = False
        if self.video_mode:
            self.start_button.setText("Start Video")
        else:
            self.start_button.setText("Start Camera")
        
        # Clear displays immediately
        self.camera_display.clear_display()
        self.processed_display.clear_display()
        
        # Use async cleanup to avoid blocking UI
        def perform_camera_cleanup():
            try:
                print("[UI] Starting camera/video cleanup...")
                
                if self.video_mode:
                    # Stop video display component
                    self.video_display.stop_playback()
                    # Reset FFT state in video mode
                    if self.fft_active:
                        self.fft_active = False
                        self.fft_button.setText("Start Audio FFT")
                else:
                    # Stop camera thread
                    self.camera_thread.stop()
                    # V2: also stop the parallel raw-PCM audio thread.
                    if REALTIME_V2:
                        self.audio_thread.stop()
                        self._v2_latest_pcm = b""
                
                # Stop server streaming if connected (non-blocking)
                if self.server_connected:
                    try:
                        self.ws_client.stop_camera()
                    except Exception as e:
                        print(f"[UI] Error stopping server streaming: {e}")
                
                print("[UI] Camera/video cleanup completed")
                
            except Exception as e:
                print(f"[UI] Error during camera/video cleanup: {e}")
            finally:
                # Update status based on connection state
                if self.server_connected:
                    self.status_bar.update_processing_status(f"Connected to server: {self.server_host}:{self.server_port}")
                else:
                    if self.video_mode:
                        self.status_bar.update_processing_status("Video stopped")
                    else:
                        self.status_bar.update_processing_status("Camera stopped")
        
        # Start cleanup asynchronously
        QTimer.singleShot(50, perform_camera_cleanup)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Glitch Machine Engine Client')
    parser.add_argument('--host', default=DEFAULT_SERVER_HOST, help=f'Server hostname (default: {DEFAULT_SERVER_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_SERVER_PORT, help=f'Server port (default: {DEFAULT_SERVER_PORT})')
    parser.add_argument(
        '--preset',
        default=None,
        help=(
            "OPTIONAL realtime preset name to request from the server "
            "(e.g. vanilla, vanilla_journey, journey_lora_fused, "
            "journey_cn_depthanything, plantoid16). Overrides GLITCHBOX_PRESET. "
            "If unset, the server applies its own configured default — the "
            "client needs no knowledge of available presets."
        ),
    )
    args = parser.parse_args()

    # Print server configuration
    print(f"Connecting to server at {args.host}:{args.port}")
    if args.preset:
        print(f"Preset (CLI): {args.preset}")

    app = QApplication(sys.argv)
    window = MainWindow(
        server_host=args.host, server_port=args.port, preset=args.preset
    )
    window.setGeometry(100, 100, 1280, 720)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

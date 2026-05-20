from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt
import time

class StatusBar(QWidget):
    """Status bar showing connection and processing state"""
    
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # Left side container
        left_container = QWidget()
        left_layout = QHBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # FPS counter
        self.fps_label = QLabel("0 FPS")
        left_layout.addWidget(self.fps_label)
        self.frame_times = []  # Store timestamps of recent frames
        self.max_frames = 60  # Keep track of last 60 frames for FPS calculation
        
        # Small spacing between FPS and connection status
        left_layout.addSpacing(5)
        
        # Connection status
        self.conn_status = QLabel("Not Connected")
        self.conn_status.setStyleSheet("color: red;")
        left_layout.addWidget(self.conn_status)
        
        # Add left container to main layout
        self.layout.addWidget(left_container)
        
        # Add stretch to push server info to the right
        self.layout.addStretch()

        # Live audio level meter (top-right, by the status). Reflects the
        # raw mic peak from the server telemetry — claps spike it even in a
        # quiet room (unlike the percentile-normalized α).
        self.audio_label = QLabel("audio ░░░░░░░░░░")
        self.audio_label.setStyleSheet("font-family: monospace;")
        self.audio_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(self.audio_label)
        self.layout.addSpacing(8)

        # Server info on the right
        self.proc_status = QLabel("")
        self.proc_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.layout.addWidget(self.proc_status)

        self.setMinimumHeight(35)
        self.setMaximumHeight(40)

    def update_audio_level(self, level: float):
        """Render a 10-cell bar for the raw mic level (0..1) + color cue."""
        level = max(0.0, min(1.0, float(level)))
        cells = int(round(level * 10))
        bar = "█" * cells + "░" * (10 - cells)
        color = "green" if level > 0.02 else "gray"
        self.audio_label.setText(f"audio {bar} {level:.2f}")
        self.audio_label.setStyleSheet(f"font-family: monospace; color: {color};")

    def update_connection_status(self, connected: bool):
        """Update connection status display"""
        if connected:
            self.conn_status.setText("Connected")
            self.conn_status.setStyleSheet("color: green;")
        else:
            self.conn_status.setText("Not Connected")
            self.conn_status.setStyleSheet("color: red;")

    def update_fps(self):
        """Update FPS counter based on received frames using a rolling window"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only the last max_frames timestamps
        if len(self.frame_times) > self.max_frames:
            self.frame_times = self.frame_times[-self.max_frames:]
        
        # Calculate FPS based on the time difference between oldest and newest frame
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                fps = (len(self.frame_times) - 1) / time_diff
                self.fps_label.setText(f"{fps:.1f} FPS")

    def _truncate_message(self, message: str, max_length: int = 50) -> str:
        """Truncate message if it's too long"""
        if len(message) > max_length:
            return message[:max_length-3] + "..."
        return message

    def update_processing_status(self, status: str):
        """Update processing status message"""
        truncated_status = self._truncate_message(status)
        self.proc_status.setText(truncated_status)
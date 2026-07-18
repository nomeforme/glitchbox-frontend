from PySide6.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Signal, QThread, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter
import numpy as np
import cv2
import requests
import threading
import sys
import os
import zmq
import time
from dotenv import load_dotenv

# Add the parent directory to the path to allow importing from the parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, DISPLAY_SCALE
from .fullscreen_window import FullscreenWindow
from .projection_mapper import ProjectionMapperWindow

# Load environment variables
load_dotenv(override=True)

# Server configuration
DEFAULT_SERVER_HOST = os.getenv("DEFAULT_SERVER_HOST")
DEFAULT_SERVER_ZMQ_PORT = os.getenv("DEFAULT_SERVER_ZMQ_PORT")

print(f"DEFAULT_SERVER_HOST: {DEFAULT_SERVER_HOST}")
print(f"DEFAULT_SERVER_ZMQ_PORT: {DEFAULT_SERVER_ZMQ_PORT}")

class StreamThread(QThread):
    """Thread for handling MJPEG stream from server"""
    frame_received = Signal(np.ndarray)
    
    def __init__(self, stream_url):
        super().__init__()
        self.stream_url = stream_url
        self.running = False

    def run(self):
        """Process the MJPEG stream"""
        try:
            response = requests.get(self.stream_url, stream=True)
            bytes_buffer = bytes()
            self.running = True
            
            while self.running:
                chunk = response.raw.read(1024)
                if not chunk:
                    break
                    
                bytes_buffer += chunk
                a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
                b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_buffer[a:b+2]
                    bytes_buffer = bytes_buffer[b+2:]
                    
                    # Decode JPEG to numpy array
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    
                    if frame is not None:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_received.emit(frame_rgb)
                        
        except Exception as e:
            print(f"[Stream] Error processing stream: {e}")
        finally:
            self.running = False

    def stop(self):
        """Stop the stream thread"""
        self.running = False
        # Try to wait briefly for graceful shutdown
        if self.wait(200):  # Wait up to 200ms
            print("[Stream] Thread stopped gracefully")
        else:
            print("[Stream] Thread didn't stop gracefully")

class ZMQThread(QThread):
    """Thread for handling ZMQ image stream.

    ``frame_received`` carries no payload — see ``get_latest_frame()``.
    """
    frame_received = Signal()

    def __init__(self):
        super().__init__()
        self.running = False
        # Single-slot mailbox: the network thread overwrites this in place
        # instead of every decoded frame riding its own queued signal
        # payload across the thread boundary. Qt's queued-connection event
        # queue does NOT coalesce custom signals — under jitter, if frames
        # arrive faster than the GUI thread can paint, a signal-per-frame
        # design backs the queue up with several stale QImage payloads that
        # the event loop then dutifully paints in arrival order, compounding
        # lag instead of degrading gracefully. With a mailbox, every queued
        # frame_received delivery just triggers a re-read of whatever is
        # *currently* latest, so a backlog collapses to "always show the
        # newest frame" instead of painting a growing sequence of stale ones.
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._latest_is_bgr = False
        # Monotonic frame counter: lets the GUI-side consumer skip
        # duplicate deliveries. Under a paint backlog, several queued
        # frame_received notifications can all resolve to the same mailbox
        # content — without this, each would re-render (and re-copy) the
        # identical frame and inflate the FPS counter.
        self._frame_seq = 0
        print("[ZMQ] Initializing ZMQ context and socket...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        zmq_address = f"tcp://{DEFAULT_SERVER_HOST}:{DEFAULT_SERVER_ZMQ_PORT}"
        print(f"[ZMQ] Attempting to connect to {zmq_address}...")
        try:
            # CONFLATE keeps only the single latest message in this socket's
            # own queue — but that alone only governs ZMQ's userspace queue.
            # Bytes already handed to the OS TCP buffers (which happens
            # eagerly, well before HWM is typically hit over TCP) are still
            # subject to ordinary in-order TCP delivery/head-of-line
            # blocking on the wire. Setting CONFLATE here closes half the
            # gap (the publisher side sets SNDHWM=2 but not CONFLATE); doing
            # it on both ends is the documented fix for exactly this
            # "growing latency under jitter" ZMQ PUB/SUB failure mode.
            #
            # IMPORTANT: must be set BEFORE connect() — like the HWM
            # options, CONFLATE shapes the pipe created at connection time
            # and setting it after connect() silently does nothing for the
            # already-established connection. (RCVTIMEO/LINGER/SUBSCRIBE
            # below are exceptions that apply immediately.)
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.connect(zmq_address)
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
            # Set socket options to prevent blocking
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second receive timeout
            self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait for pending messages on close
            print("[ZMQ] Socket connected and subscribed (CONFLATE on)")
        except Exception as e:
            print(f"[ZMQ] Failed to connect to ZMQ socket: {e}")
            import traceback
            traceback.print_exc()

    def get_latest_frame(self):
        """Thread-safe read of the mailbox.

        Returns (frame, is_bgr, seq) or None. ``seq`` increments once per
        published frame — consumers remember the last seq they rendered and
        skip when unchanged.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame, self._latest_is_bgr, self._frame_seq

    def _publish_frame(self, frame, is_bgr: bool):
        with self._frame_lock:
            self._latest_frame = frame
            self._latest_is_bgr = is_bgr
            self._frame_seq += 1
        self.frame_received.emit()

    def run(self):
        """Process the ZMQ stream"""
        try:
            print("[ZMQ] Starting ZMQ stream processing...")
            self.running = True

            while self.running:
                try:
                    # Receive bytes with timeout (using socket timeout set above)
                    data = self.socket.recv()
                    if not data:
                        continue

                    # Check if data is JPEG encoded (starts with FFD8)
                    is_jpeg = len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8

                    if is_jpeg:
                        # Decode JPEG. cv2.imdecode's native output is BGR —
                        # deliberately NOT converting to RGB here anymore.
                        # QImage.Format_BGR888 (see ProcessedDisplay) can
                        # read these bytes directly; the cvtColor was a pure
                        # per-frame CPU tax (a full-frame channel-shuffle
                        # copy) for zero benefit, since the display path can
                        # just be told the bytes are already BGR.
                        frame = cv2.imdecode(
                            np.frombuffer(data, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        if frame is None:
                            print("[ZMQ] Failed to decode JPEG frame")
                            continue
                        is_bgr = True
                    else:
                        # Raw bytes (original behavior, legacy/unused by the
                        # current realtime server which always emits JPEG —
                        # preserved as its original RGB-assumed layout).
                        frame = np.frombuffer(data, dtype=np.uint8)

                        # Calculate expected size based on display dimensions and upscaling
                        expected_size = int(DISPLAY_HEIGHT * DISPLAY_WIDTH * 3 * (DISPLAY_SCALE ** 2))
                        if len(frame) != expected_size:
                            print(f"[ZMQ] Warning: Received data size {len(frame)} doesn't match expected size {expected_size}")
                            continue

                        # Reshape to image dimensions accounting for upscaling
                        frame = frame.reshape(int(DISPLAY_HEIGHT * DISPLAY_SCALE), int(DISPLAY_WIDTH * DISPLAY_SCALE), 3)
                        is_bgr = False

                    if frame is not None:
                        self._publish_frame(frame, is_bgr)
                    else:
                        print("[ZMQ] Failed to process frame")
                        
                except zmq.error.Again:
                    print("[ZMQ] ZMQ timeout - no data received")
                    continue
                except zmq.error.ZMQError as e:
                    if not self.running:
                        print("[ZMQ] ZMQ error during shutdown (expected)")
                        break
                    print(f"[ZMQ] ZMQ error: {e}")
                    continue
                except Exception as e:
                    if not self.running:
                        print("[ZMQ] Exception during shutdown (expected)")
                        break
                    print(f"[ZMQ] Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                        
        except Exception as e:
            print(f"[ZMQ] Error in ZMQ thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[ZMQ] Cleaning up ZMQ resources...")
            self.running = False
            try:
                # Close socket first with timeout
                if hasattr(self, 'socket') and self.socket:
                    self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait for pending messages
                    self.socket.close()
                
                # Terminate context with timeout
                if hasattr(self, 'context') and self.context:
                    self.context.term()
                print("[ZMQ] ZMQ resources cleaned up successfully")
            except Exception as e:
                print(f"[ZMQ] Error during cleanup (non-critical): {e}")
            finally:
                self.socket = None
                self.context = None

    def stop(self):
        """Stop the ZMQ thread"""
        print("[ZMQ] Stopping ZMQ thread...")
        self.running = False
        
        # Try to wait briefly for graceful shutdown
        if self.wait(200):  # Wait up to 200ms
            print("[ZMQ] Thread stopped gracefully")
        else:
            print("[ZMQ] Thread didn't stop gracefully, will be terminated externally")


class GLImageWidget(QOpenGLWidget):
    """Drop-in replacement for QLabel+QPixmap on the live video path.

    QLabel.setPixmap() forces two CPU-bound costs on *every single frame*:
    QPixmap.fromImage() deep-copies + reformats the pixel data, and
    Qt.SmoothTransformation CPU-rescales the whole image to the label's
    current size — both paid again on every repaint, not just on frame
    arrival. Painting via QPainter.drawImage() inside a QOpenGLWidget lets
    Qt's OpenGL-backed paint engine do the scale/composite on the GPU
    instead. Real-world precedent for this exact swap: ~8-9% CPU vs.
    saturating a CPU core at 1920x1200@30fps.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None

    def set_image(self, q_image: QImage):
        self._image = q_image
        self.update()

    def clear(self):
        self._image = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if self._image is not None and not self._image.isNull():
            target_size = self._image.size().scaled(self.size(), Qt.KeepAspectRatio)
            x = (self.width() - target_size.width()) // 2
            y = (self.height() - target_size.height()) // 2
            painter.drawImage(QRect(x, y, target_size.width(), target_size.height()), self._image)
        else:
            painter.fillRect(self.rect(), Qt.black)
        painter.end()


class ProcessedDisplay(QWidget):
    """Widget to display processed image output"""
    
    def __init__(self, min_size=(640, 360)):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Image display — GPU-composited (see GLImageWidget); centering is
        # handled internally in its paintEvent, so no setAlignment here.
        self.image_label = GLImageWidget()
        self.image_label.setMinimumSize(*min_size)
        self.layout.addWidget(self.image_label)
        
        # Projection Mapper button
        self.button_layout = QHBoxLayout()
        self.fullscreen_button = QPushButton("Projection Mapper")
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.fullscreen_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 50%);
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: rgba(0, 0, 0, 70%);
            }
        """)
        self.button_layout.addStretch()
        self.button_layout.addWidget(self.fullscreen_button)
        self.layout.addLayout(self.button_layout)
        
        # Stream handling
        self.stream_thread = None
        self.zmq_thread = None
        self._last_rendered_seq = -1  # see _on_zmq_frame duplicate-skip
        
        # Projection mapper window
        self.projection_mapper = None
        self.is_fullscreen = False
        
        # Black frame mode
        self.black_frame_mode = False

        # Mirror mode
        self.mirrored = False
        self.is_mirrored = False
        
    def toggle_fullscreen(self):
        """Toggle projection mapper window"""
        if not self.is_fullscreen:
            if not self.projection_mapper:
                self.projection_mapper = ProjectionMapperWindow()
                # Connect close signal to update button state
                self.projection_mapper.window_closed.connect(self.on_projection_mapper_closed)

            self.projection_mapper.show()
            # Don't go fullscreen immediately - user will do that from the mapper window
            self.is_fullscreen = True
            self.fullscreen_button.setText("Close Projection Mapper")
        else:
            if self.projection_mapper:
                self.projection_mapper.close()
                self.projection_mapper = None
            self.is_fullscreen = False
            self.fullscreen_button.setText("Projection Mapper")

    def on_projection_mapper_closed(self):
        """Handle projection mapper being closed via X button"""
        self.projection_mapper = None
        self.is_fullscreen = False
        self.fullscreen_button.setText("Projection Mapper")

    def start_stream(self, user_id: str, server_uri: str = "http://localhost:7860"):
        """Start receiving the image stream
        
        Args:
            user_id: The user ID for the stream
            server_uri: The server URI (default: http://localhost:7860)
        """
        if self.stream_thread and self.stream_thread.running:
            self.stop_stream()
            
        # Create and start ZMQ thread
        print("[ZMQ] Starting ZMQ stream")
        self.zmq_thread = ZMQThread()
        self._last_rendered_seq = -1  # fresh thread → fresh seq space
        self.zmq_thread.frame_received.connect(self._on_zmq_frame)
        self.zmq_thread.start()
        
        # NOTE: Required for ZMQ to start
        stream_url = f"{server_uri}/api/stream/{user_id}"
        print(f"[Stream] Starting WebSocket stream from: {stream_url}")
        self.stream_thread = StreamThread(stream_url)
        self.stream_thread.frame_received.connect(self.update_frame)
        self.stream_thread.start()

    def stop_stream(self):
        """Stop the stream threads"""
        print("[Display] Stopping stream threads...")
        if self.stream_thread:
            print("[Display] Stopping stream thread...")
            self.stream_thread.stop()
            # Use QTimer to wait for thread to finish without blocking UI
            self._wait_for_stream_cleanup()
            
        if self.zmq_thread:
            print("[Display] Stopping ZMQ thread...")
            self.zmq_thread.stop()
            # Use QTimer to wait for thread to finish without blocking UI
            self._wait_for_zmq_cleanup()

    def _wait_for_zmq_cleanup(self):
        """Wait for ZMQ thread to finish without blocking UI"""
        if self.zmq_thread is None:
            return
            
        # Check if thread has finished
        if self.zmq_thread.isFinished():
            print("[Display] ZMQ thread finished gracefully")
            self.zmq_thread = None
            return
        
        # If thread is still running, try to terminate it gracefully
        if self.zmq_thread.isRunning():
            print("[Display] ZMQ thread still running, attempting graceful termination...")
            from PySide6.QtCore import QTimer
            
            # Try to wait a bit more, then force terminate if needed
            def check_and_terminate():
                if self.zmq_thread and self.zmq_thread.isRunning():
                    print("[Display] Force terminating ZMQ thread...")
                    self.zmq_thread.terminate()
                    # Give termination a moment to complete
                    QTimer.singleShot(100, lambda: setattr(self, 'zmq_thread', None))
                else:
                    print("[Display] ZMQ thread stopped gracefully")
                    self.zmq_thread = None
            
            QTimer.singleShot(500, check_and_terminate)  # Wait 500ms then check
        else:
            self.zmq_thread = None
            print("[Display] ZMQ thread cleanup completed")

    def _wait_for_stream_cleanup(self):
        """Wait for stream thread to finish without blocking UI"""
        if self.stream_thread is None:
            return
            
        # Check if thread has finished
        if self.stream_thread.isFinished():
            print("[Display] Stream thread finished gracefully")
            self.stream_thread = None
            return
        
        # If thread is still running, try to terminate it gracefully
        if self.stream_thread.isRunning():
            print("[Display] Stream thread still running, attempting graceful termination...")
            from PySide6.QtCore import QTimer
            
            # Try to wait a bit more, then force terminate if needed
            def check_and_terminate():
                if self.stream_thread and self.stream_thread.isRunning():
                    print("[Display] Force terminating stream thread...")
                    self.stream_thread.terminate()
                    # Give termination a moment to complete
                    QTimer.singleShot(100, lambda: setattr(self, 'stream_thread', None))
                else:
                    print("[Display] Stream thread stopped gracefully")
                    self.stream_thread = None
            
            QTimer.singleShot(300, check_and_terminate)  # Wait 300ms then check
        else:
            self.stream_thread = None
            print("[Display] Stream thread cleanup completed")

    def update_frame(self, frame: np.ndarray):
        """Update the display with a new RGB888 frame (legacy MJPEG
        StreamThread / V1 ws_client path — these still hand over
        already-RGB-ordered frames)."""
        self._render_frame(frame, bgr=False)

    def _on_zmq_frame(self):
        """Mailbox consumer for ZMQThread.frame_received (no payload).

        Always reads ZMQThread.latest_frame — the single mutex-guarded slot
        the network thread overwrites in place — rather than a payload
        bundled with this specific signal delivery. If several
        frame_received deliveries are backlogged in Qt's queued-connection
        event queue after a stall, each one just repaints whatever is
        *currently* latest instead of dutifully repainting a growing
        sequence of stale frames in arrival order.
        """
        if not self.zmq_thread:
            return
        result = self.zmq_thread.get_latest_frame()
        if result is None:
            return
        frame, is_bgr, seq = result
        if seq == self._last_rendered_seq:
            # Backlogged notification resolving to a frame we already
            # painted — skip the redundant re-copy/repaint entirely.
            return
        self._last_rendered_seq = seq
        self._render_frame(frame, bgr=is_bgr)

    def _render_frame(self, frame: np.ndarray, bgr: bool):
        """Shared rendering path for both frame sources above."""
        if frame is None:
            return

        # If black frame mode is enabled, show black frame instead
        if self.black_frame_mode:
            # Calculate image dimensions from config
            height = int(DISPLAY_HEIGHT * DISPLAY_SCALE)
            width = int(DISPLAY_WIDTH * DISPLAY_SCALE)

            # Create black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            bgr = False  # all-zero — channel order is moot

        # Apply mirroring if enabled
        if self.mirrored:
            frame = cv2.flip(frame, 1)

        # Only render on the topmost active layer
        if self.projection_mapper and self.is_fullscreen:
            # ProjectionMapperWindow.display_frame() hardcodes
            # QImage.Format_RGB888 — convert only on this (secondary,
            # optional) path so the primary live-display path below never
            # pays for a channel-shuffle it doesn't need.
            pm_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if bgr else frame
            self.projection_mapper.update_frame(pm_frame)
        else:
            # Projection mapper is not open - render in main display.
            height, width = frame.shape[:2]
            bytes_per_line = 3 * width
            qimage_format = QImage.Format_BGR888 if bgr else QImage.Format_RGB888
            # .copy() forces QImage to own its pixel buffer now (same
            # memory-safety guarantee QPixmap.fromImage() used to provide
            # implicitly) rather than referencing `frame`'s numpy buffer,
            # which goes out of scope as soon as this function returns —
            # GLImageWidget.set_image() just stores the QImage and repaints
            # asynchronously on the next paint cycle, so the buffer must
            # outlive this call. The scale-to-fit work that used to happen
            # here via Qt.SmoothTransformation on every paint now happens
            # once, on the GPU, inside GLImageWidget.paintEvent().
            q_image = QImage(frame.data, width, height, bytes_per_line, qimage_format).copy()
            self.image_label.set_image(q_image)

        # Update FPS counter in status bar
        main_window = self.window()
        if main_window:
            main_window.status_bar.update_fps()

    def clear_display(self):
        """Clear the display and stop stream"""
        self.stop_stream()
        self.image_label.clear()
        if self.projection_mapper:
            self.projection_mapper.clear_display()

    def clear_zmq_queue(self):
        """Clear any pending messages in the ZMQ queue"""
        if self.zmq_thread and self.zmq_thread.running and hasattr(self.zmq_thread, 'socket') and self.zmq_thread.socket:
            try:
                cleared_count = 0
                # Clear pending messages by receiving all available data without blocking
                while self.zmq_thread.socket.poll(timeout=0) > 0:  # 0 timeout = non-blocking
                    self.zmq_thread.socket.recv(zmq.NOBLOCK)
                    cleared_count += 1
                    # Prevent infinite loops
                    if cleared_count > 100:
                        print(f"[ZMQ] Cleared {cleared_count} messages, stopping to prevent infinite loop")
                        break
                if cleared_count > 0:
                    print(f"[ZMQ] Cleared {cleared_count} pending messages from queue")
            except zmq.error.Again:
                # No more messages to clear
                pass
            except Exception as e:
                print(f"[ZMQ] Error clearing queue (non-critical): {e}")

    def display_black_frame(self):
        """Display a black frame of the expected image size"""
        try:
            # Calculate image dimensions from config
            height = int(DISPLAY_HEIGHT * DISPLAY_SCALE)
            width = int(DISPLAY_WIDTH * DISPLAY_SCALE)
            
            # Create black frame (RGB)
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Display the black frame
            self.update_frame(black_frame)
            print(f"[Display] Showing black frame of size {width}x{height}")
            
        except Exception as e:
            print(f"[Display] Error creating black frame: {e}")

    def set_black_frame_mode(self, enabled: bool):
        """Enable or disable black frame mode"""
        self.black_frame_mode = enabled
        if enabled:
            print("[Display] Black frame mode enabled")
        else:
            print("[Display] Black frame mode disabled")

    def cleanup(self):
        """Clean up all resources"""
        print("[Display] Starting cleanup...")
        self.stop_stream()
        if self.projection_mapper:
            print("[Display] Closing projection mapper window...")
            self.projection_mapper.close()
            self.projection_mapper = None
        print("[Display] Cleanup completed")

    def set_mirror_mode(self, enabled: bool):
        """Enable or disable mirror mode"""
        self.mirrored = enabled
        self.is_mirrored = enabled
        if enabled:
            print("[Display] Mirror mode enabled")
        else:
            print("[Display] Mirror mode disabled")

    def toggle_mirror(self):
        """Toggle mirror mode"""
        self.set_mirror_mode(not self.mirrored)
        
    def __del__(self):
        """Cleanup on deletion"""
        try:
            # Safely stop threads without waiting to prevent core dumps during destruction
            if hasattr(self, 'stream_thread') and self.stream_thread:
                self.stream_thread.running = False
                if self.stream_thread.isRunning():
                    self.stream_thread.terminate()
                
            if hasattr(self, 'zmq_thread') and self.zmq_thread:
                self.zmq_thread.running = False
                if self.zmq_thread.isRunning():
                    self.zmq_thread.terminate()

            if hasattr(self, 'projection_mapper') and self.projection_mapper:
                self.projection_mapper.close()
        except Exception as e:
            print(f"[Display] Error during destruction: {e}")

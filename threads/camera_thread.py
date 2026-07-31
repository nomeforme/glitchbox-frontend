from PySide6.QtCore import QThread, Signal
import cv2
import time
import numpy as np
import sys
import os

# Add the parent directory to the path to allow importing from the parent package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, CAMERA_DEVICE_INDEX

# Capture-side negotiation: ask every camera for its full-sensor MJPG 720p
# mode. Cheap webcams implement low-res modes as a center *window crop* of
# the sensor (zoomed-in FOV) instead of a downscale, so capturing at native
# 720p is the only way to get their full field of view. MJPG matters: YUYV
# 720p is USB2-bandwidth-capped to ~10fps on such cams; MJPG runs full rate.
_CAP_WIDTH = 1280
_CAP_HEIGHT = 720

# Emit-side conditioning: frames are (optionally) center-cropped to the
# server's render aspect and downscaled to <= 640 wide before leaving the
# thread. The server stretch-resizes whatever we send to exactly its render
# resolution (core/realtime.py _decode_jpeg_to_tensor — no crop, no
# letterbox), so matching its aspect here is what keeps the geometry
# undistorted; the downscale keeps JPEG bytes at the pre-720p pipeline's
# level so slow uplinks don't choke.
#
# ``match_aspect`` (UI: "Match server aspect" checkbox, default on) selects
# the trade: on = distortion-free crop (from a 720p capture, the middle
# 960x720 of the sensor at the default 4:3); off = full-sensor FOV,
# aspect-preserving downscale only, and the server's stretch shows.
# ``target_aspect`` defaults to 4:3 (the production preset renders
# 1024x768) and is overwritten with the true render aspect from the
# session_ready capabilities on every connect.
_DEFAULT_ASPECT = (4, 3)
_EMIT_MAX_WIDTH = 640


class CameraThread(QThread):
    """Thread for handling camera capture"""
    frame_ready = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.device_index = CAMERA_DEVICE_INDEX
        # Emit conditioning (see module comment). Plain attributes, read
        # once per loop iteration — same cross-thread pattern as `running`.
        self.match_aspect = True
        self.target_aspect = _DEFAULT_ASPECT

    def _describe_mode(self):
        w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.camera.get(cv2.CAP_PROP_FPS)
        fcc = int(self.camera.get(cv2.CAP_PROP_FOURCC))
        fcc_str = "".join(chr((fcc >> (8 * i)) & 0xFF) for i in range(4))
        return f"{w}x{h} @ {fps:.1f}fps ({fcc_str.strip() or '?'})"

    def _open(self, negotiate):
        """Open the device; optionally request full-sensor MJPG 720p."""
        if negotiate:
            cam = cv2.VideoCapture(self.device_index, cv2.CAP_V4L2)
            cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, _CAP_WIDTH)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAP_HEIGHT)
        else:
            cam = cv2.VideoCapture(self.device_index)
        return cam

    def run(self):
        """Main thread loop for capturing camera frames"""
        try:
            self.camera = self._open(negotiate=True)
            # A camera (or virtual device) that rejects the negotiated mode
            # outright falls back to the plain open the client always used.
            if self.camera.isOpened():
                ok, _ = self.camera.read()
            else:
                ok = False
            if not ok:
                print("[Camera] MJPG 720p negotiation failed — "
                      "reopening with driver defaults")
                if self.camera is not None:
                    self.camera.release()
                self.camera = self._open(negotiate=False)
            if not self.camera.isOpened():
                print("[Camera] Failed to open camera")
                return
            aw, ah = self.target_aspect
            mode = (f"{aw}:{ah} center crop" if self.match_aspect
                    else "full FOV (no crop)")
            print(f"[Camera] Negotiated {self._describe_mode()}, "
                  f"emitting {mode} at <= {_EMIT_MAX_WIDTH}px wide")

            self.running = True
            fps_count = 0
            fps_t0 = time.monotonic()
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    print("[Camera] Failed to read frame")
                    break

                fps_count += 1
                now = time.monotonic()
                if now - fps_t0 >= 5.0:
                    print(f"[Camera] Capture rate: "
                          f"{fps_count / (now - fps_t0):.1f} fps")
                    fps_count = 0
                    fps_t0 = now

                # Center-crop to the server's render aspect so its
                # stretch-resize is distortion-free (skipped when the
                # "Match server aspect" checkbox is off → full FOV).
                h, w = frame.shape[:2]
                if self.match_aspect:
                    aw, ah = self.target_aspect
                    crop_w = (h * aw) // ah
                    if w > crop_w:              # wider than target (16:9 etc.)
                        x0 = (w - crop_w) // 2
                        frame = frame[:, x0:x0 + crop_w]
                        w = crop_w
                    else:                       # taller than target
                        crop_h = (w * ah) // aw
                        if h > crop_h:
                            y0 = (h - crop_h) // 2
                            frame = frame[y0:y0 + crop_h, :]
                            h = crop_h

                # Downscale (never upscale) so preview + WS send cost the
                # same as the pre-720p pipeline regardless of capture mode.
                if w > _EMIT_MAX_WIDTH:
                    new_h = max(1, round(h * _EMIT_MAX_WIDTH / w))
                    frame = cv2.resize(frame, (_EMIT_MAX_WIDTH, new_h),
                                       interpolation=cv2.INTER_AREA)

                # Convert BGR to RGB for display
                # TODO: Remove this once we have a working RGB frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(rgb_frame)

                # Small delay to prevent tight loop
                time.sleep(0.01)
                
        except Exception as e:
            print(f"[Camera] Error in camera thread: {e}")
        finally:
            self.cleanup()

    def stop(self):
        """Stop the camera thread"""
        print("[Camera] Stopping camera thread")
        self.running = False
        
        # First set running to false to break out of the loop
        self.running = False
        
        # Add a timeout for waiting to prevent hanging
        if not self.wait(2000):  # Wait max 2 seconds for thread to finish
            print("[Camera] Thread wait timed out, forcing termination")
            self.terminate()  # Force terminate if it doesn't finish in time
            
        # Only after the thread is done (or timeout), release the camera
        self.cleanup()

    def cleanup(self):
        """Clean up camera resources"""
        print("[Camera] Cleaning up camera resources")
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            self.camera = None
        self.running = False

    def __del__(self):
        """Destructor to ensure camera is released"""
        self.cleanup()
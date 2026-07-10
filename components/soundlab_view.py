"""Sound Lab window — live visualization of the server's soundlab policy
telemetry (audio_alpha_mode: "policy").

Renders the per-frame `soundlab` sub-dict of the alpha telemetry:
  bold line   final α (what the blend actually does)
  thin line   raw DSP contour (pre-staging, pre-gesture)
  shaded band CLAP staging range [lo, hi] for the current theme
  dashed rows event markers (drops orange, lulls/silence purple,
              voice entrances/dropouts blue; stem-lane tagged)

Pure QPainter — no plotting dependencies. The window is passive: it
visualizes; all control stays with the server config / control panel.
"""

from collections import deque

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit,
                               QVBoxLayout, QWidget)

SPAN_S = 120.0          # visible history window
MAX_FRAMES = 20 * 150   # ring capacity (~150 s at 20 fps)

COL_BG = QColor("#11141b")
COL_GRID = QColor("#1c2130")
COL_BAND = QColor(127, 212, 255, 18)
COL_ALPHA = QColor("#7fd4ff")
COL_CONTOUR = QColor("#3d4a63")
COL_TEXT = QColor("#93a3bd")
EVENT_COLORS = {
    "drop": QColor("#ff7f5f"),
    "lull_start": QColor("#b48cff"), "lull_end": QColor("#b48cff"),
    "silence_start": QColor("#b48cff"), "silence_end": QColor("#b48cff"),
}
COL_EVENT_DEFAULT = QColor("#5a8cff")


class _Chart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames = deque(maxlen=MAX_FRAMES)   # dicts: t, alpha, contour, lo, hi
        self.events = deque(maxlen=400)          # dicts: t, kind, band, lane
        self.setMinimumHeight(260)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), COL_BG)
        w, h = self.width(), self.height()
        if len(self.frames) < 2:
            p.setPen(COL_TEXT)
            p.drawText(self.rect(), Qt.AlignCenter,
                       "waiting for policy telemetry… "
                       "(server audio_alpha_mode must be \"policy\")")
            return
        t1 = self.frames[-1]["t"]
        t0 = t1 - SPAN_S

        def X(t):
            return (t - t0) / SPAN_S * w

        def Y(v):
            return h - 8 - max(0.0, min(1.0, v)) * (h - 16)

        # staging band (current range)
        lo, hi = self.frames[-1]["lo"], self.frames[-1]["hi"]
        p.fillRect(0, int(Y(hi)), w, int(Y(lo) - Y(hi)), COL_BAND)

        # gridlines
        p.setPen(QPen(COL_GRID, 1))
        for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
            p.drawLine(0, int(Y(gv)), w, int(Y(gv)))
            p.setPen(COL_TEXT)
            p.drawText(4, int(Y(gv)) - 3, f"{gv:.2f}")
            p.setPen(QPen(COL_GRID, 1))

        # events
        for e in self.events:
            if e["t"] < t0:
                continue
            x = int(X(e["t"]))
            col = EVENT_COLORS.get(e["kind"], COL_EVENT_DEFAULT)
            pen = QPen(col, 1, Qt.DashLine)
            p.setPen(pen)
            p.drawLine(x, 0, x, h)
            p.setPen(COL_TEXT)
            label = e["kind"] + (f":{e['band']}" if e.get("band") else "")
            p.drawText(x + 3, 12, label)

        # contour (thin) then alpha (bold)
        for key, pen in (("contour", QPen(COL_CONTOUR, 1)),
                         ("alpha_norm", QPen(COL_ALPHA, 2))):
            p.setPen(pen)
            prev = None
            for f in self.frames:
                if f["t"] < t0:
                    continue
                if key == "contour":
                    # contour drawn WITHIN the staging band (it is the
                    # base curve before gestures): lo + c * (hi - lo)
                    v = f["lo"] + f["contour"] * (f["hi"] - f["lo"])
                else:
                    v = f["alpha"]
                pt = (X(f["t"]), Y(v))
                if prev is not None:
                    p.drawLine(int(prev[0]), int(prev[1]),
                               int(pt[0]), int(pt[1]))
                prev = pt
        p.end()


class SoundlabView(QWidget):
    """Top-level Sound Lab window. Feed it each telemetry sub-dict via
    ``update_telemetry``; it repaints on a fixed timer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound Lab")
        self.resize(980, 460)
        self.setStyleSheet(
            "background:#0d0f14;color:#cfd6e4;font-size:12px;")

        layout = QVBoxLayout(self)

        hdr = QHBoxLayout()
        self.alpha_label = QLabel("–")
        f = QFont()
        f.setPointSize(26)
        f.setBold(True)
        self.alpha_label.setFont(f)
        self.alpha_label.setStyleSheet("color:#7fd4ff;")
        hdr.addWidget(self.alpha_label)

        self.info_label = QLabel("")
        hdr.addWidget(self.info_label)
        hdr.addStretch()
        layout.addLayout(hdr)

        self.chart = _Chart()
        layout.addWidget(self.chart, stretch=1)

        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumHeight(96)
        self.event_log.setStyleSheet(
            "background:#11141b;border:1px solid #232838;"
            "font-family:monospace;font-size:11px;color:#9fb0c8;")
        layout.addWidget(self.event_log)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.chart.update)
        self._timer.start(80)   # ~12 Hz repaint

    def update_telemetry(self, sl: dict) -> None:
        """Slot for WSClient.soundlab_updated (one dict per frame)."""
        try:
            frame = {
                "t": float(sl.get("t", 0.0)),
                "alpha": float(sl.get("alpha", 0.0)),
                "contour": float(sl.get("contour", 0.0)),
                "lo": float(sl.get("lo", 0.0)),
                "hi": float(sl.get("hi", 1.0)),
            }
        except (TypeError, ValueError):
            return
        self.chart.frames.append(frame)
        for e in sl.get("events") or []:
            self.chart.events.append(e)
            lane = " (stem)" if e.get("lane") == "stem" else ""
            band = f":{e['band']}" if e.get("band") else ""
            self.event_log.appendPlainText(
                f"[{e.get('t', 0):7.1f}s] {e.get('kind')}{band}{lane}")
        self.alpha_label.setText(f"{frame['alpha']:.3f}")
        probs = sl.get("theme_probs")
        probs_s = ("  " + " ".join(f"{k}:{v:.2f}" for k, v in probs.items())
                   if isinstance(probs, dict) else "")
        self.info_label.setText(
            f"state {sl.get('state', '?')}   theme {sl.get('theme', '?')}"
            f"   staging [{frame['lo']:.2f}, {frame['hi']:.2f}]"
            f"   stem {sl.get('stem', '?')}   clap {sl.get('clap', '?')}"
            f"   t {frame['t']:.1f}s{probs_s}")

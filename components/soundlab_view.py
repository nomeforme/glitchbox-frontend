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

import math
from collections import deque

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QHBoxLayout,
                               QLabel, QPlainTextEdit, QSlider, QTabWidget,
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
        try:
            self._paint(p)
        except Exception:
            pass       # never let a bad frame kill the app
        finally:
            p.end()

    def _paint(self, p):
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
                if not math.isfinite(v):
                    prev = None
                    continue
                pt = (X(f["t"]), Y(v))
                if prev is not None:
                    p.drawLine(int(prev[0]), int(prev[1]),
                               int(pt[0]), int(pt[1]))
                prev = pt


class _LaneChart(QWidget):
    """Decomposition tab: stacked per-signal lanes (trail style).

    Top lane `r` is the LINEAR COMPOSER output: r = Σ w_i·s_i over the
    enabled NORMALIZED signals with Σw_i = 1 (weights renormalized over
    the enabled set). Display-only for now — it does not drive the deck.
    A `clap` lane shows the instantaneous semantic verdict for visual
    alignment against the low-level signals."""

    # (key, color, model-tag, composable)
    LANES = [
        ("r",      QColor("#ffffff"), "mix",      False),
        ("beat",   QColor("#ff7f5f"), "beatnet",  True),
        ("down",   QColor("#ff4f9f"), "beatnet",  True),
        ("onset",  QColor("#ffb75f"), "dsp",      True),
        ("pitch",  QColor("#7fd4ff"), "pesto",    True),
        ("drums",  QColor("#ff5f7f"), "htdemucs", True),
        ("bass",   QColor("#e0e05f"), "htdemucs", True),
        ("other",  QColor("#5fe08f"), "htdemucs", True),
        ("vocals", QColor("#b48cff"), "htdemucs", True),
        ("clap",   QColor("#8fdccf"), "clap",     False),
        ("events", QColor("#93a3bd"), "p1",       False),
    ]
    COMPOSABLE = [k for k, _, _, c in LANES if c]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist = deque(maxlen=20 * 150)   # dicts keyed by lane name
        self.events = deque(maxlen=300)      # P1 events for the overlay
        # composer state: enabled + weight per composable signal
        self.enabled = {k: k in ("beat", "bass") for k in self.COMPOSABLE}
        self.weights = {k: 0.5 for k in self.COMPOSABLE}
        # rolling normalization state per signal (for r at push time)
        self._peak = {k: 1e-6 for k in self.COMPOSABLE}
        self.setMinimumHeight(340)

    @staticmethod
    def _num(v, d=0.0):
        try:
            v = float(v)
            return v if math.isfinite(v) else d
        except (TypeError, ValueError):
            return d

    def push(self, sl: dict) -> None:
        stems = sl.get("stems") or [0, 0, 0, 0]
        n = self._num
        voiced = n(sl.get("voiced", 0))
        cn = sl.get("clap_now") or ["", 0.0]
        f = {
            "t": n(sl.get("t", 0.0)),
            "beat": n(sl.get("contour", 0.5), 0.5),
            "down": n(sl.get("down", 0.0)),
            "onset": n(sl.get("onset", 0.0)),
            "pitch": n(sl.get("pitch_reg", 0.5), 0.5),
            "voiced_f": voiced,
            "drums": n(stems[0]), "bass": n(stems[1]),
            "other": n(stems[2]), "vocals": n(stems[3]),
            "clap": n(cn[1] if len(cn) > 1 else 0.0),
            "clap_l": str(cn[0] if cn else ""),
            "bpm": n(sl.get("bpm", 0)), "conf": n(sl.get("beat_conf", 0)),
            "bar": n(sl.get("bar", 0)),
            "bnet_s": str(sl.get("bnet", "?"))[:26],
            "pb": str(sl.get("pitch_backend", "?"))[:8],
        }
        # composer: r = Σ w_i · s_i(normalized), weights renormalized
        # over the ENABLED set (Σw = 1); rolling-peak normalization with
        # slow decay so it adapts without pinning
        num_ = 0.0
        den = 0.0
        for k in self.COMPOSABLE:
            v = f[k]
            self._peak[k] = max(v, self._peak[k] * 0.9995, 1e-6)
            f[k + "_n"] = min(v / self._peak[k], 1.0)
            if self.enabled.get(k):
                w = max(0.0, self.weights.get(k, 0.0))
                num_ += w * f[k + "_n"]
                den += w
        f["r"] = (num_ / den) if den > 1e-9 else 0.0
        self.hist.append(f)

    def paintEvent(self, _):
        p = QPainter(self)
        try:
            self._paint(p)
        except Exception:
            pass       # a bad frame must never take the app down
        finally:
            p.end()

    def _paint(self, p):
        p.fillRect(self.rect(), COL_BG)
        if len(self.hist) < 4:
            p.setPen(COL_TEXT)
            p.drawText(self.rect(), Qt.AlignCenter, "waiting…")
            return
        w, h = self.width(), self.height()
        n = len(self.LANES)
        lane_h = h / n
        t1 = self.hist[-1]["t"]
        t0 = t1 - 60.0                       # 60 s window, denser than tab 1
        frames = [f for f in self.hist if f["t"] >= t0]
        headers = []          # deferred: drawn LAST so nothing covers them

        # ---- pass 1: lane series --------------------------------------
        for li, (key, col, model, composable) in enumerate(self.LANES):
            y0 = li * lane_h
            p.setPen(QPen(COL_GRID, 1))
            p.drawLine(0, int(y0 + lane_h - 1), w, int(y0 + lane_h - 1))
            if key == "events":
                headers.append((y0, col, "events [p1]"))
                continue
            vals = [f[key] for f in frames
                    if f[key] is not None and math.isfinite(f[key])]
            vmax = max(max(vals), 1e-6) if vals else 1.0
            norm = (1.0 if key in ("beat", "pitch", "r", "clap") else vmax)
            dim = QColor(col)
            dim.setAlpha(70)
            disabled = composable and not self.enabled.get(key)
            base_pen = QPen(dim if disabled else col, 1.4)
            prev = None
            last_lbl = None
            for f in frames:
                v = f[key]
                if v is None or not math.isfinite(v):
                    prev = None
                    continue
                x = (f["t"] - t0) / 60.0 * w
                y = y0 + lane_h - 4 - (min(v / norm, 1.0)) * (lane_h - 10)
                if not (math.isfinite(x) and math.isfinite(y)):
                    prev = None
                    continue
                if key == "pitch":
                    live = (not disabled) and f.get("voiced_f", 0.0) > 0.05
                    p.setPen(QPen(col if live else dim,
                                  1.8 if live else 1.0))
                elif key == "r":
                    p.setPen(QPen(col, 2.2))
                else:
                    p.setPen(base_pen)
                if prev is not None:
                    p.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
                else:
                    p.drawEllipse(int(x) - 1, int(y) - 1, 3, 3)
                prev = (x, y)
                # clap lane: annotate label changes on the timeline
                if key == "clap":
                    lbl = f.get("clap_l", "")
                    if lbl and lbl != last_lbl:
                        p.setPen(col)
                        p.drawText(int(x) + 2, int(y0 + lane_h - 6), lbl)
                        last_lbl = lbl

            # ---- lane header: name [model] u:<raw> n:<norm> -------------
            fl = frames[-1]
            label = f"{key} [{model}]"
            if key == "beat":
                label += (f" {fl['bpm']:.0f}bpm conf {fl['conf']:.2f} "
                          f"[{fl.get('bnet_s', '?')}]")
            elif key == "down":
                label += f" bar {fl.get('bar', 0):.1f}s"
            elif key == "pitch":
                label += f" voiced {fl.get('voiced_f', 0):.2f}"
            elif key == "r":
                on = [k for k in self.COMPOSABLE if self.enabled.get(k)]
                den = sum(max(0.0, self.weights[k]) for k in on) or 1.0
                label = ("r [mix] = " + " + ".join(
                    f"{max(0.0, self.weights[k]) / den:.2f}·{k}"
                    for k in on)) if on else "r [mix] = (nothing enabled)"
            elif key == "clap":
                label += f" {fl.get('clap_l', '')}"
            cur = fl.get(key)
            if cur is not None and math.isfinite(cur):
                label += f"   u:{cur:.4g}  n:{min(cur / norm, 1.0):.2f}"
            headers.append((y0, col, label))

        # ---- pass 2: event markers overlaid across ALL lanes ----------
        ev_lane_i = next(i for i, l in enumerate(self.LANES)
                         if l[0] == "events")
        ey0 = ev_lane_i * lane_h
        fm = p.fontMetrics()
        visible = [e for e in self.events if e.get("t", -1) >= t0]
        for e in visible:                       # oldest first: lines
            x = int((e["t"] - t0) / 60.0 * w)
            ecol = EVENT_COLORS.get(e.get("kind", ""), COL_EVENT_DEFAULT)
            pen = QPen(ecol, 1, Qt.DashLine)
            p.setPen(pen)
            p.drawLine(x, 0, x, h)
        for e in visible:                       # newest LAST = on top
            x = int((e["t"] - t0) / 60.0 * w)
            ecol = EVENT_COLORS.get(e.get("kind", ""), COL_EVENT_DEFAULT)
            txt = e.get("kind", "") + (
                f":{e['band']}" if e.get("band") else "")
            if e.get("lane") == "stem":
                txt += "*"
            tw = fm.horizontalAdvance(txt)
            bg = QColor(COL_BG)
            bg.setAlpha(200)
            p.fillRect(x + 2, int(ey0 + lane_h / 2 - 6), tw + 6, 13, bg)
            p.setPen(ecol)
            p.drawText(x + 5, int(ey0 + lane_h / 2 + 5), txt)

        # ---- pass 3: headers last, full color ALWAYS (readability) ----
        bg = QColor(COL_BG)
        bg.setAlpha(190)
        for y0, col, label in headers:
            p.fillRect(3, int(y0 + 2), fm.horizontalAdvance(label) + 8, 13, bg)
            p.setPen(col)
            p.drawText(6, int(y0 + 13), label)


class SoundlabView(QWidget):
    """Top-level Sound Lab window. Feed it each telemetry sub-dict via
    ``update_telemetry``; it repaints on a fixed timer.

    Manual lever: a big slider under the chart. When the server session
    runs audio_alpha_mode="manual", this drives the deck (via the
    ``manual_alpha`` live knob) and the session auto-logs the human trace
    next to the policy's prediction — the imitation dataset."""

    manual_changed = Signal(float)   # 0..1, wired to update_knob upstream

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
        self.lanes = _LaneChart()
        # decomposition tab = lanes + linear-composer controls
        self.decomp = QWidget()
        dv = QVBoxLayout(self.decomp)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.addWidget(self.lanes, stretch=1)
        comp_row = QHBoxLayout()
        comp_lbl = QLabel("r =")
        comp_lbl.setStyleSheet("color:#8090a8;font-weight:bold;")
        comp_row.addWidget(comp_lbl)
        for key in self.lanes.COMPOSABLE:
            cb = QCheckBox(key)
            cb.setChecked(self.lanes.enabled[key])
            cb.toggled.connect(
                lambda on, k=key: self.lanes.enabled.__setitem__(k, on))
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0)
            sp.setSingleStep(0.05)
            sp.setDecimals(2)
            sp.setValue(self.lanes.weights[key])
            sp.setFixedWidth(60)
            sp.setStyleSheet("background:#181c26;color:#cfd6e4;")
            sp.valueChanged.connect(
                lambda v, k=key: self.lanes.weights.__setitem__(k, v))
            comp_row.addWidget(cb)
            comp_row.addWidget(sp)
        comp_row.addStretch()
        note = QLabel("weights auto-renormalized to Σ=1 · display-only")
        note.setStyleSheet("color:#5a677f;font-size:10px;")
        comp_row.addWidget(note)
        dv.addLayout(comp_row)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab{background:#181c26;color:#8090a8;padding:5px 14px;}"
            "QTabBar::tab:selected{background:#232838;color:#cfd6e4;}")
        self.tabs.addTab(self.chart, "signal")
        self.tabs.addTab(self.decomp, "decomposition")
        layout.addWidget(self.tabs, stretch=1)

        # manual lever — big target, fine steps, fires on every move
        lever_row = QHBoxLayout()
        lever_label = QLabel("manual α")
        lever_label.setStyleSheet("color:#8090a8;")
        lever_row.addWidget(lever_label)
        self.lever = QSlider(Qt.Horizontal)
        self.lever.setRange(0, 1000)
        self.lever.setValue(500)
        self.lever.setMinimumHeight(36)
        self.lever.setStyleSheet(
            "QSlider::groove:horizontal{height:12px;background:#232838;"
            "border-radius:6px;}"
            "QSlider::handle:horizontal{width:34px;margin:-10px 0;"
            "background:#7fd4ff;border-radius:8px;}")
        self.lever.valueChanged.connect(
            lambda v: (self.lever_value_label.setText(f"{v / 1000:.3f}"),
                       self.manual_changed.emit(v / 1000.0)))
        lever_row.addWidget(self.lever, stretch=1)
        self.lever_value_label = QLabel("0.500")
        self.lever_value_label.setStyleSheet(
            "color:#7fd4ff;font-weight:bold;min-width:48px;")
        lever_row.addWidget(self.lever_value_label)
        layout.addLayout(lever_row)

        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumHeight(96)
        self.event_log.setStyleSheet(
            "background:#11141b;border:1px solid #232838;"
            "font-family:monospace;font-size:11px;color:#9fb0c8;")
        layout.addWidget(self.event_log)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._repaint_active)
        self._timer.start(80)   # ~12 Hz repaint

    def _repaint_active(self):
        (self.chart if self.tabs.currentIndex() == 0 else self.lanes).update()

    def update_telemetry(self, sl: dict) -> None:
        """Slot for WSClient.soundlab_updated (one dict per frame)."""
        def num(v, d):
            try:
                v = float(v)
                return v if math.isfinite(v) else d
            except (TypeError, ValueError):
                return d
        frame = {
            "t": num(sl.get("t"), 0.0),
            "alpha": num(sl.get("alpha"), 0.0),
            "contour": num(sl.get("contour"), 0.0),
            "lo": num(sl.get("lo"), 0.0),
            "hi": num(sl.get("hi"), 1.0),
        }
        self.chart.frames.append(frame)
        self.lanes.push(sl)
        for e in sl.get("events") or []:
            self.chart.events.append(e)
            self.lanes.events.append(e)
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

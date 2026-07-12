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
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QHBoxLayout, QLabel, QPlainTextEdit, QSlider,
                               QTabWidget, QVBoxLayout, QWidget)

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
        self.setMinimumHeight(200)   # must leave room for the knob +
                                     # lever rows below (they were being
                                     # clipped at small window heights)

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
    """Decomposition tab: stacked per-signal lanes (trail style) — pure
    visualization. Composition lives in the GRAPH tab; a lane draws
    bright when its signal has a source node in the active patch (the
    server's `norm` dict carries exactly those), dim otherwise. The top
    `r` lane header renders the ACTIVE deck formula."""

    # (key, color, model-tag, composable)
    LANES = [
        ("r",      QColor("#ffffff"), "graph",    False),
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
        self.formula = ""    # active deck formula (server telemetry)
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
            "dsyn": n(sl.get("down_synth", 0)),
            "bnet_s": str(sl.get("bnet", "?"))[:26],
            "stem_s": str(sl.get("stem", "?"))[:30],
            "pb": str(sl.get("pitch_backend", "?"))[:8],
            "vox": n(sl.get("vox", 0.0)),
            "mix_s": str(sl.get("mix", ""))[:20],   # active patch name
        }
        # server-side normalized signals: EXACTLY what the composer
        # mixed this frame (floor+peak stretch) — the authoritative
        # bright line; raw stays as a dim underlay
        nrm = sl.get("norm") or {}
        for k in self.COMPOSABLE:
            sv = nrm.get(k)
            f[k + "_sn"] = n(sv) if sv is not None else None
        # r = the SERVER-computed deck signal (soundlab_mix drives the
        # blend now; in manual mode alpha is the lever, policy_alpha=r)
        f["r"] = n(sl.get("policy_alpha", sl.get("alpha", 0.0)))
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
            raw_norm = (1.0 if key in ("beat", "pitch", "r", "clap")
                        else vmax)
            dim = QColor(col)
            dim.setAlpha(70)
            # bright iff the signal has a source node in the ACTIVE
            # patch (the server's norm dict carries exactly those)
            in_mix = frames[-1].get(key + "_sn") is not None
            disabled = composable and not in_mix
            base_pen = QPen(dim if disabled else col, 1.4)
            # server-normalized series available? bright line = what the
            # composer mixed; raw drops to a faint underlay
            sn_key = key + "_sn"
            use_sn = composable and any(
                f.get(sn_key) is not None for f in frames[-5:])
            if use_sn:
                under = QColor(col)
                under.setAlpha(45)
                p.setPen(QPen(under, 1.0))
                prev = None
                for f in frames:
                    v = f[key]
                    if v is None or not math.isfinite(v):
                        prev = None
                        continue
                    x = (f["t"] - t0) / 60.0 * w
                    y = (y0 + lane_h - 4
                         - min(v / raw_norm, 1.0) * (lane_h - 10))
                    if not (math.isfinite(x) and math.isfinite(y)):
                        prev = None
                        continue
                    if prev is not None:
                        p.drawLine(int(prev[0]), int(prev[1]),
                                   int(x), int(y))
                    prev = (x, y)
            norm = 1.0 if use_sn else raw_norm
            prev = None
            last_lbl = None
            for f in frames:
                v = f.get(sn_key) if use_sn else f[key]
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
                if fl.get("dsyn", 0) > 0:   # watchdog carrying the PF
                    label += f" ~synth×{int(fl['dsyn'])}"
            elif key == "pitch":
                label += f" voiced {fl.get('voiced_f', 0):.2f}"
            elif key == "drums":
                label += f" [{fl.get('stem_s', '?')}]"
            elif key == "r":
                # keep it terse — the full formula lives on the graph tab
                label = (f"r [graph→deck] {fl.get('mix_s', '')} "
                         f"vox {fl.get('vox', 0):.2f}")
            elif key == "clap":
                label += f" {fl.get('clap_l', '')}"
            cur = fl.get(key)
            if cur is not None and math.isfinite(cur):
                sn = fl.get(sn_key) if use_sn else None
                if sn is not None and math.isfinite(sn):
                    label += f"   u:{cur:.4g}  n:{sn:.2f}"
                else:
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
    knob_changed = Signal(str, object)   # audio-group server knobs
    # graph_applied (Signal(str), JSON patch -> soundlab_graph knob) is
    # re-exported from the nodal editor in __init__.

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound Lab")
        self.resize(1100, 640)   # tall enough that the signal tab's knob
                                 # + lever rows never clip off the bottom
        # NOTE: bare declarations and selector rules cannot be mixed in
        # one QSS sheet (the parser drops the sheet) — that is why the
        # manual-α checkbox rendered as label-only text (live finding
        # 2026-07-12). Everything goes inside selector blocks. Checked
        # state draws a real TICK (svg asset; QSS has no data-URIs).
        import os as _os
        _check = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                               "assets", "checkmark.svg").replace("\\", "/")
        self.setStyleSheet(
            "QWidget{background:#0d0f14;color:#cfd6e4;font-size:12px;}"
            "QCheckBox::indicator{width:15px;height:15px;"
            "border:1px solid #4a5878;border-radius:4px;"
            "background:#181c26;}"
            "QCheckBox::indicator:checked{background:#7fd4ff;"
            f"border-color:#7fd4ff;image:url({_check});}}")

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
        # top-right, tab-agnostic controls (e.g. Soundlab Log → Disk)
        self.hdr_controls = QHBoxLayout()
        hdr.addLayout(self.hdr_controls)
        layout.addLayout(hdr)

        self.chart = _Chart()
        self.lanes = _LaneChart()

        # ---- signal tab = chart + server audio knobs -------------------
        # Sound Lab is the audio control center: everything that decides
        # how sound becomes α lives here, not in the main control panel.
        # Composition itself lives in the GRAPH tab (nodal editor).
        self.signal_tab = QWidget()
        sv = QVBoxLayout(self.signal_tab)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.addWidget(self.chart, stretch=1)

        # server audio-group knobs (built from the schema on handshake);
        # irrelevant knobs auto-hide based on the active α mode/source
        self.audio_controls = {}
        self._audio_labels = {}
        self.audio_row = QHBoxLayout()
        sv.addLayout(self.audio_row)

        # manual α: checkbox arms the lever (audio_alpha_mode manual ⇄
        # policy); the lever greys out when the graph is driving
        lever_row = QHBoxLayout()
        self.manual_cb = QCheckBox("manual α")
        self.manual_cb.setStyleSheet("color:#8090a8;font-weight:bold;")
        self.manual_cb.toggled.connect(self._manual_toggled)
        lever_row.addWidget(self.manual_cb)
        self.lever = QSlider(Qt.Horizontal)
        self.lever.setRange(0, 1000)
        self.lever.setValue(500)
        self.lever.setMinimumHeight(36)
        self._style_lever(active=False)
        self.lever.setEnabled(False)
        self.lever.valueChanged.connect(
            lambda v: (self.lever_value_label.setText(f"{v / 1000:.3f}"),
                       self.manual_changed.emit(v / 1000.0)))
        lever_row.addWidget(self.lever, stretch=1)
        self.lever_value_label = QLabel("0.500")
        self.lever_value_label.setStyleSheet(
            "color:#7fd4ff;font-weight:bold;min-width:48px;")
        lever_row.addWidget(self.lever_value_label)
        sv.addLayout(lever_row)

        # ---- decomposition tab = lanes only (pure visualization) -------
        self.decomp = QWidget()
        dv = QVBoxLayout(self.decomp)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.addWidget(self.lanes, stretch=1)

        # ---- graph tab = nodal editor (THE composition authority) ------
        from components.soundlab_nodes import NodeEditor
        self.nodes_editor = NodeEditor()
        self.graph_applied = self.nodes_editor.graph_applied

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            "QTabBar::tab{background:#181c26;color:#8090a8;padding:5px 14px;}"
            "QTabBar::tab:selected{background:#232838;color:#cfd6e4;}")
        self.tabs.addTab(self.signal_tab, "signal")
        self.tabs.addTab(self.decomp, "decomposition")
        self.tabs.addTab(self.nodes_editor, "graph")
        layout.addWidget(self.tabs, stretch=1)
        # events list stays for signal/decomposition; the graph tab has
        # its own LIVE/DRAFT readouts instead
        self.tabs.currentChanged.connect(
            lambda i: self.event_log.setVisible(i != 2))

        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumBlockCount(300)   # unbounded doc = slow Qt
        self.event_log.setMaximumHeight(96)
        self.event_log.setStyleSheet(
            "background:#11141b;border:1px solid #232838;"
            "font-family:monospace;font-size:11px;color:#9fb0c8;")
        layout.addWidget(self.event_log)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._repaint_active)
        self._timer.start(80)   # ~12 Hz repaint

    def _repaint_active(self):
        i = self.tabs.currentIndex()
        if i == 0:
            self.chart.update()
        elif i == 1:
            self.lanes.update()
        else:
            self.nodes_editor.tick()

    # ---- manual α lever --------------------------------------------------
    def _style_lever(self, active: bool) -> None:
        """Pill-shaped handle; cyan when armed, grey when the graph
        drives (checkbox unchecked)."""
        knob = "#7fd4ff" if active else "#3a4358"
        self.lever.setStyleSheet(
            "QSlider::groove:horizontal{height:12px;background:#232838;"
            "border-radius:6px;}"
            "QSlider::handle:horizontal{width:52px;height:22px;"
            f"margin:-6px 0;background:{knob};border-radius:11px;}}")

    def _manual_toggled(self, on: bool) -> None:
        self.lever.setEnabled(bool(on))
        self._style_lever(active=bool(on))
        self.knob_changed.emit("audio_alpha_mode",
                               "manual" if on else "policy")
        if on:   # arm at the lever's current position immediately
            self.manual_changed.emit(self.lever.value() / 1000.0)

    # ---- server audio-group knobs (generic mini panel) ------------------
    # knobs that only matter under specific α modes/sources auto-hide
    # (user finding 2026-07-12: band/gain shown while policy drives)
    _KNOB_RELEVANCE = {
        "audio_band": ("audio_alpha_mode", ("band",)),
        "audio_reaction_output_gain": ("audio_alpha_mode",
                                       ("band", "excitation")),
        "lora_blend_ramp_mode": ("audio_source", ("ramp",)),
    }
    def build_audio_controls(self, schema: dict) -> None:
        """Render the server's audio-group knob descriptors (select /
        range / checkbox) into the signal tab. Called on every handshake
        with the current values as defaults; emits knob_changed."""
        for row in (self.audio_row, self.hdr_controls):
            while row.count():
                item = row.takeAt(0)
                wdg = item.widget()
                if wdg is not None:
                    wdg.deleteLater()
        self.audio_controls = {}
        self._audio_labels = {}
        for field, d in sorted(schema.items(),
                               key=lambda kv: kv[1].get("order", 9999)):
            ftype = d.get("field")
            title = str(d.get("title", field)).split("(")[0].strip()
            if ftype == "select":
                lab = QLabel(title)
                lab.setStyleSheet("color:#8090a8;")
                combo = QComboBox()
                combo.addItems([str(o) for o in d.get("options", [])])
                if d.get("default") is not None:
                    combo.setCurrentText(str(d["default"]))
                combo.setStyleSheet("background:#181c26;color:#cfd6e4;")
                combo.currentTextChanged.connect(
                    lambda t, f=field: (self.knob_changed.emit(f, t),
                                        self._apply_knob_relevance()))
                self.audio_row.addWidget(lab)
                self.audio_row.addWidget(combo)
                self.audio_controls[field] = combo
                self._audio_labels[field] = lab
            elif ftype == "checkbox":
                cb = QCheckBox(title)
                cb.setChecked(bool(d.get("default", False)))
                cb.toggled.connect(
                    lambda v, f=field: self.knob_changed.emit(f, bool(v)))
                # soundlab_log lives top-right in the header, tab-agnostic
                (self.hdr_controls if field == "soundlab_log"
                 else self.audio_row).addWidget(cb)
                self.audio_controls[field] = cb
            elif ftype == "range":
                lab = QLabel(title)
                lab.setStyleSheet("color:#8090a8;")
                sp = QDoubleSpinBox()
                sp.setRange(float(d.get("min", 0.0)), float(d.get("max", 1.0)))
                sp.setSingleStep(float(d.get("step", 0.05)))
                sp.setDecimals(2)
                sp.setValue(float(d.get("default", 0.0)))
                sp.setFixedWidth(64)
                sp.setStyleSheet("background:#181c26;color:#cfd6e4;")
                sp.valueChanged.connect(
                    lambda v, f=field: self.knob_changed.emit(f, float(v)))
                self.audio_row.addWidget(lab)
                self.audio_row.addWidget(sp)
                self.audio_controls[field] = sp
                self._audio_labels[field] = lab
        self.audio_row.addStretch()
        self._apply_knob_relevance()

    def _apply_knob_relevance(self) -> None:
        """Hide knobs that the active α mode/source ignores."""
        for field, (dep, allowed) in self._KNOB_RELEVANCE.items():
            wdg = self.audio_controls.get(field)
            if wdg is None:
                continue
            dep_w = self.audio_controls.get(dep)
            cur = dep_w.currentText() if isinstance(dep_w, QComboBox) else ""
            show = cur in allowed
            wdg.setVisible(show)
            lab = self._audio_labels.get(field)
            if lab is not None:
                lab.setVisible(show)

    def update_control(self, field: str, value) -> None:
        """Programmatic set (reconnect restore / mic-found defaults).
        Signals are blocked — the caller pushes to the server itself."""
        wdg = self.audio_controls.get(field)
        if wdg is None:
            return
        wdg.blockSignals(True)
        try:
            if isinstance(wdg, QComboBox):
                wdg.setCurrentText(str(value))
            elif isinstance(wdg, QCheckBox):
                wdg.setChecked(bool(value))
            elif isinstance(wdg, QDoubleSpinBox):
                wdg.setValue(float(value))
        finally:
            wdg.blockSignals(False)
        self._apply_knob_relevance()

    def set_graph_presets(self, presets: dict) -> None:
        """Server-advertised graph presets → the nodal editor."""
        self.nodes_editor.set_presets(presets)

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
        # graph tab: live per-node values + the deck's ACTIVE formula
        self.nodes_editor.update_values(sl.get("graph_vals") or {})
        f = sl.get("formula")
        if f:
            self.nodes_editor.set_active_formula(str(f))
            self.lanes.formula = str(f)
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

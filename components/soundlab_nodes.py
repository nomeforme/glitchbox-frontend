"""Sound Lab nodal editor — TouchDesigner-style patching of the deck
driver. The graph shown here IS the α formula: sources → operators →
OUT, serialized as a dict and shipped to the server via the
`soundlab_graph` knob. Presets (weighted_mix, auto_vox) arrive from the
server's session_ready capabilities and live as YAML config files
server-side (tools/soundlab/configs/graphs/).

Interaction model (deliberately click-click, not drag-drag):
  * drag node body       → move (positions persist into the graph dict)
  * click OUTPUT port    → arms a connection (port highlights)
  * click an INPUT port  → completes it (replaces that slot; variadic
                           ops expose a dashed extra port to append)
  * select node          → parameters appear in the right panel
  * Delete key / button  → removes the selected node and its wires
  * Apply to deck        → validates, ships the patch, formula updates

The bottom label always renders the ACTIVE formula (what actually
drives the deck, from server telemetry), so what you read is what
renders.
"""

import json
import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath,
                           QPen)
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QGraphicsItem, QGraphicsObject,
                               QGraphicsPathItem, QGraphicsScene,
                               QGraphicsView, QHBoxLayout, QLabel,
                               QPushButton, QToolButton, QVBoxLayout,
                               QWidget)

from components.soundlab_docs import (GATE_DOCS, OP_LABELS, SIGNAL_LABELS,
                                      XFORM_LABELS, node_docs)

SOURCE_SIGNALS = ("beat", "down", "onset", "onset_low", "pitch",
                  "drums", "bass", "other", "vocals",
                  # legacy α producers folded in as sources (use norm=False
                  # — they arrive already calibrated 0..1)
                  "level_bass", "level_low_mid", "level_mid",
                  "level_treble", "excite", "ramp")
VARIADIC = ("wsum", "max", "min", "mul", "add")
ARITY = {"source": 0, "voxp": 0, "const": 0, "inv": 1,
         "ema": 1, "stretch": 1, "xfade": 3, "xform": 1,
         "thresh": 1, "punch": 1}
ADDABLE_OPS = ("source", "voxp", "const", "wsum", "max", "min",
               "mul", "add", "inv", "xfade", "ema", "stretch", "xform",
               "thresh", "punch")
XFADE_LABELS = ("a", "b", "mix")

# engine-side transforms reified as inline pass-through nodes:
# kind -> (param key, label, lo, hi, step, decimals)
XFORM_KINDS = {
    "onset": (("max_hz", "band-pass Hz", 500, 20000, 250, 0),
              ("smooth_s", "flux smooth s", 0.02, 0.5, 0.02, 2)),
    "down": (("decay_s", "pulse decay s", 0.1, 2.0, 0.05, 2),),
}
XFORM_DEFAULTS = {"onset": {"max_hz": 8000.0, "smooth_s": 0.08},
                  "down": {"decay_s": 0.35}}

# the noise gate is the one GLOBAL preproc (conditions the mic feed for
# every engine) — rendered as an input caption, edited via empty-canvas
# properties (key, label, lo, hi, step, decimals)
PREPROC_FIELDS = (
    ("gate_stay_db", "gate stay-open dBFS", -60, 0, 1, 0),
    ("gate_min_db", "gate abs-min dBFS", -80, -20, 1, 0),
    ("gate_open_db", "gate open margin dB", 3, 30, 1, 0),
    ("gate_mod_db", "gate modulation dB", 1, 10, 0.5, 1),
)
PREPROC_DEFAULTS = {
    "gate_stay_db": -30.0, "gate_min_db": -55.0,
    "gate_open_db": 10.0, "gate_mod_db": 3.5,
}

# pretty names for the docs section's parameter bullets
PARAM_LABELS = {
    "signal": "signal", "norm": "normalize",
    "peak_hl_s": "norm peak half-life s", "floor_rise_s": "norm floor rise s",
    "attack_s": "attack s", "release_s": "release s",
    "voiced_gain": "voiced gain", "share_ref": "stem share ref",
    "pitch_conf_min": "pitch conf gate", "value": "value",
    "weights": "weights", "tau_s": "tau s", "win_s": "window s",
    "lo_pct": "lo pct → 0", "hi_pct": "hi pct → 1", "min_span": "min span",
    "max_hz": "band-pass Hz", "smooth_s": "flux smooth s",
    "decay_s": "pulse decay s", "window_s": "median window s",
    "k": "threshold factor k",
}

COL_BG = QColor("#0d0f14")
COL_NODE = QColor("#181c26")
COL_NODE_SEL = QColor("#232838")
COL_BORDER = QColor("#2c3550")
COL_OUT = QColor("#7fd4ff")
COL_TEXT = QColor("#cfd6e4")
COL_DIMTX = QColor("#8090a8")
COL_PORT = QColor("#8fdccf")
COL_PORT_ARM = QColor("#ffb75f")
COL_EDGE = QColor("#4a5878")
COL_VAL = QColor("#8fdccf")
OP_COLORS = {"source": QColor("#b48cff"), "voxp": QColor("#7fd4ff"),
             "const": QColor("#8090a8"), "wsum": QColor("#ffb75f"),
             "max": QColor("#ff7f5f"), "min": QColor("#ff7f5f"),
             "mul": QColor("#e0e05f"), "add": QColor("#e0e05f"),
             "inv": QColor("#e0e05f"), "xfade": QColor("#ff4f9f"),
             "ema": QColor("#5fe08f"), "stretch": QColor("#5fe08f"),
             "xform": QColor("#8fdccf"),
             "thresh": QColor("#ffb75f"), "punch": QColor("#ff7f5f")}


# ---------------------------------------------------------------------------
# graph helpers (client-side mirror of core/audio/signal_graph.py)
# ---------------------------------------------------------------------------

def validate_graph(g: dict) -> str:
    """Return '' if valid else a one-line error message."""
    nodes = {n.get("id"): n for n in g.get("nodes", [])}
    if not nodes:
        return "graph has no nodes"
    if len(nodes) != len(g.get("nodes", [])):
        return "duplicate node ids"
    for n in nodes.values():
        op = n.get("op")
        ins = n.get("inputs") or []
        if op not in ADDABLE_OPS:
            return f"{n.get('id')}: unknown op {op!r}"
        ar = ARITY.get(op)
        if ar is not None and len(ins) != ar:
            return f"{n.get('id')}: {op} needs {ar} input(s), has {len(ins)}"
        if ar is None and not ins:
            return f"{n.get('id')}: {op} needs at least one input"
        for i in ins:
            if i not in nodes:
                return f"{n.get('id')}: input {i!r} missing"
        if op == "source" and (n.get("params") or {}).get(
                "signal") not in SOURCE_SIGNALS:
            return f"{n.get('id')}: bad source signal"
    if g.get("out") not in nodes:
        return "no OUT node designated"
    mark = {}

    def visit(nid):
        m = mark.get(nid, 0)
        if m == 1:
            return False
        if m == 2:
            return True
        mark[nid] = 1
        for i in nodes[nid].get("inputs") or []:
            if not visit(i):
                return False
        mark[nid] = 2
        return True

    for nid in nodes:
        if not visit(nid):
            return f"cycle through {nid!r}"
    return ""


def render_formula(g: dict) -> str:
    """Infix rendering, mirroring the server's SignalGraph.formula()."""
    nodes = {n["id"]: n for n in g.get("nodes", [])}
    refs = {nid: 0 for nid in nodes}
    for n in nodes.values():
        for i in n.get("inputs") or []:
            if i in refs:
                refs[i] += 1
    names, lines = {}, []

    def expr(nid, top=False):
        if nid in names:
            return names[nid]
        n = nodes[nid]
        op = n.get("op")
        p = n.get("params") or {}
        ins = n.get("inputs") or []
        if op == "source":
            e = (f"n({p.get('signal')})" if p.get("norm", True)
                 else str(p.get("signal")))
        elif op == "voxp":
            e = "vox"
        elif op == "const":
            e = f"{float(p.get('value', 0.0)):.2f}"
        elif op == "wsum":
            w = [max(0.0, float(x))
                 for x in (p.get("weights") or [1.0] * len(ins))]
            w += [1.0] * (len(ins) - len(w))
            den = sum(w[:len(ins)]) or 1.0
            e = "(" + " + ".join(f"{wi / den:.2f}·{expr(i)}"
                                 for wi, i in zip(w, ins)) + ")"
        elif op in ("max", "min"):
            e = f"{op}(" + ", ".join(expr(i) for i in ins) + ")"
        elif op == "mul":
            e = "(" + "·".join(expr(i) for i in ins) + ")"
        elif op == "add":
            e = "(" + " + ".join(expr(i) for i in ins) + ")"
        elif op == "inv":
            e = f"(1−{expr(ins[0])})"
        elif op == "xform":
            e = expr(ins[0])   # engine-side: transparent in the α math
        elif op == "xfade":
            ce = expr(ins[2])
            e = f"({ce}·{expr(ins[0])} + (1−{ce})·{expr(ins[1])})"
        elif op == "ema":
            e = f"ema({expr(ins[0])}, {float(p.get('tau_s', 0.12)):.2g}s)"
        elif op == "thresh":
            e = (f"thr({expr(ins[0])} − {float(p.get('k', 1.0)):.2g}"
                 f"·med{float(p.get('window_s', 1.0)):.2g}s)")
        elif op == "punch":
            e = (f"punch({expr(ins[0])}, "
                 f"↘{float(p.get('release_s', 0.25)):.2g}s)")
        elif op == "stretch":
            e = (f"stretch({expr(ins[0])}, "
                 f"p{float(p.get('lo_pct', 15.0)):.0f}→0, "
                 f"p{float(p.get('hi_pct', 97.0)):.0f}→1)")
        else:
            e = str(nid)
        if refs.get(nid, 0) > 1 and not top and op not in ("source",
                                                           "voxp", "const"):
            nm = str(nid).upper()
            names[nid] = nm
            lines.append(f"{nm} := {e}")
            return nm
        return e

    try:
        out_e = expr(g.get("out"), top=True)
    except (KeyError, IndexError, TypeError):
        return "(incomplete patch)"
    lines.append(f"α = {out_e}")
    return "   ;   ".join(lines)


# ---------------------------------------------------------------------------
# scene items
# ---------------------------------------------------------------------------

class NodeItem(QGraphicsObject):
    W = 148
    ROW = 17

    def __init__(self, editor, node: dict, is_out: bool):
        super().__init__()
        self.editor = editor
        self.node = node
        self.is_out = is_out
        self.value = None
        n_in = self._n_in_ports()
        self.H = 40 + self.ROW * max(n_in, 1)
        self.setFlags(QGraphicsItem.ItemIsMovable
                      | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(1)

    # ports ---------------------------------------------------------------
    def _n_in_ports(self) -> int:
        op = self.node["op"]
        ar = ARITY.get(op)
        if ar is not None:
            return ar
        return len(self.node.get("inputs") or []) + 1   # +1 dashed appender

    def in_port_pos(self, i: int) -> QPointF:
        return QPointF(0, 34 + self.ROW * i)

    def out_port_pos(self) -> QPointF:
        return QPointF(self.W, self.H / 2)

    def port_at(self, pos: QPointF):
        """('in', i) / ('out', 0) / None for a local click position."""
        if (pos - self.out_port_pos()).manhattanLength() < 12:
            return ("out", 0)
        for i in range(self._n_in_ports()):
            if (pos - self.in_port_pos(i)).manhattanLength() < 12:
                return ("in", i)
        return None

    # geometry / paint ------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(-8, -6, self.W + 16, self.H + 12)

    def paint(self, p: QPainter, _o, _w):
        op = self.node["op"]
        accent = OP_COLORS.get(op, COL_TEXT)
        body = QRectF(0, 0, self.W, self.H)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(COL_NODE_SEL if self.isSelected() else COL_NODE))
        pen = QPen(COL_OUT if self.is_out else
                   (accent if self.isSelected() else COL_BORDER),
                   2 if (self.is_out or self.isSelected()) else 1.2)
        p.setPen(pen)
        p.drawRoundedRect(body, 6, 6)
        # title
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        p.setPen(accent)
        prm0 = self.node.get("params") or {}
        if op == "source":
            human = SIGNAL_LABELS.get(prm0.get("signal"),
                                      str(prm0.get("signal")))
        elif op == "xform":
            human = XFORM_LABELS.get(prm0.get("kind"), "transform")
        else:
            human = OP_LABELS.get(op, op)
        title = human
        if self.is_out:
            title += "  →deck"
        p.drawText(QRectF(6, 2, self.W - 12, 14), Qt.AlignLeft, title)
        f2 = QFont()
        f2.setPointSize(8)
        p.setFont(f2)
        p.setPen(COL_DIMTX)
        sub = op
        prm = self.node.get("params") or {}
        if op == "source":
            sub = f"source:{prm.get('signal')}"
        elif op == "xform":
            kind = prm.get("kind", "?")
            if kind == "onset":
                sub = (f"engine: BP<{float(prm.get('max_hz', 8000)) / 1000:.1f}"
                       f"kHz · ema{float(prm.get('smooth_s', 0.08)) * 1000:.0f}ms")
            elif kind == "down":
                sub = f"engine: pulse τ{float(prm.get('decay_s', 0.35)):.2f}s"
            else:
                sub = f"xform:{kind}"
        elif op == "voxp":
            sub = (f"voxp · conf≥{float(prm.get('pitch_conf_min', 0.30)):.2f}"
                   f" [pesto]")
        elif op == "thresh":
            sub = (f"−{float(prm.get('k', 1.0)):.2g}·med"
                   f"{float(prm.get('window_s', 1.0)):.2g}s")
        elif op == "punch":
            sub = (f"atk{float(prm.get('attack_s', 0.01)) * 1000:.0f}ms · "
                   f"rel{float(prm.get('release_s', 0.25)):.2f}s")
        elif op == "const":
            sub = f"const {float(prm.get('value', 0.0)):.2f}"
        elif op == "ema":
            sub = f"ema τ={float(prm.get('tau_s', 0.12)):.2g}s"
        sub = f"{self.node['id']} · {sub}"   # id stays visible (formula refs)
        p.drawText(QRectF(6, 15, self.W - 12, 13), Qt.AlignLeft, sub)
        # live value
        if self.value is not None:
            p.setPen(COL_VAL)
            p.drawText(QRectF(6, self.H - 16, self.W - 12, 14),
                       Qt.AlignRight, f"{self.value:.2f}")
        # ports
        arm = self.editor.pending
        n_ports = self._n_in_ports()
        ins = self.node.get("inputs") or []
        for i in range(n_ports):
            c = self.in_port_pos(i)
            dashed = (ARITY.get(op) is None and i == n_ports - 1)
            p.setPen(QPen(COL_PORT, 1, Qt.DashLine if dashed
                          else Qt.SolidLine))
            p.setBrush(QBrush(COL_BG if dashed else COL_PORT))
            p.drawEllipse(c, 5, 5)
            if op == "xfade":
                p.setPen(COL_DIMTX)
                p.drawText(QPointF(c.x() + 8, c.y() + 4), XFADE_LABELS[i])
            elif op == "wsum" and i < len(ins):
                w = (prm.get("weights") or [])
                if i < len(w):
                    p.setPen(COL_DIMTX)
                    p.drawText(QPointF(c.x() + 8, c.y() + 4),
                               f"w={float(w[i]):.2f}")
        armed = (arm is not None and arm[0] is self)
        p.setPen(QPen(COL_PORT_ARM if armed else COL_PORT, 1))
        p.setBrush(QBrush(COL_PORT_ARM if armed else COL_PORT))
        p.drawEllipse(self.out_port_pos(), 6 if armed else 5,
                      6 if armed else 5)

    # interaction -----------------------------------------------------------
    def mousePressEvent(self, ev):
        port = self.port_at(ev.pos())
        if port is not None:
            self.editor.port_clicked(self, port)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.node["pos"] = [round(self.pos().x(), 1),
                                round(self.pos().y(), 1)]
            self.editor.update_edges()
        return super().itemChange(change, value)


class EdgeItem(QGraphicsPathItem):
    def __init__(self, src: NodeItem, dst: NodeItem, slot: int):
        super().__init__()
        self.src, self.dst, self.slot = src, dst, slot
        self.setZValue(0)
        self.setPen(QPen(COL_EDGE, 1.6))
        self.adjust()

    def adjust(self):
        a = self.src.mapToScene(self.src.out_port_pos())
        b = self.dst.mapToScene(self.dst.in_port_pos(self.slot))
        path = QPainterPath(a)
        dx = max(abs(b.x() - a.x()) * 0.5, 30)
        path.cubicTo(a + QPointF(dx, 0), b - QPointF(dx, 0), b)
        self.setPath(path)


# ---------------------------------------------------------------------------
# the editor widget
# ---------------------------------------------------------------------------

class NodeEditor(QWidget):
    """Graph tab: scene + toolbar + properties + formula readout."""

    graph_applied = Signal(str)     # json.dumps(graph) -> soundlab_graph knob

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gdict = {"name": "custom", "out": "", "nodes": [],
                      "preproc": dict(PREPROC_DEFAULTS)}
        self.items = {}          # node id -> NodeItem
        self.edges = []
        self.pending = None      # (NodeItem, ('out', 0)) armed connection
        self.presets = {}
        self._live_vals = {}
        self._sel_id = None
        self._n_new = 0
        # deck-sync baseline: canonical JSON of the last APPLIED patch
        # (positions stripped — moving nodes isn't a semantic edit)
        self._applied_json = None
        # pristine-load baseline: what the draft looked like right after
        # the last load/apply. Preset browsing compares against THIS —
        # a freshly loaded (unapplied) preset is not "unsaved changes"
        # (live bug 2026-07-12: dropdown locked after loading a preset).
        self._loaded_canon = None
        self._loading = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # toolbar ----------------------------------------------------------
        bar = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(130)
        # selecting a preset LOADS it immediately when the draft has no
        # unsaved changes; the button remains as the explicit
        # discard-my-edits path
        self.preset_combo.currentTextChanged.connect(self._on_preset_pick)
        load_btn = QPushButton("Load (discard edits)")
        load_btn.clicked.connect(self._load_selected_preset)
        self.add_combo = QComboBox()
        for _op in ADDABLE_OPS:
            self.add_combo.addItem(f"{OP_LABELS.get(_op, _op)} ({_op})",
                                   userData=_op)
        add_btn = QPushButton("Add node")
        add_btn.clicked.connect(self._add_node)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_selected)
        self.apply_btn = QPushButton("Apply → deck")
        self.apply_btn.setStyleSheet(
            "background:#1d3a4a;color:#7fd4ff;font-weight:bold;"
            "padding:4px 14px;")
        self.apply_btn.clicked.connect(self._apply)
        for w in (QLabel("preset"), self.preset_combo, load_btn,
                  QLabel("  op"), self.add_combo, add_btn, del_btn):
            if isinstance(w, QLabel):
                w.setStyleSheet("color:#8090a8;")
            bar.addWidget(w)
        bar.addStretch()
        self.hint = QLabel("click an output port, then an input port "
                           "to patch")
        self.hint.setStyleSheet("color:#5a6680;font-size:10px;")
        bar.addWidget(self.hint)
        bar.addWidget(self.apply_btn)
        root.addLayout(bar)

        # scene + properties -------------------------------------------------
        mid = QHBoxLayout()
        self.scene = QGraphicsScene()
        self.scene.setBackgroundBrush(QBrush(COL_BG))
        self.scene.selectionChanged.connect(self._on_selection)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setStyleSheet("border:1px solid #232838;")
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        mid.addWidget(self.view, stretch=1)
        from PySide6.QtWidgets import QScrollArea
        self.props = QWidget()
        self.props_lay = QVBoxLayout(self.props)
        self.props_lay.setContentsMargins(6, 2, 2, 2)
        scroll = QScrollArea()
        scroll.setWidget(self.props)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(205)
        scroll.setStyleSheet("border:1px solid #232838;")
        mid.addWidget(scroll)
        root.addLayout(mid, stretch=1)

        # LIVE (deck truth, from server telemetry) vs DRAFT (this canvas)
        # — visually distinct so it's unambiguous what is rendering
        self.deck_lbl = QLabel("")
        self.deck_lbl.setWordWrap(True)
        self.deck_lbl.setStyleSheet(
            "color:#7fd4ff;font-family:monospace;font-size:11px;"
            "background:#0f2430;border-left:4px solid #7fd4ff;padding:4px;")
        root.addWidget(self.deck_lbl)
        self.draft_lbl = QLabel("")
        self.draft_lbl.setWordWrap(True)
        root.addWidget(self.draft_lbl)
        # patch action log: param edits, patching, preset loads, applies
        # (mirrors the events log, but for what YOU changed and whether
        # it is live yet)
        from PySide6.QtWidgets import QPlainTextEdit
        self.action_log = QPlainTextEdit()
        self.action_log.setReadOnly(True)
        self.action_log.setMaximumBlockCount(200)
        self.action_log.setFixedHeight(64)
        self.action_log.setStyleSheet(
            "background:#11141b;border:1px solid #232838;"
            "font-family:monospace;font-size:10px;color:#9fb0c8;")
        root.addWidget(self.action_log)
        self._active_formula = ""
        self._refresh_formula()

    # ---- action log / dirty tracking ------------------------------------
    def _log_action(self, msg: str) -> None:
        import time as _t
        self.action_log.appendPlainText(f"[{_t.strftime('%H:%M:%S')}] {msg}")

    @staticmethod
    def _canon(g: dict) -> str:
        gg = json.loads(json.dumps(g))
        for n in gg.get("nodes", []):
            n.pop("pos", None)      # layout is not a semantic edit
        gg.pop("name", None)
        return json.dumps(gg, sort_keys=True)

    def _dirty(self) -> bool:
        """True when the draft differs from the last APPLIED patch
        (drives the DRAFT/LIVE bar)."""
        if self._applied_json is None:
            d = render_formula(self.gdict) if self.gdict.get("nodes") else ""
            return not (self._active_formula
                        and d == self._active_formula)
        return self._canon(self.gdict) != self._applied_json

    def _modified_since_load(self) -> bool:
        """True only when the USER edited the draft after the last
        load/apply — the guard for preset switching (a freshly loaded
        preset has no work to lose)."""
        if self._loaded_canon is None:
            return False
        return self._canon(self.gdict) != self._loaded_canon

    # ---- public API -------------------------------------------------------
    def set_presets(self, presets: dict) -> None:
        self.presets = dict(presets or {})
        cur = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(sorted(self.presets))
        if cur in self.presets:
            self.preset_combo.setCurrentText(cur)
        self.preset_combo.blockSignals(False)
        # first delivery: show the server's default patch — this IS what
        # the deck runs at handshake, so it doubles as the sync baseline
        if not self.gdict.get("nodes") and self.presets:
            name = ("auto_vox" if "auto_vox" in self.presets
                    else sorted(self.presets)[0])
            self._loading = True
            try:
                self.preset_combo.setCurrentText(name)
            finally:
                self._loading = False
            self.load_graph(self.presets[name])
            self._applied_json = self._canon(self.gdict)
            self._loaded_preset = name
            self._log_action(f"loaded preset '{name}' (deck default)")
            self._refresh_formula()

    def _on_preset_pick(self, name: str) -> None:
        """Dropdown selection loads immediately — unless there are
        unsaved edits (then the explicit discard button is required)."""
        if self._loading or name not in self.presets:
            return
        if self.gdict.get("nodes") and self._modified_since_load():
            self._log_action(
                f"preset '{name}' selected but draft has unsaved edits "
                "— press 'Load (discard edits)' to replace it")
            # revert the combo so it never claims a patch it didn't load
            prev = getattr(self, "_loaded_preset", None)
            if prev and prev in self.presets:
                self._loading = True
                try:
                    self.preset_combo.setCurrentText(prev)
                finally:
                    self._loading = False
            return
        self.load_graph(self.presets[name])
        self._loaded_preset = name
        self._log_action(f"loaded preset '{name}' — press Apply → deck "
                         "to make it live")

    def load_graph(self, g: dict) -> None:
        self.gdict = json.loads(json.dumps(g))   # deep copy
        self.gdict.setdefault("preproc", {})
        self.pending = None
        self._loaded_canon = self._canon(self.gdict)   # pristine baseline
        self._rebuild_scene()
        self._refresh_formula()

    def current_graph(self) -> dict:
        return self.gdict

    def update_values(self, vals: dict) -> None:
        self._live_vals = vals or {}

    def set_active_formula(self, s: str) -> None:
        if s and s != self._active_formula:
            self._active_formula = s
            self._refresh_formula()

    def tick(self) -> None:
        """Called by the repaint timer while the tab is visible."""
        for nid, it in self.items.items():
            v = self._live_vals.get(nid)
            if v != it.value:
                it.value = v
                it.update()

    # ---- scene management -------------------------------------------------
    def _rebuild_scene(self) -> None:
        self.scene.clear()
        self.items, self.edges = {}, []
        nodes = self.gdict.get("nodes") or []
        depth = self._depths(nodes)
        for k, n in enumerate(nodes):
            item = NodeItem(self, n, n["id"] == self.gdict.get("out"))
            pos = n.get("pos")
            if not pos:
                d = depth.get(n["id"], 0)
                row = sum(1 for m in nodes[:k]
                          if depth.get(m["id"], 0) == d)
                pos = [40 + d * 200, 40 + row * 95]
                n["pos"] = pos
            item.setPos(float(pos[0]), float(pos[1]))
            self.scene.addItem(item)
            self.items[n["id"]] = item
        for n in nodes:
            for slot, src in enumerate(n.get("inputs") or []):
                if src in self.items:
                    e = EdgeItem(self.items[src], self.items[n["id"]], slot)
                    self.scene.addItem(e)
                    self.edges.append(e)
        self.scene.setSceneRect(self.scene.itemsBoundingRect()
                                .adjusted(-60, -40, 120, 80))

    @staticmethod
    def _depths(nodes):
        by_id = {n["id"]: n for n in nodes}
        memo = {}

        def d(nid, seen=()):
            if nid in memo:
                return memo[nid]
            if nid in seen or nid not in by_id:
                return 0
            ins = by_id[nid].get("inputs") or []
            memo[nid] = 0 if not ins else 1 + max(
                d(i, seen + (nid,)) for i in ins)
            return memo[nid]

        return {n["id"]: d(n["id"]) for n in nodes}

    def update_edges(self) -> None:
        for e in self.edges:
            e.adjust()

    # ---- patching ---------------------------------------------------------
    def port_clicked(self, item: NodeItem, port) -> None:
        kind, slot = port
        if kind == "out":
            self.pending = (item, port)
            self.hint.setText(f"connecting {item.node['id']} → click an "
                              "input port (Esc cancels)")
            self.scene.update()
            return
        if self.pending is None:
            return
        src_item, _ = self.pending
        self.pending = None
        self.hint.setText("click an output port, then an input port to patch")
        if src_item is item:
            return
        n = item.node
        ins = list(n.get("inputs") or [])
        op = n["op"]
        if ARITY.get(op) is not None:
            while len(ins) < ARITY[op]:
                ins.append(ins[-1] if ins else src_item.node["id"])
            ins[slot] = src_item.node["id"]
        else:
            if slot < len(ins):
                ins[slot] = src_item.node["id"]
            else:
                ins.append(src_item.node["id"])
                if op == "wsum":
                    p = n.setdefault("params", {})
                    w = list(p.get("weights") or [])
                    w += [1.0] * (len(ins) - 1 - len(w)) + [0.5]
                    p["weights"] = w[:len(ins)]
        n["inputs"] = ins
        self._log_action(f"patched {src_item.node['id']} → {n['id']}"
                         "   (draft)")
        self._rebuild_scene()
        self._refresh_formula()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape and self.pending is not None:
            self.pending = None
            self.hint.setText("click an output port, then an input port "
                              "to patch")
            self.scene.update()
        elif ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected()
        else:
            super().keyPressEvent(ev)

    # ---- toolbar actions ----------------------------------------------------
    def _load_selected_preset(self):
        name = self.preset_combo.currentText()
        if name in self.presets:
            self.load_graph(self.presets[name])
            self._loaded_preset = name
            self._log_action(f"loaded preset '{name}' (edits discarded) — "
                             "press Apply → deck to make it live")

    def _add_node(self):
        op = self.add_combo.currentData() or self.add_combo.currentText()
        self._n_new += 1
        nid = f"{op}{self._n_new}"
        while any(n["id"] == nid for n in self.gdict["nodes"]):
            self._n_new += 1
            nid = f"{op}{self._n_new}"
        node = {"id": nid, "op": op, "inputs": [], "params": {}}
        if op == "source":
            node["params"] = {"signal": "onset", "norm": True}
        elif op == "voxp":
            node["params"] = {"attack_s": 0.7, "release_s": 2.5}
        elif op == "const":
            node["params"] = {"value": 0.5}
        elif op == "ema":
            node["params"] = {"tau_s": 0.12}
        elif op == "stretch":
            node["params"] = {"win_s": 30.0, "lo_pct": 15.0,
                              "hi_pct": 97.0, "min_span": 0.15}
        elif op == "wsum":
            node["params"] = {"weights": []}
        elif op == "xform":
            node["params"] = {"kind": "onset", **XFORM_DEFAULTS["onset"]}
        elif op == "thresh":
            node["params"] = {"window_s": 1.0, "k": 1.0}
        elif op == "punch":
            node["params"] = {"attack_s": 0.01, "release_s": 0.25}
        vp = self.view.mapToScene(self.view.viewport().rect().center())
        node["pos"] = [vp.x() - 70, vp.y() - 30]
        self.gdict["nodes"].append(node)
        if not self.gdict.get("out"):
            self.gdict["out"] = nid
        self._log_action(f"added node {nid} ({op})   (draft)")
        self._rebuild_scene()
        self._refresh_formula()

    def _delete_selected(self):
        sel = [it for it in self.scene.selectedItems()
               if isinstance(it, NodeItem)]
        if not sel:
            return
        dead = {it.node["id"] for it in sel}
        if self.gdict.get("out") in dead:
            self.hint.setText("cannot delete the OUT node — set another "
                              "output first")
            return
        self.gdict["nodes"] = [n for n in self.gdict["nodes"]
                               if n["id"] not in dead]
        for n in self.gdict["nodes"]:
            old = n.get("inputs") or []
            kept = [(i, s) for i, s in enumerate(old) if s not in dead]
            n["inputs"] = [s for _, s in kept]
            if n["op"] == "wsum":
                w = (n.get("params") or {}).get("weights") or []
                n.setdefault("params", {})["weights"] = [
                    w[i] if i < len(w) else 1.0 for i, _ in kept]
        self._log_action(f"deleted {', '.join(sorted(dead))}   (draft)")
        self._rebuild_scene()
        self._refresh_formula()

    def _apply(self):
        err = validate_graph(self.gdict)
        if err:
            self.draft_lbl.setStyleSheet(
                "color:#ff7f5f;font-family:monospace;font-size:11px;"
                "background:#241114;border-left:4px solid #ff7f5f;"
                "padding:4px;")
            self.draft_lbl.setText(f"● DRAFT   ✗ invalid — {err}")
            return
        self.gdict["name"] = "custom"
        self.graph_applied.emit(json.dumps(self.gdict))
        self._applied_json = self._canon(self.gdict)
        self._loaded_canon = self._applied_json   # applied = nothing to lose
        self._log_action("APPLIED → deck (patch is live)")
        self._refresh_formula()

    # ---- properties panel ------------------------------------------------
    def _on_selection(self):
        try:   # fires during scene teardown at shutdown (C++ side gone)
            sel = [it for it in self.scene.selectedItems()
                   if isinstance(it, NodeItem)]
        except RuntimeError:
            return
        self._sel_id = sel[0].node["id"] if len(sel) == 1 else None
        self._build_props()

    def _clear_props(self):
        while self.props_lay.count():
            item = self.props_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _prop_spin(self, label, val, lo, hi, step, setter, decimals=2,
                   doc=None, owner=None):
        lab = QLabel(label)
        lab.setStyleSheet("color:#8090a8;font-size:10px;")
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.setValue(float(val))
        sp.setStyleSheet("background:#181c26;color:#cfd6e4;")
        prev = {"v": float(val)}
        who = owner or self._sel_id or "?"

        def _on(v):
            setter(float(v))
            self._log_action(f"{who} · {label}: {prev['v']:.3g} → {v:.3g}"
                             "   (draft — Apply → deck to take effect)")
            prev["v"] = float(v)
            self._node_repaint()
            self._refresh_formula()
        sp.valueChanged.connect(_on)
        if doc:
            lab.setToolTip(doc)
            sp.setToolTip(doc)
        self.props_lay.addWidget(lab)
        self.props_lay.addWidget(sp)

    def _add_docs_section(self, what, param_docs, fields=None,
                          param_labels=None):
        """Collapsible 'docs' block: what the node does + the so-what of
        each parameter, in plain terms. Params also get tooltips."""
        btn = QToolButton()
        btn.setText("▸ docs")
        btn.setCheckable(True)
        btn.setStyleSheet(
            "QToolButton{color:#8fdccf;background:transparent;border:none;"
            "font-size:10px;font-weight:bold;padding:2px;}")
        body = QLabel()
        body.setWordWrap(True)
        body.setVisible(False)
        body.setStyleSheet(
            "color:#9fb0c8;font-size:10px;background:#12161f;"
            "border-left:2px solid #2c3550;padding:4px;")
        names = dict(param_labels or {})
        if fields:   # (key, label, ...) tuples -> pretty names
            for f in fields:
                names.setdefault(f[0], f[1])
        lines = [what or ""]
        for k, d in (param_docs or {}).items():
            lines.append(f"• {names.get(k, k)} — {d}")
        body.setText("\n\n".join([lines[0]] + ["\n".join(lines[1:])])
                     if len(lines) > 1 else lines[0])

        def _toggle(on):
            btn.setText("▾ docs" if on else "▸ docs")
            body.setVisible(on)
        btn.toggled.connect(_toggle)
        self.props_lay.addWidget(btn)
        self.props_lay.addWidget(body)

    def _node_repaint(self):
        it = self.items.get(self._sel_id)
        if it is not None:
            it.update()

    def _build_props(self):
        self._clear_props()
        title = QLabel("preproc / node")
        title.setStyleSheet("color:#cfd6e4;font-weight:bold;")
        self.props_lay.addWidget(title)
        node = next((n for n in self.gdict["nodes"]
                     if n["id"] == self._sel_id), None)
        if node is None:
            # nothing selected: the preproc chain (every implicit
            # transform upstream of the graph, reified as patch params)
            title.setText("preproc (mic → gate → engines)")
            pp = self.gdict.setdefault("preproc", {})
            for key, label, lo, hi, step, dec in PREPROC_FIELDS:
                self._prop_spin(
                    label, pp.get(key, PREPROC_DEFAULTS[key]), lo, hi, step,
                    lambda v, k=key: pp.__setitem__(k, v), dec,
                    doc=GATE_DOCS["params"].get(key), owner="noise gate")
            self._add_docs_section(GATE_DOCS["what"],
                                   GATE_DOCS["params"], PREPROC_FIELDS)
            note = QLabel("ships with the patch on Apply · select a node "
                          "to edit its params")
            note.setStyleSheet("color:#5a6680;font-size:10px;")
            note.setWordWrap(True)
            self.props_lay.addWidget(note)
            self.props_lay.addStretch()
            return
        op = node["op"]
        p = node.setdefault("params", {})
        if op == "source":
            human = SIGNAL_LABELS.get(p.get("signal"), str(p.get("signal")))
        elif op == "xform":
            human = XFORM_LABELS.get(p.get("kind"), "transform")
        else:
            human = OP_LABELS.get(op, op)
        title.setText(f"{human}\n{node['id']}  [{op}]")
        ndoc_what, ndoc_params = node_docs(op, p)
        if op == "source":
            lab = QLabel("signal")
            lab.setStyleSheet("color:#8090a8;font-size:10px;")
            combo = QComboBox()
            combo.addItems(list(SOURCE_SIGNALS))
            combo.setCurrentText(str(p.get("signal", "onset")))
            combo.setStyleSheet("background:#181c26;color:#cfd6e4;")
            combo.currentTextChanged.connect(
                lambda t: (p.__setitem__("signal", t), self._node_repaint(),
                           self._refresh_formula()))
            cb = QCheckBox("normalize (floor+peak)")
            cb.setChecked(bool(p.get("norm", True)))
            cb.setStyleSheet("color:#8090a8;font-size:10px;")
            cb.toggled.connect(
                lambda v: (p.__setitem__("norm", bool(v)),
                           self._refresh_formula()))
            self.props_lay.addWidget(lab)
            self.props_lay.addWidget(combo)
            self.props_lay.addWidget(cb)
            # the norm's time constants, reified (were hidden globals)
            self._prop_spin("norm peak half-life s",
                            p.get("peak_hl_s", 69.0), 5, 300, 5,
                            lambda v: p.__setitem__("peak_hl_s", v), 0)
            self._prop_spin("norm floor rise s",
                            p.get("floor_rise_s", 50.0), 5, 300, 5,
                            lambda v: p.__setitem__("floor_rise_s", v), 0)
        elif op == "voxp":
            self._prop_spin("attack s", p.get("attack_s", 0.7), 0.05, 5, 0.1,
                            lambda v: p.__setitem__("attack_s", v))
            self._prop_spin("release s", p.get("release_s", 2.5), 0.05, 10,
                            0.25, lambda v: p.__setitem__("release_s", v))
            # presence target = min(1, gain·voiced) · min(1, share/ref)
            # (was hardcoded 1.6 / 0.12 in the runtime)
            self._prop_spin("voiced gain", p.get("voiced_gain", 1.6),
                            0.5, 4.0, 0.1,
                            lambda v: p.__setitem__("voiced_gain", v))
            self._prop_spin("stem share ref", p.get("share_ref", 0.12),
                            0.02, 0.6, 0.01,
                            lambda v: p.__setitem__("share_ref", v))
            # PESTO voicing gate (engine-side, applied on Apply)
            self._prop_spin("pitch conf gate",
                            p.get("pitch_conf_min", 0.30), 0.05, 0.9, 0.05,
                            lambda v: p.__setitem__("pitch_conf_min", v))
        elif op == "xform":
            kind = str(p.get("kind", "onset"))
            lab = QLabel("engine transform")
            lab.setStyleSheet("color:#8090a8;font-size:10px;")
            kc = QComboBox()
            kc.addItems(list(XFORM_KINDS))
            kc.setCurrentText(kind)
            kc.setStyleSheet("background:#181c26;color:#cfd6e4;")

            def _set_kind(k, pp=p, nid=node["id"]):
                pp.clear()
                pp.update({"kind": k, **XFORM_DEFAULTS[k]})
                self._log_action(f"{nid} · kind → {k}   (draft)")
                self._node_repaint()
                self._refresh_formula()
                self._build_props()
            kc.currentTextChanged.connect(_set_kind)
            self.props_lay.addWidget(lab)
            self.props_lay.addWidget(kc)
            for key, label, lo, hi, step, dec in XFORM_KINDS.get(kind, ()):
                self._prop_spin(
                    label, p.get(key, XFORM_DEFAULTS[kind][key]),
                    lo, hi, step, lambda v, k=key: p.__setitem__(k, v), dec)
        elif op == "thresh":
            self._prop_spin("median window s", p.get("window_s", 1.0),
                            0.25, 8.0, 0.25,
                            lambda v: p.__setitem__("window_s", v))
            self._prop_spin("threshold factor k", p.get("k", 1.0),
                            0.25, 3.0, 0.05,
                            lambda v: p.__setitem__("k", v))
        elif op == "punch":
            self._prop_spin("attack s", p.get("attack_s", 0.01),
                            0.001, 0.5, 0.01,
                            lambda v: p.__setitem__("attack_s", v), 3)
            self._prop_spin("release s", p.get("release_s", 0.25),
                            0.05, 2.0, 0.05,
                            lambda v: p.__setitem__("release_s", v))
        elif op == "const":
            self._prop_spin("value", p.get("value", 0.5), 0, 1, 0.05,
                            lambda v: p.__setitem__("value", v))
        elif op == "wsum":
            ins = node.get("inputs") or []
            w = list(p.get("weights") or [])
            w += [1.0] * (len(ins) - len(w))
            p["weights"] = w
            for i, src in enumerate(ins):
                self._prop_spin(f"w · {src}", w[i], 0, 1, 0.05,
                                lambda v, j=i: p["weights"].__setitem__(j, v))
        elif op == "ema":
            self._prop_spin("tau s", p.get("tau_s", 0.12), 0.01, 5, 0.02,
                            lambda v: p.__setitem__("tau_s", v))
        elif op == "stretch":
            self._prop_spin("window s", p.get("win_s", 30.0), 5, 120, 5,
                            lambda v: p.__setitem__("win_s", v), 0)
            self._prop_spin("lo pct → 0", p.get("lo_pct", 15.0), 0, 49, 1,
                            lambda v: p.__setitem__("lo_pct", v), 0)
            self._prop_spin("hi pct → 1", p.get("hi_pct", 97.0), 51, 100, 1,
                            lambda v: p.__setitem__("hi_pct", v), 0)
            self._prop_spin("min span", p.get("min_span", 0.15), 0.05, 1,
                            0.05, lambda v: p.__setitem__("min_span", v))
        self._add_docs_section(ndoc_what, ndoc_params,
                               param_labels=PARAM_LABELS)
        if node["id"] != self.gdict.get("out"):
            btn = QPushButton("set as OUT (→deck)")
            btn.setStyleSheet("background:#1d3a4a;color:#7fd4ff;")
            btn.clicked.connect(lambda: self._set_out(node["id"]))
            self.props_lay.addWidget(btn)
        self.props_lay.addStretch()

    def _set_out(self, nid: str):
        self.gdict["out"] = nid
        self._log_action(f"OUT → {nid}   (draft)")
        self._rebuild_scene()
        self._refresh_formula()
        self._build_props()

    # ---- formula ---------------------------------------------------------
    def _refresh_formula(self):
        self.deck_lbl.setText(
            f"● LIVE   {self._active_formula}" if self._active_formula
            else "● LIVE   (waiting for deck telemetry…)")
        draft = render_formula(self.gdict) if self.gdict.get("nodes") \
            else "(empty patch — load a preset)"
        # JSON-level dirty check: param-only edits (e.g. voxp release_s)
        # don't change the formula but DO need an Apply — the formula
        # comparison alone missed that (live finding 2026-07-12)
        in_sync = self.gdict.get("nodes") and not self._dirty()
        if in_sync:
            self.draft_lbl.setStyleSheet(
                "color:#5a6680;font-family:monospace;font-size:11px;"
                "background:#11141b;border-left:4px solid #2c3550;"
                "padding:4px;")
            self.draft_lbl.setText("● DRAFT   = live (in sync)")
            self.apply_btn.setStyleSheet(
                "background:#1d3a4a;color:#7fd4ff;font-weight:bold;"
                "padding:4px 14px;")
        else:
            self.draft_lbl.setStyleSheet(
                "color:#ffb75f;font-family:monospace;font-size:11px;"
                "background:#221a10;border-left:4px solid #ffb75f;"
                "padding:4px;")
            self.draft_lbl.setText(f"● DRAFT   {draft}")
            # unapplied changes: the Apply button itself carries the cue
            self.apply_btn.setStyleSheet(
                "background:#7a4a12;color:#ffd9a0;font-weight:bold;"
                "padding:4px 14px;")

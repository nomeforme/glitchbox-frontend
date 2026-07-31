from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QCheckBox, QLabel, QLineEdit,
                             QScrollArea, QFrame, QComboBox, QPushButton)
from PySide6.QtCore import Qt, Signal

# Minimum heights for controls
SLIDER_MIN_HEIGHT = 40
CHECKBOX_MIN_HEIGHT = 30
TEXT_INPUT_MIN_HEIGHT = 40

class ControlPanel(QWidget):
    """Widget containing pipeline controls and settings"""

    parameter_changed = Signal(str, object)  # (param_id, value)
    # Realtime-protocol live-knob signal: emitted only for fields the
    # server marked `live_adjustable` in its session_ready capabilities.
    # `main.py` wires this to `WSClient.update_knob`.
    knob_changed = Signal(str, object)       # (field, value)
    # LoRA hot-swap request: (slug_a, slug_b, weight_a, weight_b). Emitted
    # on the Load button; main.py wires it to WSClient.swap_lora.
    lora_swap_requested = Signal(str, str, float, float)
    # LoRA enable/disable toggle (heavy swap-class server operation — NOT a
    # live knob). main.py wires it to WSClient.set_lora_enabled.
    lora_toggle_changed = Signal(bool)
    # Client-side camera crop toggle ("Match Server Aspect") — injected by
    # main.py into the schema so it renders among the server controls, but
    # it never touches the server. main.py wires it to the camera thread.
    aspect_toggle_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.controls = {}
        # param_id of the lora_swap control, if one was rendered. Lets
        # set_lora_selection() find it without guessing at tuple shapes.
        self._lora_param_id = None
        # Capabilities map from the realtime server's session_ready ack.
        # Populated by `apply_capabilities`; empty dict = no classification
        # known yet (= legacy behavior: every field is freely adjustable).
        self._capabilities = {"live_adjustable": [], "frozen": []}

        # Create outer layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setMinimumHeight(300)
        self.scroll_area.setMaximumHeight(500)

        # Create content widget for scroll area
        self.content_widget = QWidget()
        self.main_layout = QVBoxLayout(self.content_widget)
        self.main_layout.setSpacing(8)

        self.scroll_area.setWidget(self.content_widget)
        outer_layout.addWidget(self.scroll_area)

    def setup_pipeline_options(self, settings):
        """Setup UI controls based on received pipeline settings"""
        # Clear existing controls
        self.clear_layout(self.main_layout)
        self.controls = {}
        
        # Extract parameters from settings
        params = settings.get('input_params', {}).get('properties', {})
        default_values = {}

        # Get default values from server settings
        for param_id, param in params.items():
            default_values[param_id] = param.get('default', 0)

        # Render in the server-specified semantic order. Each descriptor
        # carries an explicit ``order`` int (lower = higher on the panel);
        # this is independent of dict-insertion quirks and of the fact that
        # the server merges some controls (e.g. lora_swap) in separately.
        # Params without ``order`` sink to the bottom (stable by id).
        ordered = sorted(
            params.items(),
            key=lambda kv: (kv[1].get('order', 10_000), kv[0]),
        )

        # Add controls based on parameters
        for param_id, param in ordered:
            field_type = param.get('field')
            default_value = default_values.get(param_id)

            if field_type == 'range':
                self.add_slider(param_id, param, default_value)
            elif field_type == 'checkbox':
                self.add_checkbox(param_id, param, default_value)
            elif field_type == 'textarea':
                self.add_text_input(param_id, param, default_value)
            elif field_type == 'select':
                self.add_select(param_id, param, default_value)
            elif field_type == 'lora_swap':
                self.add_lora_swap(param_id, param, default_value)
            elif field_type == 'lora_toggle':
                self.add_signal_checkbox(
                    param_id, param, default_value, self.lora_toggle_changed)
            elif field_type == 'aspect_toggle':
                self.add_signal_checkbox(
                    param_id, param, default_value, self.aspect_toggle_changed)

        # Add stretch at the end so controls stay at top
        self.main_layout.addStretch()

    def add_slider(self, param_id, param, default_value=None):
        """Add a slider control"""
        # Create container widget with minimum height
        container = QWidget()
        container.setMinimumHeight(SLIDER_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        # Label with fixed width for alignment
        label = QLabel(param.get('title', param_id))
        label.setMinimumWidth(150)
        layout.addWidget(label)
        
        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(param.get('min', 0) * 100), int(param.get('max', 1) * 100))
        if default_value is not None:
            default_val = int(default_value * 100)
        else:
            default_val = int(param.get('default', 0) * 100)
        slider.setValue(default_val)
        
        # Value display
        value_edit = QLineEdit()
        value_edit.setFixedWidth(50)
        value_edit.setText(str(default_val/100))
        
        # Connect signals
        def on_slider_change(value):
            value_edit.setText(f"{value/100:.2f}")
            self.parameter_changed.emit(param_id, value/100)
            
        def on_text_change():
            try:
                value = float(value_edit.text()) * 100
                slider.setValue(int(value))
                self.parameter_changed.emit(param_id, float(value_edit.text()))
            except ValueError:
                pass
                
        slider.valueChanged.connect(on_slider_change)
        value_edit.editingFinished.connect(on_text_change)

        layout.addWidget(slider)
        layout.addWidget(value_edit)

        self.main_layout.addWidget(container)
        self.controls[param_id] = (slider, value_edit)

    def add_signal_checkbox(self, param_id, param, default_value, signal):
        """Checkbox wired to a DEDICATED signal instead of the generic
        parameter/knob path. Used for controls whose toggle triggers a
        bespoke action in main.py (heavy LoRA enable/disable, client-side
        camera aspect crop) rather than a plain server field update."""
        container = QWidget()
        container.setMinimumHeight(CHECKBOX_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        checkbox = QCheckBox(param.get('title', param_id))
        checked = default_value if default_value is not None \
            else param.get('default', False)
        checkbox.setChecked(bool(checked))
        checkbox.toggled.connect(lambda v, sig=signal: sig.emit(bool(v)))
        layout.addWidget(checkbox)

        self.main_layout.addWidget(container)
        self.controls[param_id] = checkbox

    def add_checkbox(self, param_id, param, default_value=None):
        """Add a checkbox control"""
        # Create container widget with minimum height
        container = QWidget()
        container.setMinimumHeight(CHECKBOX_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        checkbox = QCheckBox(param.get('title', param_id))
        if default_value is not None:
            checkbox.setChecked(default_value)
        else:
            checkbox.setChecked(param.get('default', False))

        checkbox.stateChanged.connect(
            lambda v, pid=param_id: self.parameter_changed.emit(pid, bool(v)))
        layout.addWidget(checkbox)
        layout.addStretch()

        self.main_layout.addWidget(container)
        self.controls[param_id] = checkbox

    def add_select(self, param_id, param, default_value=None):
        """Add a dropdown (combo) control for an enumerated parameter."""
        container = QWidget()
        container.setMinimumHeight(SLIDER_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        label = QLabel(param.get('title', param_id))
        label.setMinimumWidth(150)
        layout.addWidget(label)

        combo = QComboBox()
        options = [str(o) for o in param.get('options', [])]
        combo.addItems(options)
        current = default_value if default_value is not None else param.get('default')
        if current is not None and str(current) in options:
            combo.setCurrentText(str(current))

        # Legacy path (parameter_changed). The live-knob path is wired by
        # apply_capabilities → _wire_live_knob for V2.
        combo.currentTextChanged.connect(
            lambda text, pid=param_id: self.parameter_changed.emit(pid, text))

        layout.addWidget(combo)
        layout.addStretch()

        self.main_layout.addWidget(container)
        self.controls[param_id] = combo

    def add_lora_swap(self, param_id, param, default_value=None):
        """Curation-index + two slug dropdowns (A, B) + Load button.

        Deferred (unlike auto-firing knobs): nothing happens until Load is
        pressed (a swap stalls the server a few seconds). The curation
        dropdown is a quick-pick — selecting an index fills A/B from the
        server's preset pairs (lora_index.yaml); A/B can also be set
        manually. ``presets`` maps curation index → [slug_a, slug_b].
        """
        container = QWidget()
        container.setMinimumHeight(SLIDER_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        label = QLabel(param.get('title', param_id))
        label.setMinimumWidth(150)
        layout.addWidget(label)

        options = [str(o) for o in param.get('options', [])]
        presets = param.get('presets', {}) or {}
        # Kept as instance state so update_lora_options() can repopulate
        # the dropdowns in place (e.g. after a LoRA Lab deploy).
        self._lora_options = options
        self._lora_presets = presets

        # Curation-index quick-pick (custom + sorted preset indices).
        cur = QComboBox()
        cur_keys = sorted(presets.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
        cur.addItems(["(custom)"] + [str(k) for k in cur_keys])

        combo_a = QComboBox(); combo_a.addItems(options)
        combo_b = QComboBox(); combo_b.addItems(options)
        da = str(param.get('default_a', '')); db = str(param.get('default_b', ''))
        if da in options:
            combo_a.setCurrentText(da)
        if db in options:
            combo_b.setCurrentText(db)

        # Per-LoRA fuse weights (editable; default 1.0 = full strength).
        # Applied at fuse time in fused mode (ignored in dynamic-blend
        # mode, which is α-driven per frame).
        dwa = str(param.get('default_weight_a', 1.0))
        dwb = str(param.get('default_weight_b', 1.0))
        weight_a = QLineEdit(dwa); weight_a.setFixedWidth(44)
        weight_b = QLineEdit(dwb); weight_b.setFixedWidth(44)

        def _on_curation(idx_text):
            # Read via self so update_lora_options() swaps take effect.
            pair = self._lora_presets.get(idx_text)
            if pair and len(pair) == 2:
                if combo_a.findText(str(pair[0])) >= 0:
                    combo_a.setCurrentText(str(pair[0]))
                if combo_b.findText(str(pair[1])) >= 0:
                    combo_b.setCurrentText(str(pair[1]))
        cur.currentTextChanged.connect(_on_curation)

        def _on_load():
            try:
                wa = float(weight_a.text())
            except ValueError:
                wa = 1.0
            try:
                wb = float(weight_b.text())
            except ValueError:
                wb = 1.0
            self.lora_swap_requested.emit(
                combo_a.currentText(), combo_b.currentText(), wa, wb
            )

        btn = QPushButton("Load")
        btn.clicked.connect(_on_load)

        def _on_swap():
            # Swap the A/B slug selections + fuse weights in place, then
            # load the swapped pair (same heavy path as Load). The server's
            # swap_ack carries the per-side trigger prefixes of the new
            # orientation, so the user-prompt A/B fields swap their
            # prefixes automatically (typed text stays put).
            a, b = combo_a.currentText(), combo_b.currentText()
            wa_t, wb_t = weight_a.text(), weight_b.text()
            combo_a.setCurrentText(b)
            combo_b.setCurrentText(a)
            weight_a.setText(wb_t)
            weight_b.setText(wa_t)
            # A swapped pair no longer matches a curation preset index.
            cur.blockSignals(True)
            cur.setCurrentText("(custom)")
            cur.blockSignals(False)
            _on_load()

        swap_btn = QPushButton("Swap")
        swap_btn.setToolTip(
            "Swap LoRA A ↔ B (slugs + weights) and load the swapped pair.\n"
            "User-prompt field prefixes follow automatically."
        )
        swap_btn.clicked.connect(_on_swap)

        layout.addWidget(QLabel("idx"))
        layout.addWidget(cur)
        layout.addWidget(combo_a)
        layout.addWidget(weight_a)
        layout.addWidget(combo_b)
        layout.addWidget(weight_b)
        layout.addWidget(btn)
        layout.addWidget(swap_btn)
        self.main_layout.addWidget(container)
        # NOTE: tuple stays at 6 entries — update_lora_options() and
        # set_lora_selection() unpack it positionally; the Swap button
        # needs no external handle.
        self.controls[param_id] = (cur, combo_a, weight_a, combo_b, weight_b, btn)
        self._lora_param_id = param_id

    def update_lora_options(self, options: list, presets: dict):
        """Repopulate the LoRA hot-swap dropdowns in place.

        Called when the server's slug registry changes at runtime (a LoRA
        Lab deploy or pair-create) — the deploy response carries the fresh
        slug/preset lists so the dropdowns stay current without a
        reconnect. Current selections are preserved when still valid.
        """
        pid = getattr(self, "_lora_param_id", None)
        if pid is None or pid not in self.controls:
            return
        widgets = self.controls[pid]
        if not (isinstance(widgets, tuple) and len(widgets) >= 6):
            return
        cur, combo_a, _wa, combo_b, _wb, _btn = widgets

        options = [str(o) for o in options]
        presets = dict(presets or {})
        self._lora_options = options
        self._lora_presets = presets

        prev_cur = cur.currentText()
        prev_a = combo_a.currentText()
        prev_b = combo_b.currentText()

        cur.blockSignals(True)
        cur.clear()
        cur_keys = sorted(presets.keys(),
                          key=lambda k: int(k) if str(k).isdigit() else k)
        cur.addItems(["(custom)"] + [str(k) for k in cur_keys])
        if cur.findText(prev_cur) >= 0:
            cur.setCurrentText(prev_cur)
        cur.blockSignals(False)

        for combo, prev in ((combo_a, prev_a), (combo_b, prev_b)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(options)
            if combo.findText(prev) >= 0:
                combo.setCurrentText(prev)
            combo.blockSignals(False)

    def add_text_input(self, param_id, param, default_value=None):
        """Add a text input control"""
        # Create container widget with minimum height
        container = QWidget()
        container.setMinimumHeight(TEXT_INPUT_MIN_HEIGHT)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 4)

        # Label with fixed width for alignment
        label = QLabel(param.get('title', param_id))
        label.setMinimumWidth(150)
        layout.addWidget(label)

        text_input = QLineEdit()
        if default_value is not None:
            text_input.setText(str(default_value))
        else:
            text_input.setText(str(param.get('default', '')))

        text_input.textChanged.connect(
            lambda v, pid=param_id: self.parameter_changed.emit(pid, v))
        layout.addWidget(text_input)

        self.main_layout.addWidget(container)
        self.controls[param_id] = text_input

    def clear_layout(self, layout):
        """Recursively clear a layout and its widgets"""
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self.clear_layout(item.layout())

    def apply_capabilities(self, caps: dict):
        """Apply realtime-protocol capabilities to the existing controls.

        Expected shape (from the server's `session_ready` ack):
            {
              "live_adjustable": ["cn_scale", "audio_reaction_output_gain", ...],
              "frozen": ["width", "height", "strength", "num_inference_steps", ...]
            }

        Behavior per field:
            * frozen          → disable + visually grey out (read-only).
            * live_adjustable → ensure enabled; rewire `valueChanged` to
                                emit `knob_changed(field, value)` so
                                main.py can forward to ws_client.update_knob.
            * unclassified    → leave existing behavior intact
                                (non-breaking for legacy fields).
        """
        self._capabilities = {
            "live_adjustable": list(caps.get("live_adjustable", [])),
            "frozen": list(caps.get("frozen", [])),
        }
        frozen = set(self._capabilities["frozen"])
        live = set(self._capabilities["live_adjustable"])

        for param_id, control in self.controls.items():
            if param_id in frozen:
                self._set_control_enabled(control, False)
                continue
            if param_id in live:
                self._set_control_enabled(control, True)
                self._wire_live_knob(param_id, control)

    def _set_control_enabled(self, control, enabled: bool):
        """Enable / disable a control (handles slider tuples + widgets)."""
        if isinstance(control, tuple):  # (slider, value_edit)
            for w in control:
                w.setEnabled(enabled)
        else:
            control.setEnabled(enabled)

    def _wire_live_knob(self, param_id: str, control):
        """Emit `knob_changed` for live-adjustable fields.

        Note: we keep the existing `parameter_changed` wiring intact;
        `knob_changed` fires in addition, so the legacy path still
        works under the V2=0 feature flag.
        """
        if isinstance(control, tuple):  # slider
            slider, _ = control
            slider.valueChanged.connect(
                lambda v, pid=param_id: self.knob_changed.emit(pid, v / 100)
            )
        elif isinstance(control, QCheckBox):
            control.stateChanged.connect(
                lambda v, pid=param_id: self.knob_changed.emit(pid, bool(v))
            )
        elif isinstance(control, QComboBox):
            control.currentTextChanged.connect(
                lambda text, pid=param_id: self.knob_changed.emit(pid, text)
            )
        elif isinstance(control, QLineEdit):
            control.textChanged.connect(
                lambda v, pid=param_id: self.knob_changed.emit(pid, v)
            )

    def update_control(self, param_id, value):
        """Update a control's value programmatically"""
        if param_id not in self.controls:
            return
            
        control = self.controls[param_id]
        if isinstance(control, tuple) and len(control) == 2:  # Slider (slider, value_edit)
            slider, value_edit = control
            slider.setValue(int(value * 100))
            value_edit.setText(f"{value:.2f}")
        elif isinstance(control, tuple):
            # Composite control (e.g. lora_swap's 6-tuple) — not a scalar
            # value. Use the dedicated setter (set_lora_selection) instead.
            return
        elif isinstance(control, QCheckBox):
            control.setChecked(value)
        elif isinstance(control, QComboBox):
            control.setCurrentText(str(value))
        elif isinstance(control, QLineEdit):
            control.setText(str(value))

    def set_lora_selection(self, slug_a, slug_b, weight_a, weight_b):
        """Programmatically set the LoRA A/B slugs + fuse weights.

        Used to restore the client's last-loaded LoRA pair after a reconnect.
        Does NOT press Load — the caller fires the swap on the server
        separately. Returns True only if both slugs still exist in the
        (rebuilt) dropdowns; returns False and changes nothing otherwise, so
        the caller can decide whether to re-fire the swap.
        """
        if self._lora_param_id is None or self._lora_param_id not in self.controls:
            return False
        control = self.controls[self._lora_param_id]
        if not (isinstance(control, tuple) and len(control) == 6):
            return False
        cur, combo_a, weight_a_edit, combo_b, weight_b_edit, _btn = control
        sa, sb = str(slug_a), str(slug_b)
        if combo_a.findText(sa) < 0 or combo_b.findText(sb) < 0:
            return False
        # "(custom)" so the curation quick-pick doesn't overwrite A/B.
        cur.setCurrentText("(custom)")
        combo_a.setCurrentText(sa)
        combo_b.setCurrentText(sb)
        weight_a_edit.setText(str(weight_a))
        weight_b_edit.setText(str(weight_b))
        return True
"""Sound Lab node documentation — the "so what" of every op, param,
source signal and gate threshold, shown in the graph editor's expandable
docs section. Style: simple but precise, low verbosity, first
principles.

Also home of the human-readable display names (node ids like `bp` are
identifiers; the canvas and sidebar show these labels instead).
"""

# ---- human-readable names -------------------------------------------------

OP_LABELS = {
    "source": "signal source",
    "voxp": "vocal presence",
    "const": "constant",
    "wsum": "weighted sum",
    "max": "maximum",
    "min": "minimum",
    "mul": "multiply",
    "add": "add",
    "inv": "invert",
    "xfade": "crossfade",
    "ema": "smooth",
    "stretch": "range stretch",
    "thresh": "de-grass (median gate)",
    "punch": "punch (attack/release)",
    "xform": "engine transform",
}

XFORM_LABELS = {"onset": "band-pass + smooth", "down": "pulse shape"}

SIGNAL_LABELS = {
    "beat": "beat pulse", "down": "downbeat", "onset": "onset flux",
    "onset_low": "kick onset (low band)",
    "pitch": "vocal register", "drums": "drums stem", "bass": "bass stem",
    "other": "other stem", "vocals": "vocals stem",
    "level_bass": "band level: bass", "level_low_mid": "band level: low-mid",
    "level_mid": "band level: mid", "level_treble": "band level: treble",
    "excite": "excitation", "ramp": "test ramp",
}

# ---- docs ------------------------------------------------------------------

OP_DOCS = {
  "source": {"what": "emits one raw engine signal per frame — the entry point for any audio feature into the graph.", "params": {"signal": "picks which engine feed to read (beat, onset, stems, band levels, …) — each has its own doc below.", "norm": "on stretches the signal to fill 0..1 using its own recent quiet/loud range; off passes it through raw.", "peak_hl_s": "higher = the ceiling forgets loud peaks more slowly, so norm reacts less to a single recent hit.", "floor_rise_s": "higher = the floor climbs toward the peak more slowly, so quiet stretches stay near 0 longer."}},
  "voxp": {"what": "estimates 0..1 how present a lead vocal is right now, combining pitch confidence with vocal-stem energy share. use it to gate visuals to singing versus instrumental sections.", "params": {"attack_s": "lower = presence jumps up faster when singing starts.", "release_s": "higher = presence lingers longer through pauses and breaths before falling.", "voiced_gain": "higher = less voiced-frame fraction is needed to hit full presence.", "share_ref": "lower = a smaller vocal-stem energy share is enough to count as a real vocal, not bleed.", "pitch_conf_min": "higher = only very confident pitch detections count as singing, cutting false positives."}},
  "const": {"what": "outputs a fixed value every frame. use it as a steady bias, floor, or a static weight-source in a mix.", "params": {"value": "sets the constant output level."}},
  "wsum": {"what": "blends inputs by weighted average, weights auto-normalized to sum to 1. use it for a smooth, continuous mix where every branch always contributes.", "params": {"weights": "raises one input's share of the blend relative to the others."}},
  "max": {"what": "takes the loudest of its inputs frame by frame. use it so a sparse spike (like a downbeat) can punch through a calmer bed signal without being averaged away.", "params": {}},
  "min": {"what": "takes the quietest of its inputs frame by frame, acting like an AND. use it so the output only rises when every branch agrees.", "params": {}},
  "mul": {"what": "multiplies its inputs together. use it to gate one signal by another — anywhere the gate is 0, the output is silenced.", "params": {}},
  "add": {"what": "sums its inputs, clipped at 1. use it to stack contributions that should pile up rather than blend.", "params": {}},
  "inv": {"what": "flips a signal: output = 1−input. use it to turn a presence signal into an absence signal, or a gate into its opposite.", "params": {}},
  "xfade": {"what": "crossfades two inputs by a third, mix·a + (1−mix)·b. use it when the blend position itself should be driven by a live signal instead of a fixed knob.", "params": {}},
  "ema": {"what": "smooths a signal by exponential averaging. use it to remove frame-to-frame jitter before it reaches the output.", "params": {"tau_s": "higher = smoother but more sluggish, lagging further behind the live signal."}},
  "thresh": {"what": "subtracts the local median — the between-hit 'grass' — so troughs fall to a true zero and only genuine peaks survive. the classical onset peak-picking stage; this is what makes a flux signal read as hits instead of texture.", "params": {"window_s": "longer = the grass estimate is steadier and adapts slowly; shorter = it tracks fast texture changes.", "k": "higher = stricter — only peaks well above the local baseline pass; lower = more texture leaks through."}},
  "punch": {"what": "fast-attack / slow-release envelope: peaks land at full height instantly, then decay at a set rate — the vu-meter / beat-flash shape. use it after de-grass to make hits visually snappy.", "params": {"attack_s": "lower = the rise is more instantaneous.", "release_s": "higher = each hit glows longer; lower = tighter blips."}},
  "stretch": {"what": "rescales a signal using its own trailing percentile range so it reliably spans 0..1. use it downstream of a signal that's technically 0..1 but rarely uses the full range.", "params": {"win_s": "longer = the stretch adapts to dynamics over a longer history, changing less moment to moment.", "lo_pct": "higher = more of the quiet end of the range gets pinned to 0.", "hi_pct": "lower = more of the loud end of the range gets pinned to 1.", "min_span": "higher = caps how much a nearly-flat signal gets amplified, avoiding noise blown up to full range."}},
}

SIGNAL_DOCS = {
  "beat": "BeatNet (CRNN + particle filter) tracks tempo and beat phase; value is confidence × a raised-cosine pulse peaking exactly on each predicted beat. Follows the felt pulse, not the fastest subdivision — falls back to a comb-filter tracker if BeatNet loses lock.",
  "down": "a decaying pulse fired only on BeatNet's predicted downbeat, zero on the beats between. Marks bars, not beats — sparse on purpose, for phrase-level structure.",
  "onset": "spectral flux: frame-to-frame rise in log-compressed spectral magnitude, band-passed below ~8kHz, lightly smoothed. Fires on every percussive attack, on-grid or not — the fastest, twitchiest signal here.",
  "onset_low": "spectral flux from only the bottom of the spectrum (<~450 Hz) — kick and bass attacks only, immune to hats and synth movement by construction. The 'oomph' detector: patch it when techno should hit pointwise.",
  "pitch": "PESTO (streaming pitch model) tracks f0 on the isolated vocal stem; value is that pitch's percentile rank (p10–p90) within the singer's own recent range. High means singing high for this voice, not high in absolute Hz.",
  "drums": "RMS loudness of the drum stem after htdemucs GPU source separation. Tracks how hard the kit is hitting right now.",
  "bass": "RMS loudness of the bass stem after htdemucs GPU source separation. Tracks low-end instrument energy independent of drums.",
  "other": "RMS loudness of htdemucs's residual stem — synths, guitars, keys, strings, everything not drums/bass/vocals.",
  "vocals": "RMS loudness of the isolated vocal stem after htdemucs GPU source separation. Tracks how present/loud the voice is, regardless of pitch.",
  "level_bass": "legacy mel-band envelope over the low frequencies, percentile auto-gain-normalized and smoothed. Cheap, robust loudness following for the bass register without source separation.",
  "level_low_mid": "legacy mel-band envelope over the low-mid frequencies, percentile auto-gain-normalized and smoothed. Cheap, robust loudness following for that register without source separation.",
  "level_mid": "legacy mel-band envelope over the mid frequencies, percentile auto-gain-normalized and smoothed. Cheap, robust loudness following for that register without source separation.",
  "level_treble": "legacy mel-band envelope over the high frequencies, percentile auto-gain-normalized and smoothed. Cheap, robust loudness following for cymbals/hi-hats/air without source separation.",
  "excite": "legacy hand-tuned composite: band energy + spectral flux + transient/arc cues blended into one music-intensity scalar. A one-knob proxy for \"how much is going on\" predating the per-stem signals.",
  "ramp": "a deterministic cosine cycle driven by the frame counter — no audio input at all. For testing or running visuals with no mic; bypasses the noise gate.",
}

GATE_DOCS = {
  "what": "gates the mic feed so only music drives the visuals — while closed, detectors freeze, gpu idles, and α decays to floor.",
  "params": {
    "gate_stay_db": "raise it and only very loud passages stay forced-open; lower it and more moderate volume also can't be closed once reached.",
    "gate_min_db": "raise it and quiet music can't open the gate; lower it and fainter signal is allowed to.",
    "gate_open_db": "raise it and the gate needs a bigger jump above the quiet floor to open; lower it and it opens on smaller rises.",
    "gate_mod_db": "raise it and only more dynamic, breathing loudness counts as music; lower it and steadier signal can open the gate too.",
  },
}

XFORM_DOCS = {
  "onset": {"what": "onset detection: flux is measured only below max_hz, then smoothed — this shapes how punchy vs. noisy the rhythm signal feels.", "params": {"max_hz": "lower it to isolate kick/snare/bass onsets; raise it to also catch cymbals and hats.", "smooth_s": "raise it for rounder, slower onset bumps; lower it for spikier, more nervous ones."}},
  "down": {"what": "downbeat pulse: each detected bar-start fires a linear decay to zero — this sets how the beat pulse feels in time.", "params": {"decay_s": "raise it so each downbeat glows longer and less sparse; lower it for tighter, shorter blips."}},
}


def node_docs(op: str, params: dict) -> tuple:
    """(what, {param_key: doc}) for a node — resolves xform by kind and
    appends the selected signal's doc for sources."""
    if op == "xform":
        kind = str((params or {}).get("kind", "onset"))
        d = XFORM_DOCS.get(kind, {"what": "", "params": {}})
        return d["what"], d["params"]
    d = OP_DOCS.get(op, {"what": "", "params": {}})
    what = d["what"]
    if op == "source":
        sig = str((params or {}).get("signal", ""))
        sd = SIGNAL_DOCS.get(sig)
        if sd:
            what = f"{what}\n\n{SIGNAL_LABELS.get(sig, sig)}: {sd}"
    return what, d["params"]

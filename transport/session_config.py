"""SessionConfig + AudioZoomConfig dataclasses.

Mirrors the server-side contract that Agent B is constructing in
`sdxl-travel-ablation/core/realtime.py`. The default values track
`experiments/manifests/_p16stage3_final.yaml` (Stage-3 final baseline)
plus the plantoid 16 caller defaults.

Both dataclasses provide `to_dict()` for JSON serialization (sent in
the session_config handshake) and `from_dict()` for re-hydration from
the `GET /api/installations/plantoid16/defaults` endpoint response.

Field names MUST match server-side exactly; do not rename without a
coordinated server-side change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional


@dataclass
class AudioZoomConfig:
    """Audio-reactive pipeline tuning.

    These fields nest under `audio_zoom_*` in the YAML manifest but are
    grouped here for clarity. `to_dict()` flattens them back to the
    `audio_zoom_*` namespace on the wire so the server sees the same
    keys it sees from the YAML loader.
    """

    space: str = "pixel"                # zoom space: "pixel" or "latent"
    max: float = 1.0                    # 1.0 = controller OFF
    n_bands: int = 4
    f_min: float = 30.0
    f_max: float = 8000.0
    compression: str = "cbrt"
    attack_ms: float = 80.0
    release_ms: float = 400.0
    window_seconds: float = 5.0
    warmup_seconds: float = 0.3
    p_low: float = 10.0
    p_high: float = 90.0
    output_smoothing_ms: float = 200.0

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to the `audio_zoom_*` wire namespace."""
        return {f"audio_zoom_{k}": v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioZoomConfig":
        """Read either flattened (`audio_zoom_*`) or nested form."""
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            wire_key = f"audio_zoom_{f.name}"
            if wire_key in data:
                kwargs[f.name] = data[wire_key]
            elif f.name in data:
                kwargs[f.name] = data[f.name]
        return cls(**kwargs)


@dataclass
class SessionConfig:
    """Full session configuration sent in the handshake.

    Defaults mirror `_p16stage3_final.yaml`. Caller-side overrides
    (init_video, audio_zoom_wav, prompts_*, loras, seed, etc.) are
    expected to be filled in by the orchestrator before serialization;
    fields that the orchestrator usually supplies are left as `None`.
    """

    # ----- Denoiser / pipeline mode -----
    denoiser: str = "streambatch"
    mode: str = "img2img"
    controlnet: str = "none"
    cn_scale: float = 0.0               # only meaningful when controlnet != "none"

    # ----- Diffusion knobs -----
    strength: float = 0.7
    delta_noise_pct: float = 10.0
    n_rungs: int = 2
    num_inference_steps: int = 50
    prompt_travel: str = "slerp"
    noise_travel: str = "none"
    init_encode_once: bool = False      # forced false by init_video path

    # ----- Pixel feedback path -----
    feedback_strength: float = 0.4

    # ----- LoRA blend curve (pair-LoRA mode) -----
    lora_blend_curve: str = "cosine_tent"
    lora_blend_curve_scale: float = 1.0
    audio_lora_blend_method: str = "naive"

    # ----- Latent carryover (Proposals E + F) -----
    latent_carryover: float = 0.2
    latent_carryover_ema: float = 0.1

    # ----- H smoothing (OFF by default in Stage 3) -----
    x0_smooth_sigma: Optional[float] = None
    x0_smooth_beta_low: Optional[float] = None
    x0_smooth_beta_high: Optional[float] = None

    # ----- Output gain on the audio-reactive α driver -----
    audio_reaction_output_gain: float = 1.0

    # ----- Output resolution / dtype -----
    width: int = 576
    height: int = 768
    dtype: str = "bfloat16"

    # ----- Plantoid 16 / canonical-run fields emitted by /defaults -----
    # These were missing from the original D-side schema; the server's
    # `build_plantoid16_defaults` emits them and SessionConfig.from_dict
    # would silently drop them without explicit fields here.
    final_prompts_a: List[str] = field(default_factory=list)
    final_prompts_b: List[str] = field(default_factory=list)
    lora: str = "21"                    # preset index, shorthand, or comma-list
    audio_band: str = "treble"          # mel band driving reactivity
    h_smoothing: bool = False           # JJ baseline: latent_carryover instead

    # ----- Audio capture contract -----
    # The client captures PCM s16le mono at this rate; the server's
    # RealtimeFFTAudioAnalyzer is constructed with the same value via the
    # SessionConfig handshake. Mismatch → silent garbage FFT. Defaults to
    # the server's default (44100) so handshakes that omit the field
    # round-trip to the same rate on both sides.
    sample_rate: int = 44100
    fps: int = 20                       # informs server's samples_per_chunk

    # ----- Caller-supplied per-session overrides (orchestrator fills) -----
    # Left as None on the bare default; the orchestrator passes them in
    # from the caller (e.g. plantoid16 installation defaults).
    init_video: Optional[str] = None
    seed: Optional[int] = None
    audio_zoom_wav: Optional[str] = None
    audio_zoom_band: Optional[List[int]] = None
    audio_lora_blend_band: Optional[List[int]] = None

    # Pair-LoRA mode fields
    prompts_a_inline: Optional[List[str]] = None
    prompts_b_inline: Optional[List[str]] = None
    audio_lora_blend_paths: Optional[List[str]] = None

    # Single-LoRA mode fields
    prompts_inline: Optional[List[str]] = None
    loras: Optional[List[str]] = None
    audio_mux_path: Optional[str] = None

    # ----- Audio-reactive sub-config (nested for readability) -----
    audio_zoom: AudioZoomConfig = field(default_factory=AudioZoomConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to the wire format expected by the server.

        Nested `audio_zoom` is flattened back to `audio_zoom_*` keys
        so the server sees the same schema it sees from the YAML loader.
        """
        out: Dict[str, Any] = {}
        for f in fields(self):
            if f.name == "audio_zoom":
                continue
            out[f.name] = getattr(self, f.name)
        out.update(self.audio_zoom.to_dict())
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionConfig":
        """Re-hydrate from a JSON dict (e.g. the /defaults endpoint).

        Tolerates both the flattened `audio_zoom_*` form and a nested
        `audio_zoom: {...}` form; missing fields fall back to dataclass
        defaults so partial server responses still work.
        """
        own_fields = {f.name for f in fields(cls)} - {"audio_zoom"}
        top_kwargs: Dict[str, Any] = {
            k: v for k, v in data.items() if k in own_fields
        }
        # Audio zoom sub-config
        if "audio_zoom" in data and isinstance(data["audio_zoom"], dict):
            audio_zoom = AudioZoomConfig.from_dict(data["audio_zoom"])
        else:
            audio_zoom = AudioZoomConfig.from_dict(data)
        return cls(audio_zoom=audio_zoom, **top_kwargs)

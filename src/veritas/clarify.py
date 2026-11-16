"""Articulatory-bottleneck clarifier — dysarthric -> clearer speech resynthesis.

The idea (VERITAS Phase 2). Run dysarthric audio through the SPARC acoustic-to-
articulatory bottleneck (``encode`` -> 12 EMA tract variables @ 50 fps), OPTIONALLY
move that articulatory trajectory toward the *healthy* articulatory manifold
(estimated from TORGO control speakers), then decode back to a waveform with the
SPARC HiFi-GAN synthesiser. Dysarthria typically CENTRALISES and REDUCES the
range of articulation; projecting the trajectory back onto the healthy first/second
moments is a principled "de-centralisation" of the vocal-tract motion.

Everything here is REAL and DETERMINISTIC:
  * the projection is a closed-form moment-matching / whitening-recolouring map
    whose parameters are all ESTIMATED FROM DATA (healthy-control EMA statistics);
    there are no hand-tuned magic constants (the only literal is a 1e-8 variance
    floor for divide-by-zero safety, documented below);
  * ``strength`` in [0, 1] interpolates continuously between pure re-synthesis
    (0 = decode the untouched code) and full projection (1);
  * speaker identity (``spk_emb``), pitch and loudness are left UNTOUCHED — this
    is articulatory clarification of the SAME speaker, not voice conversion.

Determinism scope (verified). Given a fixed SPARC ``code``, the clarification
``decode(clarify_code(code))`` is BIT-EXACT: the projection is closed-form numpy
and the SPARC HiFi-GAN synthesiser is frozen (``decode`` of the same code is
identical). SPARC's ``encode`` is bit-exact for EMA / speaker embedding, but its
GPU pitch extractor (torchcrepe) is ~1% non-bit-deterministic run-to-run — a
channel the clarifier never modifies, and consistent with the repo's documented
GPU FP nondeterminism (``src/seed_utils.py``). Pass a cached ``code`` (encode
once, reuse across strengths — what the eval and demo do) for exact reproduction.

Import :mod:`src.veritas` BEFORE constructing the SPARC inverter so weights load
from the local HF cache (offline bootstrap).

--------------------------------------------------------------------------------
EXPERIMENTAL SAFETY GATE (read this).
--------------------------------------------------------------------------------
``experimental_faithfulness_gate`` reuses the project's cycle-consistency
faithfulness (``faithfulness.py``) to FLAG frames the articulatory bottleneck
reconstructs with low cycle-consistency. It is labelled EXPERIMENTAL / UNVALIDATED
on purpose: the VERITAS H1 experiment found that per-frame cycle-consistency does
NOT predict true articulatory inversion error against measured TORGO EMA
(energy-controlled partial Spearman ~ 0). The gate is a research overlay for
transparency only — it does NOT accept/reject or alter clarified audio, and it is
NOT a validated measure of clarification quality. See ``EXPERIMENTAL_GATE_NOTE``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.veritas.faithfulness import frame_faithfulness

# SPARC EMA channel order (6 articulators x (X, Y) @ 50 fps). Mirrors
# ``src.veritas.torgo_ema.SPARC_EMA_ORDER`` — the authoritative pairing for
# ``code['ema']``: tongue dorsum/body/tip, jaw (lower incisor), upper/lower lip.
EMA_ORDER: tuple[str, ...] = (
    "TDX", "TDY", "TBX", "TBY", "TTX", "TTY",
    "LIX", "LIY", "ULX", "ULY", "LLX", "LLY",
)
EMA_DIM: int = 12

# Numerical variance floor ONLY (divide-by-zero guard); NOT a tuning knob. Under
# per-tract moment matching the OUTPUT per-channel std equals the healthy std by
# construction, so this floor cannot amplify a channel — it only prevents a NaN
# when a channel is exactly static. Documented for honesty; result-independent.
_VAR_EPS: float = 1e-8


def _sym_sqrt(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition (regularised)."""
    C = np.asarray(cov, dtype=np.float64)
    n = C.shape[0]
    reg = eps * (np.trace(C) / n)
    w, V = np.linalg.eigh(C + reg * np.eye(n))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _sym_inv_sqrt(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Symmetric inverse square root via eigendecomposition (regularised)."""
    C = np.asarray(cov, dtype=np.float64)
    n = C.shape[0]
    reg = eps * (np.trace(C) / n)
    w, V = np.linalg.eigh(C + reg * np.eye(n))
    w = np.clip(w, reg, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


@dataclass
class HealthyEMAStats:
    """Healthy-control articulatory manifold statistics (the projection target).

    Fit ONCE on SPARC ``code['ema']`` frames pooled over TORGO *control*
    (severity-0) speakers: per-channel mean ``mu`` (12,), per-channel std
    ``sigma`` (12,) and the full articulatory covariance ``cov`` (12, 12). All
    are plain data estimates — no hand-tuned constants. ``mu``/``sigma`` drive the
    default per-tract moment matching; ``cov`` drives the optional full-covariance
    CORAL whitening-recolouring.
    """

    mu: np.ndarray
    sigma: np.ndarray
    cov: np.ndarray

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float32).reshape(EMA_DIM)
        self.sigma = np.asarray(self.sigma, dtype=np.float32).reshape(EMA_DIM)
        self.cov = np.asarray(self.cov, dtype=np.float64).reshape(EMA_DIM, EMA_DIM)
        self._cov_sqrt = _sym_sqrt(self.cov)  # healthy recolouring matrix (CORAL)

    @classmethod
    def fit(cls, ema_frames: np.ndarray) -> "HealthyEMAStats":
        """Estimate the healthy manifold from pooled control EMA frames (N, 12)."""
        X = np.asarray(ema_frames, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != EMA_DIM:
            raise ValueError(f"expected (N, {EMA_DIM}) EMA frames, got {X.shape}")
        if len(X) < EMA_DIM + 1:
            raise ValueError(f"too few control frames to fit stats: {len(X)}")
        return cls(mu=X.mean(0), sigma=X.std(0), cov=np.cov(X, rowvar=False))

    def save(self, path: str) -> None:
        np.savez(
            path, mu=self.mu, sigma=self.sigma, cov=self.cov,
            ema_order=np.array(EMA_ORDER),
        )

    @classmethod
    def load(cls, path: str) -> "HealthyEMAStats":
        d = np.load(path, allow_pickle=False)
        return cls(mu=d["mu"], sigma=d["sigma"], cov=d["cov"])


def project_ema(
    ema: np.ndarray,
    stats: HealthyEMAStats,
    strength: float,
    mode: str = "per_tract",
) -> np.ndarray:
    """Move an EMA trajectory (T, 12) toward the healthy manifold; return (T, 12).

    ``strength`` in [0, 1] linearly interpolates the *articulatory code* between
    the original (0) and the fully projected target (1). Two principled maps:

    ``per_tract`` (default, robust): per-channel moment matching. Rescale each
      tract variable's deviation-from-its-own-utterance-mean so the utterance's
      per-channel mean/variance match the healthy control mean/variance:
          target = mu_h + (ema - mu_utt) * (sigma_h / sigma_utt).
      Temporal dynamics (the *shape* of the trajectory) are preserved; only the
      articulatory working point and range are normalised toward healthy. Output
      per-channel std == sigma_h by construction (so no channel can blow up).

    ``coral``: full-covariance whitening-recolouring (a.k.a. CORAL / PCA whitening
      toward the target). Whiten the utterance by its own covariance^(-1/2), then
      recolour by the healthy covariance^(1/2):
          target = mu_h + (ema - mu_utt) @ (Cov_utt^(-1/2) @ Cov_h^(1/2)).
      Matches the full 12-D articulatory second moment (cross-tract correlations),
      not just per-channel variance.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    E = np.asarray(ema, dtype=np.float64)
    if E.ndim != 2 or E.shape[1] != EMA_DIM:
        raise ValueError(f"expected (T, {EMA_DIM}) EMA, got {E.shape}")
    if strength == 0.0 or len(E) < 2:
        return E.astype(np.float32)

    mu_utt = E.mean(0)
    if mode == "per_tract":
        sigma_utt = np.sqrt(E.var(0) + _VAR_EPS)  # numerical floor only
        gain = stats.sigma.astype(np.float64) / sigma_utt
        target = stats.mu.astype(np.float64) + (E - mu_utt) * gain
    elif mode == "coral":
        cov_utt = np.cov(E, rowvar=False)
        transform = _sym_inv_sqrt(cov_utt) @ stats._cov_sqrt
        target = stats.mu.astype(np.float64) + (E - mu_utt) @ transform
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'per_tract' or 'coral')")

    out = (1.0 - strength) * E + strength * target
    return out.astype(np.float32)


class ArticulatoryClarifier:
    """Dysarthric -> clearer speech via the SPARC articulatory bottleneck.

    ``inverter`` is a :class:`src.veritas.inversion.SPARCInverter`. ``healthy_stats``
    is a fitted :class:`HealthyEMAStats` (the projection target); if ``None`` or
    ``strength == 0`` the clarifier degrades gracefully to pure SPARC re-synthesis.
    Deterministic — the projection is closed-form and the SPARC synthesiser is
    frozen. ``seed`` seeds numpy/torch defensively at construction.
    """

    def __init__(
        self,
        inverter,
        healthy_stats: HealthyEMAStats | None = None,
        mode: str = "per_tract",
        seed: int = 42,
    ):
        self.inverter = inverter
        self.stats = healthy_stats
        self.mode = mode
        self._seed(seed)

    @staticmethod
    def _seed(seed: int) -> None:
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:  # noqa: BLE001 — torch optional at import time
            pass

    def encode(self, wav) -> dict:
        return self.inverter.encode(wav)

    def clarify_code(self, code: dict, strength: float) -> dict:
        """Return a NEW code dict with the EMA projected (all else untouched)."""
        out = dict(code)
        if self.stats is not None and strength > 0:
            out["ema"] = project_ema(
                np.asarray(code["ema"], dtype=np.float32), self.stats, strength, self.mode
            )
        return out

    def clarify(
        self, wav, strength: float = 1.0, out_sr: int = 16000, code: dict | None = None
    ) -> np.ndarray:
        """Clarify one waveform. ``strength`` in [0,1] (0 = pure resynthesis).

        Returns a float32 waveform at ``out_sr`` (SPARC synthesises at 16 kHz).
        Pass a precomputed ``code`` (from :meth:`encode`) to reuse one SPARC
        encode across strengths and get bit-exact reproduction.
        """
        w, _meta = self.clarify_with_meta(wav, strength=strength, out_sr=out_sr, code=code)
        return w

    def clarify_with_meta(
        self, wav=None, strength: float = 1.0, out_sr: int = 16000, code: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """As :meth:`clarify` but also return a metadata dict (strength, mode,
        n_frames, per-channel articulatory displacement) for eval/UI. Pass a
        precomputed ``code`` to reuse one encode and guarantee determinism."""
        import torch
        import torchaudio

        strength = float(np.clip(strength, 0.0, 1.0))
        if code is None:
            if wav is None:
                raise ValueError("provide either `wav` or a precomputed `code`")
            code = self.encode(wav)
        ema0 = np.asarray(code["ema"], dtype=np.float32)
        code2 = self.clarify_code(code, strength)
        ema1 = np.asarray(code2["ema"], dtype=np.float32)
        w = np.asarray(self.inverter.decode(code2), dtype=np.float32).squeeze()
        if int(getattr(self.inverter, "sr", out_sr)) != out_sr:
            w = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(w)), self.inverter.sr, out_sr
            ).numpy()
        w = np.asarray(w, dtype=np.float32)
        n = min(len(ema0), len(ema1))
        disp = float(np.mean(np.abs(ema1[:n] - ema0[:n]))) if n else 0.0
        meta = {
            "strength": strength,
            "mode": self.mode,
            "projected": bool(self.stats is not None and strength > 0),
            "n_ema_frames": int(len(ema1)),
            "mean_abs_ema_displacement": disp,
            "out_sr": out_sr,
            "n_samples": int(len(w)),
        }
        return w, meta


# --------------------------------------------------------------------------- #
# EXPERIMENTAL / UNVALIDATED safety gate (research overlay only)
# --------------------------------------------------------------------------- #
EXPERIMENTAL_GATE_NOTE: str = (
    "EXPERIMENTAL / UNVALIDATED research overlay. Flags frames where the SPARC "
    "articulatory bottleneck reconstructs the input with LOW cycle-consistency in "
    "WavLM feature space. It is NOT a validated measure of clarification quality "
    "or inversion reliability: the VERITAS H1 experiment found that per-frame "
    "cycle-consistency does NOT predict true articulatory inversion error against "
    "measured TORGO EMA (energy-controlled partial Spearman ~ 0). Shown for "
    "transparency only; it does not accept, reject, or alter clarified audio."
)


@dataclass
class GateResult:
    """Output of :func:`experimental_faithfulness_gate` (research overlay)."""

    faith: np.ndarray            # (T,) per-frame cycle-consistency in [0, 1]
    threshold: float             # data-relative flag threshold (a quantile)
    flagged_mask: np.ndarray     # (T,) bool: frames at/below threshold
    frac_flagged: float
    flag_quantile: float
    experimental: bool
    label: str


def experimental_faithfulness_gate(
    fx: np.ndarray,
    fxh: np.ndarray,
    healthy_ref=None,
    flag_quantile: float = 0.20,
) -> GateResult:
    """EXPERIMENTAL / UNVALIDATED. Flag low-cycle-consistency frames for display.

    Reuses :func:`src.veritas.faithfulness.frame_faithfulness` on WavLM features of
    the input (``fx``) and of its articulatory re-synthesis (``fxh``); optionally
    centred on a healthy manifold (``healthy_ref``). ``flag_quantile`` is a
    DATA-RELATIVE visualisation threshold (flag the lowest-confidence fraction of
    frames), NOT a validated clinical cutoff. See ``EXPERIMENTAL_GATE_NOTE`` — this
    signal is NOT known to predict clarification quality and must be presented as a
    research overlay only.
    """
    faith = frame_faithfulness(np.asarray(fx), np.asarray(fxh), healthy_ref)
    if len(faith) == 0:
        return GateResult(
            faith=faith, threshold=0.0, flagged_mask=faith.astype(bool),
            frac_flagged=0.0, flag_quantile=flag_quantile, experimental=True,
            label=EXPERIMENTAL_GATE_NOTE,
        )
    thr = float(np.quantile(faith, flag_quantile))
    flagged = faith <= thr
    return GateResult(
        faith=faith,
        threshold=thr,
        flagged_mask=flagged,
        frac_flagged=float(np.mean(flagged)),
        flag_quantile=float(flag_quantile),
        experimental=True,
        label=EXPERIMENTAL_GATE_NOTE,
    )

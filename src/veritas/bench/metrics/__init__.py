"""Reference metric adapters shipped with VERITAS-Bench.

These are REAL, reusable implementations that double as worked examples for
plugging in your own metric:
  * :class:`PooledWavLMProbe`      — a probe-kind metric (Finding 1).
  * :class:`ATDReferenceDistance`  — a reference-kind metric (Findings 2 & 3).
  * :class:`PhonologicalSubspace`  — an independent REIMPLEMENTATION of the
    strongest current training-free metric family (phonological-subspace d'),
    included so the battery can audit it for the confounds its authors do not
    report controlling for (speaker-identity leakage, energy).
"""
from src.veritas.bench.metrics.atd import ATDReferenceDistance
from src.veritas.bench.metrics.phonosubspace import PhonologicalSubspace
from src.veritas.bench.metrics.pooled_probe import PooledWavLMProbe

__all__ = ["PooledWavLMProbe", "ATDReferenceDistance", "PhonologicalSubspace"]

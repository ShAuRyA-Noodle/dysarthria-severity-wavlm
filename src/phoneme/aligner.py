"""Montreal Forced Aligner (MFA) wrapper for Track B.

Setup steps live in scripts/install_mfa.sh. This module wraps the CLI:

    mfa align <corpus_dir> <dictionary> <acoustic_model> <out_dir>

and parses the resulting TextGrid intervals into Python tuples.

On Windows MFA's binaries (fstcompile etc.) live under
``<env>/Library/bin`` which is not on PATH unless conda activates
the env. We therefore prefer the ``scripts/run_mfa.sh`` wrapper that
prepends the right paths automatically.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

try:
    from praatio import textgrid as _tg
    _HAS_PRAATIO = True
except ImportError:
    _HAS_PRAATIO = False


SILENCE_LABELS = {"sil", "sp", "spn", "", " "}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _mfa_command() -> list[str] | None:
    """Return the command list to invoke MFA, or None if not installed.

    Preference order:
      1. ``scripts/run_mfa.sh`` (wrapper handles Windows PATH).
      2. Plain ``mfa`` on PATH (if the user already activated the env).
    """
    wrapper = _project_root() / "scripts" / "run_mfa.sh"
    if wrapper.exists():
        return ["bash", str(wrapper)]
    direct = shutil.which("mfa")
    if direct:
        return [direct]
    return None


def mfa_available() -> bool:
    """Return True iff MFA is callable (wrapper or direct)."""
    return _mfa_command() is not None


def run_mfa_align(
    corpus_dir: Path,
    dictionary_path: Path,
    acoustic_model: str,
    out_dir: Path,
    *,
    beam: int = 10,
    retry_beam: int = 40,
    clean: bool = True,
) -> int:
    """Invoke `mfa align`. Returns the subprocess exit code.

    Use beam=100, retry_beam=400 on heavily dysarthric subsets.
    """
    base_cmd = _mfa_command()
    if base_cmd is None:
        raise RuntimeError(
            "MFA not found. Install via scripts/install_mfa.sh "
            "(creates conda env `aligner` + downloads english_us_arpa)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *base_cmd, "align",
        str(corpus_dir), str(dictionary_path), acoustic_model, str(out_dir),
        "--beam", str(beam),
        "--retry_beam", str(retry_beam),
    ]
    if clean:
        cmd.append("--clean")
    return subprocess.run(cmd, check=False).returncode


def parse_textgrid(tg_path: Path) -> list[tuple[str, float, float]]:
    """Extract (phone, t_start, t_end) tuples from MFA TextGrid `phones` tier."""
    if not _HAS_PRAATIO:
        raise RuntimeError("praatio not installed (pip install praatio)")
    tg = _tg.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    tier = tg.getTier("phones")
    return [
        (iv.label, iv.start, iv.end)
        for iv in tier.entries
        if iv.label and iv.label not in SILENCE_LABELS
    ]


def alignment_log_likelihood(tg_path: Path) -> Optional[float]:
    """If MFA wrote a per-utterance log-likelihood, parse it. Otherwise None."""
    log_path = tg_path.with_suffix(".log")
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "log-likelihood" in line.lower():
            for tok in line.split():
                try:
                    return float(tok)
                except ValueError:
                    continue
    return None

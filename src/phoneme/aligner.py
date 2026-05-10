"""Montreal Forced Aligner (MFA) wrapper for Track B.

Setup steps live in scripts/install_mfa.sh. This module wraps the CLI:

    mfa align <corpus_dir> <dictionary> <acoustic_model> <out_dir>

and parses the resulting TextGrid intervals into Python tuples.
"""
from __future__ import annotations

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


def mfa_available() -> bool:
    """Return True iff `mfa` CLI is on PATH."""
    return shutil.which("mfa") is not None


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
    if not mfa_available():
        raise RuntimeError(
            "MFA not found on PATH. Install via scripts/install_mfa.sh "
            "and activate the `aligner` conda env."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mfa", "align",
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

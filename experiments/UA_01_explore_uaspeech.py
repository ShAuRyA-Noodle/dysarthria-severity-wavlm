"""Experiment UA-01 — Explore UA-Speech dataset.

Crawls both C-channel and FM-channel audio directories, reports per-speaker
statistics, sample-rate distributions, duration ranges, and intelligibility
score mappings.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

# Ensure Unicode prints correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uaspeech_parser import (  # noqa: E402
    parse_uaspeech_filename,
    get_uaspeech_speaker_info,
    UA_INTELLIGIBILITY,
    UA_SEVERITY,
    _get_audio_info,
)

# ---------------------------------------------------------------------------
# Paths — raw audio directories
# ---------------------------------------------------------------------------

# C-channel = control speakers (already extracted)
C_AUDIO_DIR = PROJECT_ROOT / "UASpeech_original_C" / "UASpeech" / "audio" / "original"

# FM-channel = dysarthric + possibly more speakers
FM_AUDIO_DIR = PROJECT_ROOT / "UASpeech_original_FM" / "UASpeech" / "audio" / "original"

# Alternate FM locations to check
FM_ALT_DIRS = [
    PROJECT_ROOT / "UASpeech_original_FM" / "audio" / "original",
    PROJECT_ROOT / "UASpeech_original_FM" / "UASpeech",
    PROJECT_ROOT / "UASpeech_original_FM",
]

MIC_FILTER = "M3"  # Primary microphone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_fm_audio_dir() -> Path | None:
    """Find the actual FM audio directory after extraction."""
    if FM_AUDIO_DIR.exists():
        return FM_AUDIO_DIR
    for alt in FM_ALT_DIRS:
        if alt.exists():
            # Check if it has WAV files or speaker subdirs
            wavs = list(alt.rglob("*.wav"))
            if wavs:
                return alt
    return None


def crawl_directory(audio_dir: Path) -> dict:
    """Crawl a directory and collect per-speaker stats."""
    stats = defaultdict(lambda: {
        "count": 0,
        "sample_rates": set(),
        "durations": [],
        "mics": set(),
    })

    if not audio_dir.exists():
        print(f"  [warn] Directory not found: {audio_dir}")
        return stats

    wav_files = sorted(audio_dir.rglob("*.wav"))
    print(f"  Found {len(wav_files)} total WAV files in {audio_dir}")

    for wav_path in wav_files:
        parsed = parse_uaspeech_filename(str(wav_path))
        spk = parsed["speaker_id"]
        mic = parsed["mic"]

        if spk == "unknown":
            continue

        # Only count files matching our mic filter
        if mic != MIC_FILTER:
            stats[spk]["mics"].add(mic)
            continue

        stats[spk]["mics"].add(mic)
        stats[spk]["count"] += 1

        try:
            info = _get_audio_info(str(wav_path))
            stats[spk]["sample_rates"].add(info["sample_rate"])
            stats[spk]["durations"].append(info["duration_sec"])
        except Exception as e:
            pass  # Skip corrupt files

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("  UA-Speech Dataset Exploration")
    print("=" * 70)

    # ── Speaker info table ──────────────────────────────────────────────
    print("\n── Speaker Intelligibility & Severity ──")
    speaker_info = get_uaspeech_speaker_info()
    print(speaker_info.to_string(index=False))
    print(f"\nTotal speakers in ground truth: {len(speaker_info)}")
    print(f"  Dysarthric: {(~speaker_info.is_control).sum()}")
    print(f"  Controls:   {speaker_info.is_control.sum()}")

    # ── Crawl C-channel ─────────────────────────────────────────────────
    print(f"\n── Crawling C-channel: {C_AUDIO_DIR} ──")
    c_stats = crawl_directory(C_AUDIO_DIR)

    # ── Crawl FM-channel ────────────────────────────────────────────────
    fm_dir = find_fm_audio_dir()
    fm_stats = {}
    if fm_dir:
        print(f"\n── Crawling FM-channel: {fm_dir} ──")
        fm_stats = crawl_directory(fm_dir)
    else:
        print("\n[warn] FM-channel audio directory not found!")
        print("  Checked:", FM_AUDIO_DIR)
        for alt in FM_ALT_DIRS:
            print(f"  Checked: {alt}")

    # ── Merge stats ─────────────────────────────────────────────────────
    all_speakers = set(list(c_stats.keys()) + list(fm_stats.keys()))

    print(f"\n── Per-Speaker Summary (mic={MIC_FILTER}) ──")
    print(f"{'Speaker':<10} {'WAVs':>6} {'SR(s)':>12} {'Dur min':>8} {'Dur max':>8} "
          f"{'Dur mean':>9} {'Intell%':>8} {'Sev':>4} {'Type':<10}")
    print("-" * 90)

    total_wavs = 0
    all_sample_rates = set()
    all_durations = []

    for spk in sorted(all_speakers):
        s = {}
        # Merge from both channels
        for source in [c_stats, fm_stats]:
            if spk in source:
                if not s:
                    s = {
                        "count": 0,
                        "sample_rates": set(),
                        "durations": [],
                    }
                s["count"] += source[spk]["count"]
                s["sample_rates"] |= source[spk]["sample_rates"]
                s["durations"].extend(source[spk]["durations"])

        if not s or s["count"] == 0:
            continue

        total_wavs += s["count"]
        all_sample_rates |= s["sample_rates"]
        all_durations.extend(s["durations"])

        intell = UA_INTELLIGIBILITY.get(spk, -1.0)
        sev = UA_SEVERITY.get(spk, -1)
        is_ctrl = spk.startswith("C")
        spk_type = "Control" if is_ctrl else "Dysarthric"

        dur_min = min(s["durations"]) if s["durations"] else 0.0
        dur_max = max(s["durations"]) if s["durations"] else 0.0
        dur_mean = sum(s["durations"]) / len(s["durations"]) if s["durations"] else 0.0
        sr_str = ",".join(str(x) for x in sorted(s["sample_rates"]))

        print(f"{spk:<10} {s['count']:>6} {sr_str:>12} {dur_min:>8.2f} {dur_max:>8.2f} "
              f"{dur_mean:>9.2f} {intell:>8.1f} {sev:>4} {spk_type:<10}")

    # ── Overall summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total speakers found:     {len(all_speakers)}")
    print(f"Total WAV files (mic={MIC_FILTER}): {total_wavs}")
    print(f"Sample rate distribution: {sorted(all_sample_rates)}")

    if all_durations:
        print(f"Duration range:           {min(all_durations):.2f}s - {max(all_durations):.2f}s")
        print(f"Duration mean:            {sum(all_durations)/len(all_durations):.2f}s")
        print(f"Total audio:              {sum(all_durations)/3600:.2f} hours")

    # Intelligibility range for dysarthric speakers
    dys_intell = [
        UA_INTELLIGIBILITY[spk] for spk in all_speakers
        if spk in UA_INTELLIGIBILITY and not spk.startswith("C")
    ]
    if dys_intell:
        print(f"Intelligibility range:    {min(dys_intell):.1f}% - {max(dys_intell):.1f}%")

    # Severity distribution
    sev_counts = defaultdict(int)
    sev_labels = {0: "High/Control", 1: "Mid", 2: "Low", 3: "Very Low"}
    for spk in all_speakers:
        sev = UA_SEVERITY.get(spk, -1)
        sev_counts[sev] += 1

    print("\nSeverity distribution:")
    for sev_val in sorted(sev_counts.keys()):
        label = sev_labels.get(sev_val, f"Unknown({sev_val})")
        print(f"  {label} (proxy={sev_val}): {sev_counts[sev_val]} speakers")


if __name__ == "__main__":
    main()

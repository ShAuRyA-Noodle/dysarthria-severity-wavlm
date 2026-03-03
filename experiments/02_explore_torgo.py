"""Experiment 02 — Explore TORGO dataset structure, audio stats, and label distribution."""

import wave
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORGO_RAW    = PROJECT_ROOT / "data" / "raw" / "torgo"
LOG_DIR      = PROJECT_ROOT / "results" / "logs"
META_DIR     = PROJECT_ROOT / "data" / "processed" / "metadata"

LOG_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Speaker metadata (authoritative — from .claude_context.md)
# ---------------------------------------------------------------------------
TORGO_SEVERITY = {
    "F01": 2, "F03": 1, "F04": 2,
    "M01": 2, "M02": 2, "M03": 1, "M04": 2, "M05": 1,
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
}

TORGO_GENDER = {
    "F01": "F", "F03": "F", "F04": "F",
    "M01": "M", "M02": "M", "M03": "M", "M04": "M", "M05": "M",
    "FC01": "F", "FC02": "F", "FC03": "F",
    "MC01": "M", "MC02": "M", "MC03": "M", "MC04": "M",
}

MIC_TYPES = ("wav_headMic", "wav_arrayMic")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_wav_info(filepath: Path) -> dict:
    """Return {duration_sec, sample_rate, n_channels, n_samples} for a WAV file."""
    with wave.open(str(filepath), "r") as wf:
        sr = wf.getframerate()
        n_samples = wf.getnframes()
        n_channels = wf.getnchannels()
    return {
        "sample_rate": sr,
        "n_samples": n_samples,
        "n_channels": n_channels,
        "duration_sec": n_samples / sr if sr > 0 else 0.0,
    }


def read_prompt(prompt_dir: Path, file_stem: str) -> str:
    """Read text from a prompt .txt file; return stem if missing."""
    txt_path = prompt_dir / f"{file_stem}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="replace").strip()
    return file_stem


def discover_speakers() -> list[dict]:
    """
    Walk TORGO_RAW and return a list of speaker dicts:
      {speaker_id, group_dir, speaker_dir, severity, gender}
    Group dirs: F, FC, M, MC
    """
    speakers = []
    for group in ("F", "FC", "M", "MC"):
        group_dir = TORGO_RAW / group
        if not group_dir.is_dir():
            continue
        for spk_dir in sorted(group_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk_id = spk_dir.name
            if spk_id not in TORGO_SEVERITY:
                continue  # skip Notes, etc.
            speakers.append({
                "speaker_id": spk_id,
                "group": group,
                "speaker_dir": spk_dir,
                "severity": TORGO_SEVERITY[spk_id],
                "gender": TORGO_GENDER[spk_id],
            })
    return speakers


def collect_utterances(speaker: dict) -> list[dict]:
    """
    For a given speaker, collect all utterances (both mic types).
    Returns list of dicts with {speaker_id, session, mic_type, file_stem,
                                 prompt_text, wav_path, **wav_info}.
    """
    utterances = []
    spk_dir: Path = speaker["speaker_dir"]

    for session_dir in sorted(spk_dir.iterdir()):
        if not session_dir.is_dir() or session_dir.name == "Notes":
            continue
        session = session_dir.name
        prompt_dir = session_dir / "prompts"

        for mic in MIC_TYPES:
            mic_dir = session_dir / mic
            if not mic_dir.is_dir():
                continue
            for wav_path in sorted(mic_dir.glob("*.wav")):
                stem = wav_path.stem
                prompt_text = read_prompt(prompt_dir, stem) if prompt_dir.is_dir() else stem
                try:
                    info = get_wav_info(wav_path)
                except Exception:
                    info = {"sample_rate": -1, "n_samples": 0, "n_channels": 1, "duration_sec": 0.0}
                utterances.append({
                    "speaker_id": speaker["speaker_id"],
                    "session": session,
                    "mic_type": mic,
                    "file_stem": stem,
                    "prompt_text": prompt_text,
                    "wav_path": str(wav_path),
                    **info,
                })
    return utterances


# ---------------------------------------------------------------------------
# Main exploration
# ---------------------------------------------------------------------------

def main() -> None:
    lines: list[str] = []

    def log(text: str = "") -> None:
        print(text)
        lines.append(text)

    log("=" * 70)
    log("TORGO DATASET EXPLORATION REPORT")
    log("=" * 70)

    # ── 1. Discover speakers ────────────────────────────────────────────────
    speakers = discover_speakers()
    log(f"\n[1] FOLDER STRUCTURE")
    log(f"    TORGO raw root: {TORGO_RAW}")
    log(f"    Speakers found: {len(speakers)}")

    all_utterances: list[dict] = []
    speaker_stats: dict[str, dict] = {}

    for spk in speakers:
        utts = collect_utterances(spk)
        all_utterances.extend(utts)

        head_utts  = [u for u in utts if u["mic_type"] == "wav_headMic"]
        array_utts = [u for u in utts if u["mic_type"] == "wav_arrayMic"]
        sessions   = sorted({u["session"] for u in utts})

        speaker_stats[spk["speaker_id"]] = {
            "severity": spk["severity"],
            "gender": spk["gender"],
            "sessions": sessions,
            "n_head": len(head_utts),
            "n_array": len(array_utts),
            "head_utts": head_utts,
        }

        log(f"\n    Speaker: {spk['speaker_id']}  "
            f"[severity={spk['severity']}, gender={spk['gender']}]")
        log(f"      Group dir   : {spk['group']}/")
        log(f"      Sessions    : {', '.join(sessions)}")
        log(f"      headMic WAVs: {len(head_utts)}")
        log(f"      arrayMic WAVs: {len(array_utts)}")

        head_words = sorted({u["prompt_text"] for u in head_utts})
        log(f"      Unique prompts (head): {len(head_words)}")
        preview = head_words[:5]
        log(f"      Sample prompts: {preview}" + (" ..." if len(head_words) > 5 else ""))

    # ── 2. Audio statistics ─────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("[2] AUDIO STATISTICS")

    head_utts_all = [u for u in all_utterances if u["mic_type"] == "wav_headMic"]
    array_utts_all = [u for u in all_utterances if u["mic_type"] == "wav_arrayMic"]

    for mic_label, utts in [("headMic", head_utts_all), ("arrayMic", array_utts_all)]:
        log(f"\n  -- {mic_label} ({len(utts)} files) --")
        if not utts:
            log("    (none found)")
            continue
        srs = sorted({u["sample_rate"] for u in utts})
        durations = [u["duration_sec"] for u in utts if u["duration_sec"] > 0]
        log(f"    Sample rates found : {srs}")
        if durations:
            log(f"    Duration min       : {min(durations):.3f} s")
            log(f"    Duration max       : {max(durations):.3f} s")
            log(f"    Duration mean      : {statistics.mean(durations):.3f} s")
            log(f"    Duration median    : {statistics.median(durations):.3f} s")
            total_hrs = sum(durations) / 3600
            log(f"    Total audio        : {total_hrs:.2f} hours  ({sum(durations)/60:.1f} min)")

    log("\n  -- Combined (both mics) --")
    all_durations = [u["duration_sec"] for u in all_utterances if u["duration_sec"] > 0]
    all_srs = sorted({u["sample_rate"] for u in all_utterances if u["sample_rate"] > 0})
    log(f"    All sample rates   : {all_srs}")
    log(f"    Total utterances   : {len(all_utterances)}")
    log(f"    Total audio        : {sum(all_durations)/3600:.2f} hours")

    # ── 3. Vocabulary ───────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("[3] VOCABULARY (headMic only)")

    head_prompts_by_spk: dict[str, set] = defaultdict(set)
    for u in head_utts_all:
        head_prompts_by_spk[u["speaker_id"]].add(u["prompt_text"])

    all_prompts_union: set[str] = set()
    for prompts in head_prompts_by_spk.values():
        all_prompts_union |= prompts

    # Words shared by ALL speakers
    shared: set[str] = set(all_prompts_union)
    for prompts in head_prompts_by_spk.values():
        shared &= prompts

    # Words present in control speakers only
    control_ids = [s for s in TORGO_SEVERITY if TORGO_SEVERITY[s] == 0]
    control_prompts: set[str] = set()
    for cid in control_ids:
        if cid in head_prompts_by_spk:
            if not control_prompts:
                control_prompts = set(head_prompts_by_spk[cid])
            else:
                control_prompts &= head_prompts_by_spk[cid]

    log(f"\n    Total unique prompts (union): {len(all_prompts_union)}")
    log(f"    Prompts shared by ALL speakers: {len(shared)}")
    log(f"    Prompts shared by all CONTROLS: {len(control_prompts)}")

    # Classify: instruction vs word/sentence
    instruction_prompts = {p for p in shared if p.startswith("[") or p.startswith("(")}
    word_prompts = shared - instruction_prompts
    log(f"    Shared word/sentence prompts: {len(word_prompts)}")
    log(f"    Shared instruction prompts  : {len(instruction_prompts)}")
    log(f"\n    Sample shared words (first 20): {sorted(word_prompts)[:20]}")

    # ── 4. Severity distribution ────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("[4] SEVERITY DISTRIBUTION (headMic)")

    severity_counts: dict[int, int] = defaultdict(int)
    for u in head_utts_all:
        sev = TORGO_SEVERITY.get(u["speaker_id"], -1)
        severity_counts[sev] += 1

    sev_labels = {0: "Control", 1: "Mild-Moderate", 2: "Severe"}
    log(f"\n    {'Severity':<6} {'Label':<16} {'Utterances':>12}")
    log(f"    {'-'*38}")
    for sev in sorted(severity_counts):
        label = sev_labels.get(sev, "Unknown")
        log(f"    {sev:<6} {label:<16} {severity_counts[sev]:>12}")
    log(f"    {'-'*38}")
    log(f"    {'Total':<23} {sum(severity_counts.values()):>12}")

    log(f"\n    Per-speaker utterance counts (headMic):")
    log(f"    {'Speaker':<8} {'Sev':>4} {'Gender':>7} {'Sessions':<25} {'N_utts':>8}")
    log(f"    {'-'*60}")
    for spk_id, st in sorted(speaker_stats.items()):
        log(f"    {spk_id:<8} {st['severity']:>4} {st['gender']:>7} "
            f"{', '.join(st['sessions']):<25} {st['n_head']:>8}")

    # ── 5. Mic type detection ───────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("[5] MICROPHONE CONDITIONS")
    log(f"    headMic  utterances: {len(head_utts_all)}")
    log(f"    arrayMic utterances: {len(array_utts_all)}")
    log("    -> We will USE headMic only (higher quality for dysarthria research)")
    log("    -> arrayMic is retained in raw but excluded from processing")

    # ── 6. File naming convention ───────────────────────────────────────────
    log("\n" + "=" * 70)
    log("[6] FILE NAMING CONVENTION")
    log("    Pattern: {TORGO_ROOT}/{Group}/{SpeakerID}/{Session}/wav_headMic/{NNNN}.wav")
    log("    Example: data/raw/torgo/F/F01/Session1/wav_headMic/0001.wav")
    log("    Prompt:  data/raw/torgo/F/F01/Session1/prompts/0001.txt")
    log("    NNNN: zero-padded integer, not directly a word name")
    log("    Prompt text: plain text file containing the word or sentence prompt")

    # ── Save outputs ────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("[7] SAVING OUTPUTS")

    # Full exploration report
    report_path = LOG_DIR / "torgo_exploration.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"    Exploration report -> {report_path}")

    # Shared vocabulary (only real words, not instructions)
    vocab_path = META_DIR / "shared_vocabulary.txt"
    sorted_vocab = sorted(word_prompts)
    vocab_path.write_text("\n".join(sorted_vocab), encoding="utf-8")
    log(f"    Shared vocabulary  -> {vocab_path}  ({len(sorted_vocab)} entries)")

    log("\n" + "=" * 70)
    log("EXPLORATION COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()

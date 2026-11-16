"""End-to-end reproduction pipeline for ATD on TORGO + UASpeech + IPVS + L2/VCTK.

Runs every canonical experiment in order. Each step is resumable: if its
expected outputs already exist on disk, it is skipped unless ``--force`` is
passed.

Usage
-----
    python reproduce_all.py                  # full TORGO pipeline, skip done
    python reproduce_all.py --force          # re-run every step
    python reproduce_all.py --skip-extract   # reuse existing embeddings
    python reproduce_all.py --include-uaspeech      # also run UA_01..UA_05 + 34
    python reproduce_all.py --include-cross-corpus  # also run CC_* + 23 + 28
    python reproduce_all.py --include-l2arctic      # also run L2_03..L2_05
    python reproduce_all.py --include-vctk          # also run V_03..V_05
    python reproduce_all.py --include-all           # every corpus + cross-corpus
    python reproduce_all.py --smoke          # smoke-test (dry plan, no execution)

The consolidator (``21_consolidate_paper_numbers.py``) always runs LAST, after
every enabled pipeline, so ``results/paper_numbers.json`` is rebuilt from a
complete set of inputs. ``--include-all`` enables every corpus pipeline so the
consolidator sees all of its inputs.

Each step is shelled out as a separate process so a failure in step N does
not cripple the Python state — re-run from the same command. Logs are
written under ``results/logs/reproduce_all/<step>.log``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS = PROJECT_ROOT / "experiments"
RESULTS = PROJECT_ROOT / "results"
LOG_DIR = RESULTS / "logs" / "reproduce_all"


@dataclass
class Step:
    """One pipeline step: a script + the artifact whose presence means 'done'."""

    name: str
    script: str
    output_check: Path | None = None  # if exists & not --force, skip
    optional: bool = False  # if True, errors don't abort pipeline

    @property
    def script_path(self) -> Path:
        return EXPERIMENTS / self.script


# ---------------------------------------------------------------------------
# TORGO pipeline — the spine. Steps 22/24/25/26/27/31/32/33/41/42/rebuild_table1
# are TORGO-derived analyses that depend only on outputs of 06..20 (atd_scores,
# speaker_atd, layer_analysis, etc.), so they live here and run by default.
# ---------------------------------------------------------------------------
TORGO_PIPELINE: list[Step] = [
    Step("01_verify_env",      "01_verify_environment.py",
         RESULTS / "logs" / "environment_check.txt"),
    Step("02_explore",         "02_explore_torgo.py",
         PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"),
    Step("03_preprocess",      "03_preprocess_audio.py",
         PROJECT_ROOT / "data" / "processed" / "audio_16khz"),
    Step("04_extract_ssl",     "04_extract_ssl.py",
         PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"),
    Step("05_build_refs",      "05_build_references.py",
         PROJECT_ROOT / "data" / "processed" / "references" / "reference_manifest.json"),
    Step("06_compute_atd",     "06_compute_atd.py",
         RESULTS / "atd_scores.csv"),  # also writes results/speaker_atd.csv
    Step("07_validate",        "07_validate_atd.py",
         RESULTS / "atd_validation_report.json"),
    Step("08_baselines",       "08_baselines.py",
         RESULTS / "tables" / "table1_main_results.csv"),
    Step("09_layer_analysis",  "09_layer_analysis.py",
         RESULTS / "tables" / "layer_analysis.csv"),  # also writes best_layer.json
    Step("10_loso",            "10_loso_validation.py",
         RESULTS / "tables" / "loso_results.csv"),
    Step("11_ablations",       "11_ablations.py",
         RESULTS / "tables" / "table2_ablations.csv"),
    Step("12_figures",         "12_generate_figures.py",
         RESULTS / "figures" / "fig1_layer_analysis.pdf"),
    Step("13_tables",          "13_generate_tables.py",
         PROJECT_ROOT / "paper" / "tables" / "table1_main_results.tex"),
    Step("14_critical_fixes",  "14_critical_fixes.py",
         RESULTS / "critical_fixes_report.json"),
    Step("15_perturbation",    "15_perturbation_validation.py",
         RESULTS / "perturbation_atd.csv", optional=True),
    Step("16_phoneme_pilot",   "16_phoneme_pilot.py",
         RESULTS / "phoneme_summary.json", optional=True),
    Step("17_ipvs_bootstrap",  "17_ipvs_vowel_bootstrap.py",
         RESULTS / "cross_corpus" / "ipvs_vowel_bootstrap.json", optional=True),
    Step("18_torgo_loo",       "18_torgo_loo_diagnostics.py",
         RESULTS / "torgo_loo_diagnostics.json"),
    Step("19_torgo_bootstrap", "19_torgo_bootstrap_ci.py",
         RESULTS / "torgo_bootstrap_ci.json"),
    Step("20_mfcc_perturbation", "20_mfcc_perturbation.py",
         RESULTS / "perturbation_atd_vs_mfcc.json", optional=True),
    # --- TORGO-derived analyses (depend on 06/09/20 outputs) ---
    Step("22_streaming_latency", "22_streaming_latency_bench.py",
         RESULTS / "streaming_parity.json", optional=True),
    Step("24_manifold_umap",   "24_manifold_umap.py",
         RESULTS / "manifold_umap_coords.csv", optional=True),
    Step("26_atd_confidence",  "26_atd_confidence_score.py",
         RESULTS / "atd_confidence.csv"),
    Step("27_perturbation_layers", "27_perturbation_all_layers.py",
         RESULTS / "perturbation_layer_hierarchy.json", optional=True),
    Step("29_clinical_reports", "29_clinical_reports.py",
         RESULTS / "clinical_reports", optional=True),  # LLM path
    Step("31_power_analysis",  "31_power_analysis.py",
         RESULTS / "power_analysis.json"),
    Step("32_mixed_effects",   "32_mixed_effects_per_utterance.py",
         RESULTS / "torgo_mixed_effects.json"),
    Step("33_bayesian_rho",    "33_bayesian_rho_posterior.py",
         RESULTS / "torgo_bayesian_rho.json"),
    Step("41_loso_jackknife",  "41_loso_jackknife_ci.py",
         RESULTS / "torgo_loso_jackknife_ci.json", optional=True),
    Step("42_finetuned_ssl",   "42_finetuned_ssl_baseline.py",
         RESULTS / "finetuned_ssl_baseline.json", optional=True),
    Step("rebuild_table1",     "rebuild_table1.py",
         RESULTS / "tables" / "table1_main_results.csv"),
]

UASPEECH_PIPELINE: list[Step] = [
    Step("UA_01_explore",      "UA_01_explore_uaspeech.py",
         PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_uaspeech.csv",
         optional=True),
    Step("UA_02_preprocess",   "UA_02_preprocess_uaspeech.py", optional=True),
    Step("UA_03_extract_ssl",  "UA_03_extract_ssl_uaspeech.py", optional=True),
    Step("UA_04_atd",          "UA_04_cross_corpus_atd.py",
         RESULTS / "uaspeech_atd_scores.csv", optional=True),
    Step("UA_05_combined",     "UA_05_combined_analysis.py",
         RESULTS / "combined_validation_report.json", optional=True),
    Step("34_ua_intelligibility", "34_uaspeech_intelligibility_anchor.py",
         RESULTS / "uaspeech_intelligibility_anchor.json", optional=True),
]


CROSS_CORPUS_PIPELINE: list[Step] = [
    Step("CC_01_extract_ipvs", "CC_01_extract_ipvs.py",
         PROJECT_ROOT / "data" / "embeddings" / "ssl_index_ipvs_xlsr_300m.json",
         optional=True),
    Step("CC_02_within",       "CC_02_within_corpus_atd.py",
         RESULTS / "cross_corpus" / "ipvs_atd_layer9.csv", optional=True),
    Step("CC_03_zero_shot",    "CC_03_zero_shot_cross.py",
         RESULTS / "cross_corpus" / "zero_shot_torgo2ipvs.json", optional=True),
    Step("CC_04_pooled",       "CC_04_pooled_analysis.py",
         RESULTS / "cross_corpus" / "pooled_lme.json", optional=True),
    Step("CC_05_layer_sweep",  "CC_05_layer_sweep.py",
         RESULTS / "cross_corpus" / "layer_sweep_wavlm_large.csv", optional=True),
    Step("23_ipvs_mfcc",       "23_ipvs_mfcc_baseline.py",
         RESULTS / "cross_corpus" / "ipvs_mfcc_baseline.json", optional=True),
    Step("28_ssl_backbones",   "28_cross_ssl_backbones.py",
         RESULTS / "cross_corpus" / "ssl_backbone_compare.json", optional=True),
]


L2ARCTIC_PIPELINE: list[Step] = [
    Step("L2_03_extract_ssl",  "L2_03_extract_ssl.py",
         PROJECT_ROOT / "data" / "processed" / "features" / "l2arctic" / "extraction_log.json",
         optional=True),
    Step("L2_04_atd_loco",     "L2_04_atd_loco.py",
         RESULTS / "l2arctic_speaker_atd.csv", optional=True),
    Step("L2_05_specificity",  "L2_05_specificity_test.py",
         RESULTS / "l2arctic_specificity.json", optional=True),
]


VCTK_PIPELINE: list[Step] = [
    Step("V_03_extract_ssl",   "V_03_extract_ssl_vctk.py",
         PROJECT_ROOT / "data" / "processed" / "features" / "vctk" / "extraction_log.json",
         optional=True),
    Step("V_04_atd_loco",      "V_04_atd_loco_vctk.py",
         RESULTS / "vctk_speaker_atd.csv", optional=True),
    Step("V_05_specificity",   "V_05_specificity_vctk.py",
         RESULTS / "vctk_specificity.json", optional=True),
]


# Cross-corpus aggregation that needs MULTIPLE corpora's speaker-ATD CSVs.
#   30 (z-rank dx)  : speaker_atd + uaspeech_speaker_atd + ipvs_speaker_atd
#   25 (diff dx)    : speaker_atd + uaspeech_speaker_atd + ipvs_speaker_atd
#   L2_06 (pooled)  : torgo + uaspeech + l2arctic + vctk speaker ATDs
# These run after the per-corpus pipelines, before the consolidator.
MULTI_CORPUS_PIPELINE: list[Step] = [
    Step("25_diff_dx",         "25_diff_dx_classifier.py",
         RESULTS / "differential_dx_loso.json", optional=True),
    Step("30_zrank_dx",        "30_per_corpus_zrank_classifier.py",
         RESULTS / "differential_dx_zrank_loso.json", optional=True),
    Step("L2_06_pooled",       "L2_06_pooled_mixed_effects.py",
         RESULTS / "cross_corpus_pooled_mixed_effects.json", optional=True),
]


# Always last: rebuild paper_numbers.json from every input produced above.
CONSOLIDATE_STEP = Step(
    "21_consolidate_numbers", "21_consolidate_paper_numbers.py",
    None,  # always re-run (output overwritten)
)


def _run_step(step: Step, *, force: bool, dry_run: bool, env_overrides: dict[str, str]) -> bool:
    """Run a single step. Return True on success (or skipped), False on hard fail."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.name}.log"

    if not step.script_path.exists():
        msg = f"[SKIP] {step.name}: script not found at {step.script_path}"
        print(msg)
        if step.optional:
            return True
        return False

    if step.output_check is not None and step.output_check.exists() and not force:
        print(f"[SKIP] {step.name}: output exists ({step.output_check.name})")
        return True

    cmd = [sys.executable, str(step.script_path)]
    print(f"[RUN ] {step.name}: {' '.join(shlex.quote(c) for c in cmd)}")

    if dry_run:
        return True

    env = os.environ.copy()
    env.update(env_overrides)

    started = _dt.datetime.now()
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write(f"# {step.name} | started {started.isoformat()}\n")
            logf.write(f"# cmd: {' '.join(cmd)}\n\n")
            logf.flush()
            result = subprocess.run(
                cmd, cwd=PROJECT_ROOT, env=env, stdout=logf,
                stderr=subprocess.STDOUT, check=False,
            )
    except KeyboardInterrupt:
        print(f"\n[ABRT] {step.name}: interrupted by user")
        raise

    elapsed = (_dt.datetime.now() - started).total_seconds()
    if result.returncode != 0:
        print(f"[FAIL] {step.name}: exit={result.returncode} after {elapsed:.1f}s — see {log_path}")
        return step.optional
    print(f"[ OK ] {step.name}: {elapsed:.1f}s")
    return True


def _summary(passed: list[str], failed: list[str], skipped: list[str]) -> None:
    print("\n" + "=" * 72)
    print(f"  reproduce_all summary — {_dt.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 72)
    print(f"  passed:  {len(passed)}  ({', '.join(passed) if passed else '—'})")
    print(f"  skipped: {len(skipped)} ({', '.join(skipped) if skipped else '—'})")
    print(f"  failed:  {len(failed)}  ({', '.join(failed) if failed else '—'})")
    print("=" * 72)


def build_pipeline(args: argparse.Namespace) -> list[Step]:
    """Assemble the ordered list of steps for the given CLI args.

    The consolidator (21) is always appended LAST, after every enabled
    pipeline, so paper_numbers.json is rebuilt from a complete input set.
    """
    include_ua = args.include_uaspeech or args.include_all
    include_cc = args.include_cross_corpus or args.include_all
    include_l2 = args.include_l2arctic or args.include_all
    include_vctk = args.include_vctk or args.include_all

    pipeline: list[Step] = list(TORGO_PIPELINE)
    if include_ua:
        pipeline.extend(UASPEECH_PIPELINE)
    if include_cc:
        pipeline.extend(CROSS_CORPUS_PIPELINE)
    if include_l2:
        pipeline.extend(L2ARCTIC_PIPELINE)
    if include_vctk:
        pipeline.extend(VCTK_PIPELINE)
    # Multi-corpus aggregation runs after per-corpus pipelines.
    if include_ua or include_cc or include_l2 or include_vctk:
        pipeline.extend(MULTI_CORPUS_PIPELINE)
    # Consolidator always last.
    pipeline.append(CONSOLIDATE_STEP)

    if args.skip_extract:
        extract_names = {"04_extract_ssl", "UA_03_extract_ssl", "CC_01_extract_ipvs",
                         "L2_03_extract_ssl", "V_03_extract_ssl"}
        pipeline = [s for s in pipeline if s.name not in extract_names]
    if args.from_step:
        names = [s.name for s in pipeline]
        if args.from_step not in names:
            raise SystemExit(f"--from-step {args.from_step!r} not in pipeline {names}")
        pipeline = pipeline[names.index(args.from_step):]
    if args.only:
        only = [s for s in pipeline if s.name == args.only]
        if not only:
            raise SystemExit(f"--only {args.only!r} did not match any step")
        pipeline = only
    return pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step regardless of existing outputs.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip all *_extract_ssl steps (use cached embeddings).")
    parser.add_argument("--include-uaspeech", action="store_true",
                        help="Also run UA_01..UA_05 + 34 after the TORGO pipeline.")
    parser.add_argument("--include-cross-corpus", action="store_true",
                        help="Also run CC_01..CC_05 + 23 + 28 (IPVS).")
    parser.add_argument("--include-l2arctic", action="store_true",
                        help="Also run L2_03..L2_05 (L2-ARCTIC accent specificity).")
    parser.add_argument("--include-vctk", action="store_true",
                        help="Also run V_03..V_05 (VCTK native-accent specificity).")
    parser.add_argument("--include-all", action="store_true",
                        help="Enable every corpus + cross-corpus pipeline.")
    parser.add_argument("--smoke", action="store_true",
                        help="Dry-run: print step plan, don't execute.")
    parser.add_argument("--from-step", default=None,
                        help="Start from named step (e.g. 06_compute_atd).")
    parser.add_argument("--only", default=None,
                        help="Run only this single step (by name).")
    args = parser.parse_args(argv)

    pipeline = build_pipeline(args)

    print(f"Pipeline: {len(pipeline)} step(s)")
    for s in pipeline:
        print(f"  - {s.name}: {s.script}")
    print()

    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    env_overrides = {"PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "42"}

    for step in pipeline:
        out_existed = step.output_check is not None and step.output_check.exists() and not args.force
        ok = _run_step(step, force=args.force, dry_run=args.smoke, env_overrides=env_overrides)
        if not ok:
            failed.append(step.name)
            if not step.optional:
                _summary(passed, failed, skipped)
                return 1
            continue
        if out_existed:
            skipped.append(step.name)
        else:
            passed.append(step.name)

    _summary(passed, failed, skipped)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

# VERITAS Phase 1 · Milestone 1 — Faithfulness Core + Inversion (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and offline-validate the novel core of VERITAS — a GT-free cycle-consistency *faithfulness* signal that detects when frozen-SSL articulatory inversion normalizes disordered speech toward "healthy" — on real local TORGO data, no external download required.

**Architecture:** Wrap the open-source SPARC coder (frozen WavLM → 12 EMA tract variables, invertible). Compute faithfulness by re-synthesizing audio from the articulatory code and measuring how far the re-synthesis drifts from the input in WavLM-Large feature space (cycle consistency), fused with an out-of-distribution score fit on healthy speech. Validate that faithfulness drops on dysarthric input (H2) using the 15 real TORGO speakers already on disk.

**Tech Stack:** Python 3.11, PyTorch 2.2.2, `speech-articulatory-coding` (SPARC), `transformers` 4.40 (WavLM-Large), `torchaudio`, `librosa`, `scipy`, existing repo perturbation lib (`src/perturbations.py`) and severity map (`src/constants.py`). Env: conda `dysarthria`, python at `C:\Users\Dell\.conda\envs\dysarthria\python.exe`. Seed 42.

**Scope note:** M1 delivers inversion + faithfulness + H2 (faithfulness-vs-severity) on local data. EMA ground-truth validations (H1 faithfulness-vs-inversion-error on HPRC; H3 anti-normalization on TORGO-EMA) are **data-gated** and deferred to M1b once HPRC / TORGO-EMA are downloaded — flagged as the final task.

---

## File Structure

- Create `src/veritas/__init__.py` — package marker.
- Create `src/veritas/wavlm_features.py` — self-contained WavLM-Large frame-feature extractor (layer 9, 50 fps, 1024-d). One responsibility: audio array → features.
- Create `src/veritas/inversion.py` — `SPARCInverter`: thin wrapper over SPARC `encode`/`decode` + `resynth`. One responsibility: audio ↔ articulatory code.
- Create `src/veritas/faithfulness.py` — `cycle_faithfulness`, `OODScorer`, `fused_faithfulness`. The novel core.
- Create `experiments/veritas/V1_faithfulness_vs_severity.py` — H2 experiment over TORGO.
- Create tests: `tests/veritas/test_inversion.py`, `tests/veritas/test_faithfulness.py`, `tests/veritas/test_wavlm_features.py`.
- Create `results/veritas/` (output dir).

Reuse (do not rewrite): `src/perturbations.py` (real DSP perturbations of real audio, for the faithfulness discrimination test), `src/constants.py` (`TORGO_SEVERITY`, `TORGO_CONTROL_SPEAKERS`, `RANDOM_SEED`), `src/seed_utils.py`.

---

## Task 0: Environment + SPARC smoke test

**Files:** none (env only); scratch check.

- [ ] **Step 1: Install SPARC into the conda env**

Run:
```bash
"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pip install speech-articulatory-coding
```
Expected: installs `sparc` + deps (torchcrepe etc.), exit 0.

- [ ] **Step 2: Smoke-test SPARC on a real TORGO wav**

Run (pick any existing control wav; adjust path to a real file found via `Get-ChildItem FC01 -Recurse -Filter *.wav | select -First 1`):
```bash
"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -c "
from sparc import load_model
import glob
wav = glob.glob(r'C:/Users/Dell/Desktop/Dysar/FC/**/*.wav', recursive=True)[0]
coder = load_model('en', device='cpu')
code = coder.encode(wav)
print('ema', code['ema'].shape, 'keys', sorted(code.keys()))
w = coder.decode(**code); print('resynth', getattr(w,'shape',len(w)), 'sr', coder.sr)
"
```
Expected: prints `ema (L, 12)` and a resynth waveform + a sample rate. If `encode` rejects a path, note the accepted input form (array + sr) — the wrapper in Task 2 must match reality.

- [ ] **Step 3: Commit the dependency pin**

Add `speech-articulatory-coding` to `requirements-core.txt`, then:
```bash
git add requirements-core.txt && git commit -m "build: add SPARC (speech-articulatory-coding) dependency for VERITAS"
```

---

## Task 1: WavLM feature extractor

**Files:**
- Create: `src/veritas/wavlm_features.py`
- Create: `src/veritas/__init__.py` (empty)
- Test: `tests/veritas/test_wavlm_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/veritas/test_wavlm_features.py
import numpy as np
from src.veritas.wavlm_features import WavLMFeatures

def test_features_shape_and_rate():
    fx = WavLMFeatures(layer=9, device="cpu")
    audio = np.zeros(16000, dtype=np.float32)  # 1.0 s of silence @ 16 kHz
    feats = fx.extract(audio, sr=16000)
    assert feats.ndim == 2 and feats.shape[1] == 1024
    # WavLM-Large is ~50 fps -> ~49-50 frames for 1 s
    assert 45 <= feats.shape[0] <= 51
```

- [ ] **Step 2: Run test, verify it fails**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_wavlm_features.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/veritas/wavlm_features.py
from __future__ import annotations
import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoFeatureExtractor

class WavLMFeatures:
    """Frozen WavLM-Large frame features at a chosen transformer layer (50 fps)."""
    def __init__(self, layer: int = 9, device: str = "cpu", model_name: str = "microsoft/wavlm-large"):
        self.layer = layer
        self.device = device
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if sr != 16000:
            audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
        inputs = self.fe(audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        hs = self.model(inputs).hidden_states           # tuple: [embeddings, layer1..layer24]
        return hs[self.layer].squeeze(0).cpu().numpy()  # (T, 1024)
```

- [ ] **Step 4: Run test, verify pass**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_wavlm_features.py -v`
Expected: PASS (first run downloads WavLM-Large; already cached in this repo's prior work).

- [ ] **Step 5: Commit**

```bash
git add src/veritas/__init__.py src/veritas/wavlm_features.py tests/veritas/test_wavlm_features.py
git commit -m "feat(veritas): WavLM-Large frame-feature extractor"
```

---

## Task 2: SPARC inversion wrapper

**Files:**
- Create: `src/veritas/inversion.py`
- Test: `tests/veritas/test_inversion.py`

- [ ] **Step 1: Write the failing test** (uses a real TORGO wav discovered at runtime)

```python
# tests/veritas/test_inversion.py
import glob, numpy as np
from src.veritas.inversion import SPARCInverter

WAV = glob.glob(r"C:/Users/Dell/Desktop/Dysar/FC/**/*.wav", recursive=True)[0]

def test_encode_decode_roundtrip():
    inv = SPARCInverter(device="cpu")
    code = inv.encode(WAV)
    assert code["ema"].shape[1] == 12
    audio16k = inv.resynth(WAV)           # decode + resample to 16 kHz
    assert audio16k.ndim == 1 and audio16k.dtype == np.float32
    assert audio16k.shape[0] > 1600       # >0.1 s of audio
```

- [ ] **Step 2: Run test, verify it fails**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_inversion.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** (adapt `encode` input form to what Task 0 Step 2 revealed)

```python
# src/veritas/inversion.py
from __future__ import annotations
import numpy as np
import torch, torchaudio
from sparc import load_model

class SPARCInverter:
    """Wrap SPARC: audio -> articulatory code (ema+pitch+loudness+spk) -> audio."""
    def __init__(self, ckpt: str = "en", device: str = "cpu"):
        self.coder = load_model(ckpt, device=device)
        self.sr = int(getattr(self.coder, "sr", 16000))

    def encode(self, wav) -> dict:
        return self.coder.encode(wav)          # wav = path or np array per Task 0 finding

    def decode(self, code: dict) -> np.ndarray:
        w = self.coder.decode(**code)
        return np.asarray(w, dtype=np.float32).squeeze()

    def resynth(self, wav, out_sr: int = 16000) -> np.ndarray:
        code = self.encode(wav)
        w = self.decode(code)
        if self.sr != out_sr:
            w = torchaudio.functional.resample(torch.from_numpy(w), self.sr, out_sr).numpy()
        return np.asarray(w, dtype=np.float32)
```

- [ ] **Step 4: Run test, verify pass**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_inversion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/veritas/inversion.py tests/veritas/test_inversion.py
git commit -m "feat(veritas): SPARC inversion wrapper (encode/decode/resynth)"
```

---

## Task 3: Cycle-consistency faithfulness (the novel core)

**Files:**
- Create: `src/veritas/faithfulness.py`
- Test: `tests/veritas/test_faithfulness.py`

- [ ] **Step 1: Write the failing test** — faithfulness must be HIGHER for a clean control utterance than for the SAME utterance after a strong real DSP distortion (formant shift). Uses the existing real-audio perturbation lib.

```python
# tests/veritas/test_faithfulness.py
import glob, numpy as np, librosa
from src.veritas.faithfulness import cycle_faithfulness
from src.veritas.inversion import SPARCInverter
from src.veritas.wavlm_features import WavLMFeatures
from src.perturbations import formant_shift  # real DSP on real audio

WAV = glob.glob(r"C:/Users/Dell/Desktop/Dysar/FC/**/*.wav", recursive=True)[0]

def test_faithfulness_drops_under_distortion():
    inv, fx = SPARCInverter(device="cpu"), WavLMFeatures(device="cpu")
    x, sr = librosa.load(WAV, sr=16000)
    x_bad = formant_shift(x, sr, scale=1.15)          # strong articulatory distortion
    f_clean = cycle_faithfulness(x, sr, inv, fx).mean()
    f_bad   = cycle_faithfulness(x_bad, sr, inv, fx).mean()
    assert 0.0 <= f_bad <= f_clean <= 1.0
    assert f_clean - f_bad > 0.02                      # measurable drop
```
(If `formant_shift`'s signature differs, match `src/perturbations.py` exactly — check it first.)

- [ ] **Step 2: Run test, verify it fails**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_faithfulness.py::test_faithfulness_drops_under_distortion -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# src/veritas/faithfulness.py
from __future__ import annotations
import numpy as np

def _cos_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    T = min(len(a), len(b))
    a, b = a[:T], b[:T]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return num / den                          # cosine in [-1, 1]

def cycle_faithfulness(audio: np.ndarray, sr: int, inverter, wavlm) -> np.ndarray:
    """GT-free per-frame faithfulness in [0,1]: how much the articulatory
    bottleneck preserves the input, measured in WavLM space after re-synthesis.
    audio->encode->decode->re-encode(WavLM); compare WavLM(x) vs WavLM(x_hat)."""
    xhat = inverter.resynth(audio if sr == 16000 else _to16k(audio, sr))
    fx   = wavlm.extract(audio, sr=sr)
    fxh  = wavlm.extract(xhat, sr=16000)
    cos  = _cos_per_frame(fx, fxh)
    return np.clip((cos + 1.0) / 2.0, 0.0, 1.0)   # map [-1,1] -> [0,1]

def _to16k(audio, sr):
    import torchaudio, torch
    return torchaudio.functional.resample(torch.from_numpy(np.asarray(audio, np.float32)), sr, 16000).numpy()
```

- [ ] **Step 4: Run test, verify pass**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_faithfulness.py::test_faithfulness_drops_under_distortion -v`
Expected: PASS. (If the drop is < 0.02, inspect: distortion strength, resynth length alignment. Do NOT weaken the assertion to force a pass — diagnose.)

- [ ] **Step 5: Commit**

```bash
git add src/veritas/faithfulness.py tests/veritas/test_faithfulness.py
git commit -m "feat(veritas): GT-free cycle-consistency faithfulness metric"
```

---

## Task 4: Out-of-distribution scorer (fused signal)

**Files:**
- Modify: `src/veritas/faithfulness.py` (add `OODScorer`, `fused_faithfulness`)
- Modify: `tests/veritas/test_faithfulness.py` (add test)

- [ ] **Step 1: Write the failing test** — OOD score for dysarthric-speaker frames > control-speaker frames, after fitting on controls.

```python
# add to tests/veritas/test_faithfulness.py
import glob, librosa, numpy as np
from src.veritas.faithfulness import OODScorer
from src.veritas.wavlm_features import WavLMFeatures

def _feats(pattern, fx, n=3):
    out = []
    for w in glob.glob(pattern, recursive=True)[:n]:
        x, sr = librosa.load(w, sr=16000)
        out.append(fx.extract(x, sr=16000))
    return np.concatenate(out, 0)

def test_ood_higher_for_dysarthric():
    fx = WavLMFeatures(device="cpu")
    ctrl = _feats(r"C:/Users/Dell/Desktop/Dysar/FC/**/*.wav", fx)      # controls
    dys  = _feats(r"C:/Users/Dell/Desktop/Dysar/F/**/*.wav", fx)       # dysarthric (F01/F03/F04)
    ood = OODScorer().fit(ctrl)
    assert ood.score(dys).mean() > ood.score(ctrl).mean()
```

- [ ] **Step 2: Run, verify fail**

Run: `... -m pytest tests/veritas/test_faithfulness.py::test_ood_higher_for_dysarthric -v`
Expected: FAIL (OODScorer undefined).

- [ ] **Step 3: Implement** (append to `src/veritas/faithfulness.py`)

```python
class OODScorer:
    """Diagonal-covariance Mahalanobis distance to the healthy feature manifold."""
    def fit(self, healthy_feats: np.ndarray):
        self.mu = healthy_feats.mean(0)
        self.var = healthy_feats.var(0) + 1e-6
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        d = (feats - self.mu) ** 2 / self.var
        return np.sqrt(d.sum(1))                 # per-frame OOD distance

def fused_faithfulness(cycle_F: np.ndarray, ood: np.ndarray, ood_ref: float) -> np.ndarray:
    """Combine cycle faithfulness with an OOD penalty normalised by a healthy
    reference OOD level. Returns per-frame faithfulness in [0,1]."""
    T = min(len(cycle_F), len(ood))
    ood_pen = np.clip(1.0 - (ood[:T] / (ood_ref + 1e-8) - 1.0), 0.0, 1.0)
    return np.clip(0.5 * cycle_F[:T] + 0.5 * ood_pen, 0.0, 1.0)
```

- [ ] **Step 4: Run, verify pass**

Run: `... -m pytest tests/veritas/test_faithfulness.py -v`
Expected: PASS (both faithfulness tests).

- [ ] **Step 5: Commit**

```bash
git add src/veritas/faithfulness.py tests/veritas/test_faithfulness.py
git commit -m "feat(veritas): OOD scorer + fused faithfulness"
```

---

## Task 5: H2 experiment — faithfulness vs severity on TORGO

**Files:**
- Create: `experiments/veritas/V1_faithfulness_vs_severity.py`
- Test: `tests/veritas/test_v1_experiment.py` (smoke, tiny subset)

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/veritas/test_v1_experiment.py
import json, subprocess, sys, os
def test_v1_runs_smoke(tmp_path):
    out = tmp_path / "faith.json"
    r = subprocess.run([sys.executable, "experiments/veritas/V1_faithfulness_vs_severity.py",
                        "--limit-per-speaker", "2", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert "spearman_rho" in d and "per_speaker" in d and len(d["per_speaker"]) >= 3
```

- [ ] **Step 2: Run, verify fail**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_v1_experiment.py -v`
Expected: FAIL (script missing).

- [ ] **Step 3: Implement**

```python
# experiments/veritas/V1_faithfulness_vs_severity.py
"""H2: does GT-free faithfulness DROP with dysarthria severity? (real TORGO audio)"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, librosa
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.seed_utils import set_global_seed
from src.constants import TORGO_SEVERITY, TORGO_CONTROL_SPEAKERS
from src.veritas.inversion import SPARCInverter
from src.veritas.wavlm_features import WavLMFeatures
from src.veritas.faithfulness import cycle_faithfulness

SPK_DIR = {  # speaker_id -> top-level group dir (TORGO layout on disk)
    **{s: ("FC" if s.startswith("FC") else "F") for s in TORGO_SEVERITY if s.startswith("F")},
    **{s: ("MC" if s.startswith("MC") else "M") for s in TORGO_SEVERITY if s.startswith("M")},
}

def speaker_wavs(spk, limit):
    grp = SPK_DIR[spk]
    pats = glob.glob(str(ROOT / grp / "**" / f"*{spk}*" / "**" / "*.wav"), recursive=True) \
        or glob.glob(str(ROOT / grp / "**" / "*.wav"), recursive=True)
    return [w for w in pats if spk in w][:limit]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-speaker", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "results" / "veritas" / "faithfulness_severity.json"))
    args = ap.parse_args()
    set_global_seed(42)
    inv, fx = SPARCInverter(device="cpu"), WavLMFeatures(device="cpu")
    per_spk = {}
    for spk, sev in TORGO_SEVERITY.items():
        vals = []
        for w in speaker_wavs(spk, args.limit_per_speaker):
            try:
                x, sr = librosa.load(w, sr=16000)
                if len(x) < 1600: continue
                vals.append(float(cycle_faithfulness(x, sr, inv, fx).mean()))
            except Exception as e:
                print(f"skip {w}: {e}")
        if vals:
            per_spk[spk] = {"severity": sev, "mean_faithfulness": float(np.mean(vals)),
                            "n": len(vals), "is_control": spk in TORGO_CONTROL_SPEAKERS}
    sev = [v["severity"] for v in per_spk.values()]
    fth = [v["mean_faithfulness"] for v in per_spk.values()]
    rho, p = spearmanr(sev, fth) if len(per_spk) > 2 else (float("nan"), float("nan"))
    out = {"hypothesis": "faithfulness DROPS with severity (expect negative rho)",
           "spearman_rho": float(rho), "p_value": float(p),
           "n_speakers": len(per_spk), "per_speaker": per_spk}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"rho={rho:.4f} p={p:.4f} n={len(per_spk)} -> {args.out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, verify pass**

Run: `"C:/Users/Dell/.conda/envs/dysarthria/python.exe" -m pytest tests/veritas/test_v1_experiment.py -v`
Expected: PASS. Then run the full experiment (real result):
```bash
"C:/Users/Dell/.conda/envs/dysarthria/python.exe" experiments/veritas/V1_faithfulness_vs_severity.py
```
Expected: prints `rho=<value>`; writes `results/veritas/faithfulness_severity.json`. **Record the honest rho.** Negative rho supports H2 (faithfulness drops with severity). If rho is ~0 or positive, that is a real, informative result about the metric — report it, do not massage it.

- [ ] **Step 5: Commit**

```bash
git add experiments/veritas/V1_faithfulness_vs_severity.py tests/veritas/test_v1_experiment.py results/veritas/faithfulness_severity.json
git commit -m "feat(veritas): H2 experiment — faithfulness vs TORGO severity"
```

---

## Task 6: Wire into repro + document the honest M1 result

**Files:**
- Modify: `reproduce_all.py` (add V1 as a VERITAS step)
- Create: `docs/veritas/M1_RESULTS.md`

- [ ] **Step 1** Add a `Step` for `experiments/veritas/V1_faithfulness_vs_severity.py` to `reproduce_all.py` following the existing `Step` dataclass pattern (output_check = `results/veritas/faithfulness_severity.json`).

- [ ] **Step 2** Write `docs/veritas/M1_RESULTS.md` stating: the metric, the H2 rho/p actually obtained, the discrimination-under-distortion test result, and — honestly — what is and isn't yet validated (EMA validations H1/H3 deferred to M1b, pending HPRC/TORGO-EMA).

- [ ] **Step 3: Commit**

```bash
git add reproduce_all.py docs/veritas/M1_RESULTS.md
git commit -m "docs(veritas): wire V1 into reproduce_all + honest M1 results"
```

---

## Task 7 (DATA-GATED — M1b): EMA ground-truth validation

Deferred until real articulatory data is on disk. Do NOT fabricate.
- **Download** HPRC EMA (open) + confirm TORGO-EMA availability.
- **H1:** correlate per-frame `cycle_faithfulness` with true inversion error (‖SPARC ema − reference EMA‖) on held-out healthy HPRC. Expect negative correlation (low faithfulness ↔ high inversion error).
- **H3:** on TORGO's ~2 real dysarthric EMA speakers, show faithfulness-gated inversion preserves true articulatory deviation vs naive SPARC (which regresses to normal), reported at true n.
Write these as their own tasks when data lands.

---

## Self-Review

- **Spec coverage:** M1 implements the design's §1 faithfulness core (cycle-consistency + OOD fusion = Tasks 3–4) and H2 (Task 5). H1/H3/H4 and the live system (renderer/streaming/deviation engine) are explicitly out of M1 scope → later milestones (M2 deviation+references, M3 streaming, M4 renderer/UI, M1b EMA validation). No design requirement is silently dropped.
- **Placeholder scan:** every code step contains real, runnable code; the only deferred item (Task 7) is explicitly data-gated, not a hidden TODO in shipped code.
- **Type consistency:** `WavLMFeatures.extract(audio, sr)->(T,1024)`, `SPARCInverter.encode/decode/resynth`, `cycle_faithfulness(audio, sr, inverter, wavlm)->(T,)`, `OODScorer.fit(feats).score(feats)->(T,)`, `fused_faithfulness(cycle_F, ood, ood_ref)->(T,)` — names/signatures consistent across tasks and the V1 experiment.
- **Honesty guard:** Task 5 Step 4 and Task 3 Step 4 explicitly forbid weakening assertions to force a pass; H2 result is reported whatever its sign.

**Known real-world risks the executor must handle (not placeholders):** SPARC `encode` input form (path vs array) — resolved empirically in Task 0; SPARC `sr` value — read from `coder.sr`; TORGO on-disk path/glob for per-speaker wavs — the `speaker_wavs` glob may need adjusting to the actual `F/FC/M/MC` subfolder naming (verify with a real listing before the full run).

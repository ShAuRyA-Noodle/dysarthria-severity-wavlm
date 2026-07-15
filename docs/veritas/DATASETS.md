# VERITAS Milestone-1b — Articulatory & Clinician-Rated Datasets

Ground-truth data for validating the GT-free faithfulness metric against **real
articulator measurements** (H1: faithfulness vs inversion error on healthy
speech; H3: anti-normalization on dysarthric speech) and against **clinician
perceptual ratings**.

Compiled 2026-07-09. Every "on disk" claim below was verified against the local
filesystem; every remote URL was probed live (HTTP status shown).

---

## 0. Status summary

| Dataset | Role | Status | On disk | Access |
|---|---|---|---|---|
| **TORGO EMA** (local) | **H3 pathological EMA + H1 healthy EMA** | ✅ **REAL, VERIFIED, LOCAL** | 7,177 `.pos` (1.45 GB) + wav/amps (7.06 GB total) | already present — **no download needed** |
| **TORGO Frenchay ratings** (local) | clinician-rated severity | ✅ **REAL, LOCAL** | 11 `Notes/*.csv` | already present |
| **HPRC (Haskins)** | H1 primary healthy EMA | ⛔ **GATED** — canonical link dead | no | contact Haskins Labs / M. Tiede (steps below) |
| **NeuroVoz** | GRBAS clinician-rated PD speech | ⛔ **GATED** — Zenodo DUA request | no | Zenodo "Request access" + DUA (steps below) |
| USC 75-speaker rtMRI | (reference) | 📄 document-only | no | open, figshare (huge, ~ hundreds of GB) |
| MNGU0 EMA | open H1 healthy-EMA fallback | 📄 document-only | no | free, register at mngu0.org |
| AUSpeech (audio-ultrasound) | (reference) | 📄 document-only | no | ScienceDB, 676 GB |

**Bottom line:** the single most important asset — **real 3-D EMA articulatory
ground truth for both dysarthric and control speakers — is already on disk**
(TORGO), fully decoded and integrity-checked below. It unblocks **both H1 (7
control speakers) and H3 (7 dysarthric speakers)** with zero download. The two
requested downloads (HPRC, NeuroVoz) are both **access-gated** and cannot be
fetched programmatically; exact request procedures are documented so they can be
retrieved manually. HuggingFace is unreachable from this machine (TLS failure
even via PowerShell/curl), so HF mirrors are not an option.

---

## 1. TORGO EMA — LOCAL, REAL, VERIFIED  ⭐ (critical unblock)

3-D electromagnetic articulography (Carstens **AG500**) + time-aligned acoustics
from dysarthric and control speakers. This is the pathological EMA ground truth
for **H3** and, via its control speakers, healthy EMA for **H1**.

- **Source / provenance:** TORGO database, Rudzicz, Namasivayam & Wolff (2012),
  *Language Resources and Evaluation* 46(4):523–541,
  <https://doi.org/10.1007/s10579-011-9145-0>. Free research release:
  <https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html> (also LDC
  `LDC2012S02`). The local `F/ FC/ M/ MC/` trees are this release.
  (Note: the sibling `DS_10283_3443/` dir is **VCTK**, Edinburgh DataShare
  10283/3443 — *not* TORGO.)
- **License:** free for academic/research use (TORGO release terms). Not for
  commercial redistribution.
- **On-disk paths:**
  `C:\Users\Dell\Desktop\Dysar\{F,FC,M,MC}\<SPK>\Session{1,2,3}\`
  EMA lives in the `pos/`, `rawpos/`, `amps/` subfolders; synchronous audio in
  `wavall/`.
- **Size (measured):** `pos/` **1,450.3 MB** (7,177 files) · `rawpos/`
  (incl. `posamps/`) **2,695.5 MB** · `amps/` **1,244.1 MB** · `wavall/`
  **1,669.7 MB** → **≈ 7.06 GB** of EMA + EMA-synchronous audio.
- **Integrity check (run 2026-07-09):** all **7,177** `.pos` files have
  `size % 336 == 0` (exact multiple of 84 float32 columns × 4 bytes); **0**
  malformed, **0** zero-byte. Frame counts 63–8,631 → 0.32 s–43.16 s @ 200 Hz.
  `pos` count (7,177) matches `rawpos`/`amps` (7,183) up to a handful of missing
  files. This is strong evidence the articulatory data is complete and uncorrupted.

### 1.1 Speaker inventory (EMA-bearing speakers only)

TORGO group codes: `F`=female dysarthric, `M`=male dysarthric, `FC`=female
control, `MC`=male control.

| Group | Speaker | Class | `.pos` utts (all sessions) |
|---|---|---|---|
| Dysarthric | F03 | female PD/CP dysarthric | 661 |
| Dysarthric | F04 | female dysarthric | 250 |
| Dysarthric | M01 | male dysarthric (athetoid/dyskinetic) | 100 |
| Dysarthric | M02 | male dysarthric | 169 |
| Dysarthric | M03 | male dysarthric | 421 |
| Dysarthric | M04 | male dysarthric | 298 |
| Dysarthric | M05 | male dysarthric | 523 |
| Control | FC01 | female control | 255 |
| Control | FC02 | female control | 999 |
| Control | FC03 | female control | 795 |
| Control | MC01 | male control | 754 |
| Control | MC02 | male control | 696 |
| Control | MC03 | male control | 600 |
| Control | MC04 | male control | 656 |
| | | **TOTAL** | **7,177** |

→ **7 dysarthric + 7 control** speakers with real EMA. (F01 has clinician
ratings but **no** EMA — audio only.)

### 1.2 `.pos` load format (decoded + verified)

- **Binary:** little-endian **float32**, no header. Row-major frames.
- **Columns per frame:** **84 = 12 sensors × 7 values**. Per sensor, in order:
  `[x, y, z, phi, theta, RMS, extra]`
  - `x, y, z` — Cartesian position in **mm** (AG500 measurement volume).
  - `phi, theta` — two Euler angles of the coil axis (degrees), the AG500's 5 DOF.
  - `RMS` — residual of the position fit (measurement quality; lower = better).
    Verified: values ~0.08–40, always the 6th slot.
  - `extra` — always `0.0` in this release (unused 7th slot).
- **Sampling rate:** **200 Hz** (AG500 hardware rate; authoritative, Rudzicz 2012).
- **Frames:** `n_frames = filesize / 336`. Time of frame *i* = `i / 200` s.
- `pos/` = **head-corrected** positions (use these). `rawpos/` = raw
  (pre-head-correction) + a `posamps/` subdir of raw amplitudes. `amps/` =
  `NNNN.amp` (per-sensor field amplitudes) + `NNNN.ini` (12-channel calibration
  header; text, lists `channel1..channel12`).

Minimal reader:

```python
import numpy as np
def load_torgo_pos(path, n_sensors=12, n_cols=7, fs=200):
    a = np.fromfile(path, dtype='<f4')
    a = a.reshape(-1, n_sensors, n_cols)      # (frames, 12, 7)
    xyz   = a[..., 0:3]                        # mm
    angles= a[..., 3:5]                        # phi, theta (deg)
    rms   = a[..., 5]                          # fit residual (quality)
    t     = np.arange(a.shape[0]) / fs         # seconds
    return xyz, angles, rms, t
```

### 1.3 Sensor → articulator mapping (IMPORTANT caveat)

AG500 gives up to 12 coils. Per Rudzicz (2012) the articulator coils are placed
on **upper lip (UL), lower lip (LL), tongue tip (TT), tongue middle (TM), tongue
back (TB)**, plus mouth-corner and **reference** coils (nasion / upper incisor /
behind-ears) used for head correction. **The exact coil-index → articulator
assignment is not uniform across speakers/sessions**, and several coils are
flagged dead per speaker (e.g. `M01/Notes/M01.csv`: *"Coils 1, 2, 3, 6 never
on"*; `MC01/Notes`: *"S10 replaced by S4 due to bad position"*). **Before using a
given channel, resolve its articulator identity + validity from (a) the TORGO
release documentation and (b) that speaker's `Notes/<SPK>.csv` recording notes,
and drop channels with degenerate/zero trajectories.** Do not hard-code a global
mapping.

### 1.4 Alignment to audio

- **EMA-synchronous audio:** `wavall/NNNN.wav.all` — RIFF WAVE, **16 kHz,
  2-channel, 16-bit PCM** (verified from headers). Recorded on the AG500 rig,
  triggered with the sweep, so it shares the EMA's `t=0`.
- **Frame ↔ sample:** 16000 / 200 = **80 audio samples per EMA frame**; EMA
  frame *i* ↔ audio sample `round(i*80)`.
- **Caveat:** audio runs ~0.6–0.8 s **longer** than the EMA sweep (trailing
  capture); align at the **common start** and truncate/pad at the tail. Do **not**
  assume equal duration.
- The separate `wav_headMic/` and `wav_arrayMic/` are higher-quality mic
  captures of the *same* prompts but are **not** sample-locked to the EMA — for
  articulatory alignment use `wav.all`. Prompt text is in `prompts/NNNN.txt`
  (e.g. `0001` = "fear"); phone alignments (when present) in `phn_*/NNNN.phn`.

### 1.5 Clinician-rated severity (bonus — partially covers the "clinician-rated" need)

Each speaker has `.../<SPK>/Notes/<SPK>.csv`. For **dysarthric** speakers this is
a **Frenchay Dysarthria Assessment** by a clinician:

- Subsystems scored: **Reflex** (cough/swallow/drool), **Respiration**,
  **Lips**, **Jaw**, **Palate**, **Laryngeal**, **Tongue**, and
  **Intelligibility** (words / sentences / conversation), each on an ordinal
  **a–e** scale (`a` ≈ normal → `e` ≈ most severe; intermediates like `c/d`
  used), plus a free-text clinical **Summary** and influencing factors.
- Example `F01`: Tongue-Elevation `d`, Tongue-Alternate `d/e`, Intel-Sentences
  `d/e`, Laryngeal `d` — a severe profile; `F03`/`F04` are much milder (mostly
  `a`). **Control** speakers' Notes carry only demographics + per-session EMA
  recording notes (no Frenchay ratings — they are healthy).
- CSV files present (11): `F01, F03, F04, FC01–03, M01–05, MC01–03`.
- These are **per-speaker** severity (not per-utterance GRBAS). Published
  TORGO intelligibility/severity groupings also exist in Rudzicz (2012) if a
  categorical severity label is needed.

### 1.6 Use in VERITAS

- **H1 (faithfulness ↔ inversion error, healthy):** use the **7 control
  speakers** (FC01–03, MC01–04) — real healthy EMA, on disk now.
- **H3 (anti-normalization, pathological):** use the **7 dysarthric speakers**
  (F03/F04/M01–M05) vs the controls — real pathological EMA, on disk now.
- The Frenchay ratings give an independent clinician severity axis to correlate
  against faithfulness.

---

## 2. HPRC — Haskins Production Rate Comparison  ⛔ GATED (H1 primary, external)

The requested primary H1 healthy-EMA set. **Could not be downloaded: the
canonical public link is dead** and no live open mirror exists.

- **What it is:** Tiede et al. (2017), "Quantifying kinematic aspects of
  reduction in a contrasting rate production task." 8 speakers (4 F: F01–F04,
  4 M: M01–M04), 720 phonetically-balanced IEEE sentences × normal + fast rate
  → ~1,440 utts/speaker, ~7.2–7.9 h.
- **EMA format (from the literature):** NDI **WAVE**, 5-D EMA at **100 Hz**;
  audio **44.1 kHz**. Sensors: tongue **tip / body / root**, **upper & lower
  lip**, **jaw/mandible**, plus references (**left & right mastoid, upper &
  lower incisor**). Trajectories low-pass filtered (5 Hz refs / 20 Hz
  articulators). Distributed as per-utterance MATLAB `.mat` (audio + EMA
  together); community preprocessors expose per-speaker `wav/ + data/ +
  TextGrids/` (see `github.com/bootphon/articulatory_inversion`).
- **License:** research use (per Haskins terms).
- **Access — LINK IS DEAD (verified 2026-07-09):** the widely-cited Yale Box
  share `https://yale.app.box.com/s/cfn8hj2puveo65fq54rp1ml2mk7moj3h` returns
  **HTTP 404** (tested with browser UA on `/s/` and `/v/`). OSF `df5ub` (cited by
  one secondary source) is **private (401)**. No HuggingFace/Zenodo mirror; HF is
  unreachable from this host.
- **How to actually obtain it (manual):**
  1. Email **Haskins Laboratories** / **Mark Tiede** (`tiede@haskins.yale.edu`)
     requesting the current HPRC / "EMA-IEEE" download link; cite Tiede et al.
     (2017). Yale Box shares are periodically re-issued, so ask for a fresh URL.
  2. Alternatively request via the SPARC / Berkeley speech-articulatory groups
     that redistribute processed HPRC for inversion research.
  3. Once a live URL is in hand, download **resumably** into
     `data/veritas/hprc/` with the recipe in §6.
- **H1 without HPRC:** use **TORGO controls** (§1, on disk) and/or **MNGU0**
  (§5.2, openly downloadable after registration) as the healthy-EMA reference.

---

## 3. NeuroVoz — Castilian-Spanish Parkinsonian speech, GRBAS  ⛔ GATED

Requested clinician-rated (GRBAS) set. **Open-access publication, but the data
files are access-restricted behind a Data Usage Agreement — not directly
downloadable.**

- **What it is:** Mendes-Laureano et al. (2024), *Scientific Data*,
  <https://doi.org/10.1038/s41597-024-04186-z> (PMC11655668). 112 participants
  (54 Parkinson's ON-medication + 58 healthy controls); sustained vowels, DDK,
  16 listen-and-repeat utterances, spontaneous monologue; 2,977 audio files,
  **1.01 GB**.
- **Labels / format:**
  - Audio: **WAV, 44.1 kHz** (avg SNR 24.3 dB).
  - **GRBAS** by a speech therapist, in **CSV**. Sustained vowels: 8 columns =
    `filename` + 5 integer GRBAS scores (**G**rade, **R**oughness,
    **B**reathiness, **A**sthenia, **S**train) + overall GRBAS + observations.
    LR/monologue: 16 columns (GRBAS + extra perceptual: glottal attack, tone,
    phonation quality).
- **License:** **CC BY-NC-ND 4.0**.
- **Access — RESTRICTED (verified 2026-07-09):** Zenodo record
  <https://zenodo.org/records/13647600> (HTTP 200) and concept DOI
  `10.5281/zenodo.10777656`. API shows `access_right: restricted`, `files: []`,
  and an `access_request` endpoint — i.e. a manual request/DUA is required.
- **How to obtain it (manual):**
  1. Open the Zenodo record → click **"Request access."**
  2. Copy the **full Data Usage Agreement** text from the repository into the
     request-message box and fill in the requested personal/affiliation info.
  3. Submit; on approval Zenodo grants file access to your account.
  4. Download resumably into `data/veritas/neurovoz/` (recipe §6). Post-grant,
     files follow `https://zenodo.org/records/13647600/files/<name>?download=1`,
     and the Zenodo API then exposes per-file **MD5** for integrity.

---

## 4. Reference sets — document-only (do NOT download)

### 4.1 USC 75-speaker real-time MRI (Lim et al. 2021)
- **What:** 75 speakers, 2-D sagittal RT-MRI video + synchronized audio, + 3-D
  volumetric vocal-tract MRI + T2 anatomical. Lim et al. (2021), *Scientific
  Data* 8, arXiv:2102.07896.
- **Access (verified live):** figshare
  <https://doi.org/10.6084/m9.figshare.13725546> (→ figshare article 13725546,
  HTTP 302→live); project site <https://sail.usc.edu/span/75speakers/> (200);
  loader code <https://github.com/usc-mrel/usc_speech_mri>; MREL sharing
  <https://mrel.usc.edu/sharing/>. **License GPL v3**, open, no registration.
- **Caveat:** raw RT-MRI is very large (hundreds of GB). Modality is MRI, not
  EMA — relevant only as an auxiliary articulatory reference. Related 2023 set
  adds articulator ground-truth segmentations
  (<https://doi.org/10.1038/s41597-023-02766-z>).

### 4.2 MNGU0 EMA (Richmond, Hoole & King 2011) — good open H1 fallback
- **What:** single speaker, Carstens **AG500** EMA + audio, **>1,300**
  phonetically-diverse utterances (Day-1 subset). Announced Interspeech 2011.
- **Access (verified live):** <https://www.mngu0.org/> (HTTP 200). **Free for
  research** after creating an account and accepting the licence on the
  forum-style site; Day-1 EMA is provided as normalized + basic EMA + audio.
- **Why it matters here:** openly obtainable healthy AG500 EMA — the cleanest
  **H1 fallback** if HPRC access stalls. Registration is manual (account +
  licence click-through), so it is document-only for now; download recipe §6.

### 4.3 AUSpeech — audio-ultrasound tongue-movement DB (Mandarin, 2025)
- **What:** synchronized audio + ultrasound tongue imaging; **43 healthy + 11
  dysarthric** speakers (matches the "11 dysarthric" brief). Audio WAV
  16 kHz/16-bit mono; ultrasound `.dcm` (920×700, 60 fps); transcripts
  `.lab`/`.TextGrid`. *Scientific Data* 2025,
  <https://doi.org/10.1038/s41597-025-04917-w> (PMC11992241).
- **Access (verified live):** Science Data Bank
  <https://cstr.cn/31253.11.sciencedb.18722> → resolves to
  `scidb.cn/detail?dataSetId=7b6a498ee725439ba3a060985fece3cd` (HTTP 200).
- **Caveat:** **≈ 676 GB total**; ultrasound (not EMA). Reference only.

---

## 5. (removed — merged into §4)

---

## 6. Resumable download recipe (PowerShell, for gated sets once access is granted)

This machine's Python cannot reliably reach HF/CDNs; **PowerShell/.NET reaches
the general web**. Always resumable (laptop can lose power). HuggingFace is
unreachable here regardless (TLS failure) — do not route through HF.

```powershell
# --- Option A: curl.exe with resume (-C -). Best for single-file HTTP(S). ---
$ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36"
$dst = "C:\Users\Dell\Desktop\Dysar\data\veritas\neurovoz"   # or ...\hprc
New-Item -ItemType Directory -Force $dst | Out-Null
# -C -  resumes a partial file; re-run the same line after a power loss.
& curl.exe -L -C - --retry 5 --retry-delay 10 -A $ua `
    -o (Join-Path $dst "FILENAME.zip") "https://.../FILENAME.zip?download=1"

# --- Option B: BITS (survives reboots, auto-resumes). ---
Start-BitsTransfer -Source "https://.../FILENAME.zip" `
    -Destination (Join-Path $dst "FILENAME.zip") `
    -DisplayName "veritas-download" -Priority Foreground
#  If interrupted:  Get-BitsTransfer | Resume-BitsTransfer

# --- Integrity: verify against the checksum the repo publishes. ---
Get-FileHash (Join-Path $dst "FILENAME.zip") -Algorithm MD5   # or SHA256
```

Intended local layout (create on demand; nothing written there yet):
```
data/veritas/
  hprc/       # Haskins HPRC .mat / wav+ema  (after Haskins grants a link)
  neurovoz/   # NeuroVoz wav + GRBAS csv      (after Zenodo DUA approval)
```

---

## 7. What is actually usable *today*

- **Real EMA on disk, verified:** TORGO — 7,177 head-corrected `.pos`
  (float32, 12×7 cols, 200 Hz), 7 dysarthric + 7 control speakers, +
  EMA-synchronous 16 kHz audio. **Unblocks H1 (controls) and H3 (dysarthric).**
- **Real clinician ratings on disk:** TORGO Frenchay Dysarthria Assessments
  (`Notes/*.csv`, a–e per subsystem) for the dysarthric speakers.
- **Gated, documented for manual retrieval:** HPRC (dead link → email Haskins),
  NeuroVoz (Zenodo DUA request).
- **Reference-only:** USC 75-spk rtMRI (open, huge), MNGU0 (open, register —
  best external H1 fallback), AUSpeech (open, 676 GB, ultrasound).

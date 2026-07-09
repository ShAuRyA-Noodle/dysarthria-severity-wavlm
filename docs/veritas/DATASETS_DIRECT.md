# DIRECT-DOWNLOAD Datasets — Dysarthria / Pathological Speech + Articulatory Inversion

Compiled 2026-07-10. Goal: find **every publicly DIRECT-DOWNLOADABLE** dataset that
strengthens (A) dysarthria/PD/ALS/pathological **speech with severity or
intelligibility labels** and (B) **measured articulation** (EMA / rtMRI /
ultrasound / X-ray microbeam) to ground acoustic-to-articulatory inversion (SPARC).

**Verification method.** Every URL below was probed live (Zenodo/figshare/HF/Synapse
APIs, PhysioNet page, Apache directory indexes, HTTP HEAD). Classifications are
honest and conservative — a dead or gated link wastes time.

**Access tiers used here (the user's rule):**
- **DIRECT** — a real HTTP file you can `curl`/`wget`/browser-download now. No account, no form, no approval.
- **BORDERLINE** — one *instant, self-serve* step (free account + click-accept, anonymous rsync, or a browser Box/Mendeley "Download" button). No human approves you. The exact step is stated per row.
- **GATED** — a human must approve (email a professor, sign a DUA, Zenodo "request access", ACT review). **Excluded** from the get-these tables; listed separately.

> Reachability note: this project's machine has had **TLS failures reaching HuggingFace** in the past (see `docs/veritas/DATASETS.md`). HF rows are direct *in principle*; if HF is unreachable locally, prefer the non-HF origin (e.g. EasyCall official zip, TORGO Toronto, IPVS via IEEE account).

---

## 1. DIRECT-DOWNLOAD — GET THESE

### 1A. Speech (break the dysarthria-severity ceiling + cross-corpus)

Ranked by value to a dysarthria-severity + cross-corpus project. "DIR" = direct, "BOR" = borderline (self-serve step in the Access column).

| # | Dataset | Access | Verified URL | Size | Format | Labels (severity/intelligibility/pathology) | License |
|---|---|---|---|---|---|---|---|
| 1 | **VOC-ALS** (102 ALS + 51 HC) | **BOR** — free Synapse account (email+ToS; API `accessRequirement`=0 → no DUA/committee) | `https://www.synapse.org/Synapse:syn53009474` (DOI 10.7303/syn53009474) | ~ few GB | WAV 8 kHz | **Ordinal dysarthria severity** from ALSFRS-R speech subscore: severe 5 / moderate 13 / mild 31 / non-dysarthric 53; vowels + /pa-ta-ka/; clinical Excel | CC BY 4.0 |
| 2 | **EasyCall** (31 dysarthric + 24 HC, Italian) | **DIR** (official) or **DIR** (HF) | official `http://neurolab.unife.it/easycallcorpus/EasyCall.zip` **(HTTP only — port 443 refused)**; HF `https://huggingface.co/datasets/changelinglab/easycall-dysarthria` | 629 MB / 1.19 GB | zip WAV / parquet | **Per-speaker dysarthria severity** (Therapy Outcome Measure, neurologist); 21,386 utts (commands + non-commands) | CC BY-NC 2.0 |
| 3 | **IPVS — Italian Parkinson's Voice & Speech** (28 PD + 37 HC) | **DIR** (HF mirror, `gated:false`) or **BOR** (IEEE DataPort free account) | HF `https://huggingface.co/datasets/birgermoell/Italian_Parkinsons_Voice_and_Speech`; origin `https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech` | 565 MB | WAV + XLSX / audiofolder | PD vs HC; **PD binned by severity** (1-5/6-10/11-16/17-28); balanced text, vowels, /pa-ta-ka/ | CC BY 4.0 |
| 4 | **`resproj007/pathological_speech`** (merged multi-corpus) | **DIR** (HF, `gated:false`) — ⚠ license unstated, likely contains re-uploaded TORGO/UASpeech | `https://huggingface.co/datasets/resproj007/pathological_speech` | 5.32 GB / 46,517 rows | parquet, 16 kHz | **condition + diagnosis + severity + intelligibility** columns (TORGO+UASpeech+Libri merged) | not stated ⚠ |
| 5 | **MDVR-KCL** (16 PD + 21 HC, mobile) | **DIR** (Zenodo open) | `https://zenodo.org/records/2867216` → `https://zenodo.org/api/records/2867216/files/26_29_09_2017_KCL.zip/content` | 606 MB | WAV | Filename-encoded **Hoehn&Yahr + UPDRS-II.5 + UPDRS-III.18**; reading passage + spontaneous | CC BY 4.0 |
| 6 | **TORGO** (canonical dysarthric, incl. EMA) | **DIR** (Toronto, cite-only — *not* LDC-gated) | `http://www.cs.toronto.edu/~complingweb/data/TORGO/{F,FC,M,MC}.tar.bz2` | ~18 GB | tar.bz2: WAV + AG500 EMA | Dysarthric (CP/ALS) vs control per speaker; +Frenchay ratings in release | academic use, cite Rudzicz 2012 |
| 7 | **UASpeech** (intelligibility bands) | **DIR** via HF mirrors — ⚠ unauthorized re-uploads (official is gated) | e.g. `https://huggingface.co/datasets/ngdiana/uaspeech_severity`, `AravindVadlapudi02/UA_speech_{very-low,low,mid,high}` | ~8-40 MB ea | parquet | **Intelligibility bands** (very-low→high) | none/MIT (re-upload) ⚠ |
| 8 | **SVD — Saarbrücken Voice Database** (~2000 German) | **DIR** (Zenodo open) | `https://zenodo.org/records/16874898` → `.../files/data.zip/content` (+ per-pathology zips) | ~30-47 GB (70 files) | NSP→WAV + EGG | **Medical diagnosis** (~70 classes: recurrent-nerve paralysis, laryngitis, dysphonias, some neurogenic); vowels + sentence. **Voice disorder, not dysarthria; no severity scale** | CC BY 4.0 |
| 9 | **PVQD — Perceptual Voice Qualities DB** (296) | **DIR** (Mendeley "Download All" — no login) | `https://data.mendeley.com/datasets/9dz247gnyb/4` | ~ hundreds MB | WAV 44.1 kHz + XLSX | **CAPE-V + GRBAS perceptual severity** (19 raters); sustained vowels + CAPE-V sentences. **Dysphonia (voice disorder)** | CC BY 4.0 |
| 10 | **VOICED** (208: 150 path / 58 HC) | **DIR** (PhysioNet open) | `https://physionet.org/content/voiced/1.0.0/` | 110 MB | WFDB + txt | Dysphonia type + **VHI + RSI + demographics**; 5 s vowel /a/. **Voice disorder, not dysarthria** | ODC-BY 1.0 |
| 11 | **UCI Parkinson speech** (Sakar 301/2018, Little 174) | **DIR** (open zip) | `https://archive.ics.uci.edu/static/public/174/parkinsons.zip` (+ 301) | <1 MB | CSV | **UPDRS + PD label** — ⚠ **extracted acoustic features, NO raw audio** | CC BY 4.0 |
| 12 | **Coswara** (adjacency / OOD only) | **DIR** (Zenodo/GitHub open) | `https://zenodo.org/records/7188627`; `https://github.com/iiscleap/Coswara-Data` | ~13 GB | WAV | COVID status only — **NOT dysarthria, no severity/intelligibility**; large healthy OOD corpus | CC BY 4.0 |

**One-line downloads (1A):**
```bash
# 1 VOC-ALS  (create free Synapse account first, then:)
pip install synapseclient && synapse login && synapse get -r syn53009474
# 2 EasyCall (NOTE http:// only)
curl -O http://neurolab.unife.it/easycallcorpus/EasyCall.zip      # or: load_dataset("changelinglab/easycall-dysarthria")
# 3 IPVS
huggingface-cli download birgermoell/Italian_Parkinsons_Voice_and_Speech --repo-type dataset --local-dir ipvs
# 4 merged multi-corpus
python -c "from datasets import load_dataset; load_dataset('resproj007/pathological_speech')"
# 5 MDVR-KCL
wget -O MDVR-KCL.zip "https://zenodo.org/api/records/2867216/files/26_29_09_2017_KCL.zip/content"
# 6 TORGO
wget http://www.cs.toronto.edu/~complingweb/data/TORGO/{F,FC,M,MC}.tar.bz2
# 7 UASpeech (re-upload)  ->  load_dataset("ngdiana/uaspeech_severity")
# 8 SVD
wget -O svd_data.zip "https://zenodo.org/api/records/16874898/files/data.zip/content"
# 9 PVQD  ->  browser "Download All" at data.mendeley.com/datasets/9dz247gnyb/4 (no clean wget)
# 10 VOICED
wget -r -N -c -np https://physionet.org/files/voiced/1.0.0/
# 11 UCI
curl -O https://archive.ics.uci.edu/static/public/174/parkinsons.zip
# 12 Coswara
wget -O coswara.zip "https://zenodo.org/api/records/7188627/files/iiscleap/Coswara-Data-dataset-paper-publication.zip/content"
```

**What each buys (1A):** **VOC-ALS** = the cleanest *ordinal* dysarthria-severity target and an ALS arm (best single ceiling-breaker). **EasyCall** = Italian dysarthric + clinician TOM severity, non-gated. **IPVS** = cleanest-license PD severity. **`pathological_speech`** = ready-made merged severity/intelligibility benchmark (fast, but license-risky). **MDVR-KCL** = real-clinic PD connected speech (H&Y/UPDRS). **TORGO** = license-clean route to the canonical dysarthric set + its EMA. **UASpeech mirrors** = intelligibility bands (stopgap). **SVD/PVQD/VOICED** = large open *voice-disorder* corpora → healthy/pathological manifold + specificity controls (not dysarthria targets). **UCI** = tabular UPDRS only. **Coswara** = big healthy OOD/robustness corpus.

### 1B. Articulatory (ground/validate SPARC inversion)

Ranked by value to validating acoustic-to-articulatory inversion. All **healthy** unless noted.

| # | Dataset | Access | Verified URL | Size | Format / articulators | License |
|---|---|---|---|---|---|---|
| 1 | **USC-TIMIT** (10 spk) ⭐ | **DIR** (Zenodo open, new May-2026) | `https://zenodo.org/records/19422914` → `.../files/USC-TIMIT.zip/content` | 8.5 GB | **rtMRI (10 spk) + EMA (4 spk)** same speakers/460 sentences + audio | CC BY 4.0 |
| 2 | **MOCHA-TIMIT** (~10 spk) | **DIR** (open Apache index) | `https://data.cstr.ed.ac.uk/mocha/` (fsew0/msak0/maps0 + 7 more in `/unchecked/`) | ~1.6 GB | **EMA** (.ema trackfiles) + EPG + laryngograph + 16 kHz WAV; lips/jaw/tongue tip-body-dorsum/velum | QMUC academic, non-commercial |
| 3 | **XRMB — Wisconsin X-ray Microbeam** (~50 spk) | **BOR** — public Box folder, no login/DUA (browser download; not clean curl) | data `https://berkeley.app.box.com/v/xray-microbeam-database-data`; labels `https://github.com/rsprouse/xray_microbeam_database` | few GB | **Gold-pellet flesh-point trajectories** (4 tongue, UL, LL, jaw) + WAV; alignments in repo | BSD-3 (repo); data cite Westbury 1994 |
| 4 | **USC 75-speaker rtMRI** (75 spk) | **DIR** (figshare open) | `https://api.figshare.com/v2/articles/13725546`; 2D-video subset `https://ndownloader.figshare.com/files/26416990` | 2.55 GB subset · **570 GB** full | rtMRI video + audio + 3D + T2 MRI; whole midsagittal tract | CC BY 4.0 |
| 5 | **UltraSuite** (UXTD/UXSSD/UPX/Cleft/UX2020) | **BOR** — anonymous rsync (no login) | `https://ultrasuite.github.io/download/` | 82-191 GB each | **Ultrasound tongue** + audio + labels; **UXSSD/UPX/Cleft = DISORDERED child speech** (SSD, cleft) | CC BY-NC 4.0 |
| 6 | **TaL corpus** (82 spk) | **BOR** — anonymous rsync | `https://ultrasuite.github.io/data/tal_corpus/` | 2-498 GB | **Ultrasound tongue + lip video** + audio | CC BY-NC 4.0 |
| 7 | **rtMRI segmentation masks** (Zenodo 10046815) | **DIR** (Zenodo open) | `https://zenodo.org/records/10046815` → `.../files/data.zip/content` | 22 MB | DICOM + `.mat` **articulator masks** (tongue, soft palate, tract) — validation labels | CC BY 4.0 |

**One-line downloads (1B):**
```bash
# 1 USC-TIMIT (rtMRI+EMA, gold)
curl -L -o USC-TIMIT.zip "https://zenodo.org/api/records/19422914/files/USC-TIMIT.zip/content"
# 2 MOCHA-TIMIT (everything, incl. /unchecked)
wget -r -np -nH --cut-dirs=1 -R "index.html*" https://data.cstr.ed.ac.uk/mocha/
# 3 XRMB: browser-download the Box folder; then  git clone https://github.com/rsprouse/xray_microbeam_database
# 4 USC 75-spk (start with 2.5 GB video subset)
curl -L -o usc75_2drt.zip https://ndownloader.figshare.com/files/26416990
# 5 UltraSuite (needs an rsync client on Windows)
rsync -av ultrasuite-rsync.inf.ed.ac.uk::ultrasuite/core-uxtd .    # or core-uxssd/core-upx/core-cleft/core-ux2020
# 6 TaL
rsync -av ultrasuite-rsync.inf.ed.ac.uk::tal-corpus/TaL80 .
# 7 rtMRI masks
curl -L -o rtmri_masks.zip "https://zenodo.org/api/records/10046815/files/data.zip/content"
```

**What each buys (1B):** **USC-TIMIT** ⭐ = the *gold* case — same speakers in **both** rtMRI and EMA, so SPARC inversion can be cross-checked against two measurement modalities. **MOCHA-TIMIT** = the classic, tiny, no-friction EMA↔audio benchmark (fsew0/msak0 are the standard inversion eval speakers). **XRMB** = ~50-speaker flesh-point articulation for speaker-generalization. **USC 75-spk** = largest open vocal-tract-imaging set (75 spk). **UltraSuite** = the only *direct* articulatory data that includes **disordered speech** (pediatric SSD/cleft, ultrasound). **TaL** = multi-speaker tongue+lip imaging. **Zenodo 10046815** = ready articulator-contour ground truth for validating segmentation/inversion.

---

## 2. GATED — SKIP (needs a human to approve)

Excluded from §1 because each requires email approval, a signed DUA, or a Zenodo "request access". Listed so no time is wasted probing them.

| Dataset | Gate (exact) | What it would buy |
|---|---|---|
| **SAP — Speech Accessibility Project** (UIUC) | Sign UIUC **Data Use Agreement** + application review (`speechaccessibilityproject.beckman.illinois.edu`) | The **largest** dysarthria/disordered-speech corpus (thousands of speakers, severity). The real prize — but DUA-gated. |
| **PC-GITA** (Spanish PD, 50 PD/50 HC) | Email/contact-form **Orozco-Arroyave** (`gita.udea.edu.co/contact`) — "available on reasonable request" | Cross-language PD with clinical severity. |
| **NeuroVoz** (Spanish PD GRBAS, 54 PD/58 HC) | Zenodo **`access_right: restricted`**, "Request access" (records 13647600 & 10777657) | Parkinsonian speech with **GRBAS** perceptual severity. |
| **EWA-DB** (Slovak: 175 PD, 87 AD, 62 MCI, 1323 HC) | Zenodo files **restricted** → fill form, **sign & return EULA**, manual approval (record 10952480); ELDA also gated | Largest non-English PD/AD/MCI cross-lingual arm. |
| **UASpeech** (official) | Email **Mark Hasegawa-Johnson** (`jhasegaw@illinois.edu`) for sftp license. (IEEE DataPort copy = paywalled empty shell.) | Canonical dysarthric intelligibility set. *Note: de-facto obtainable via HF re-uploads (§1A #7) at licensing risk.* |
| **mPower** Parkinson (Synapse syn4993293) | Synapse Certified quiz (self-serve) **+ Validated profile + Intended-Data-Use statement → ACT human review** | Very large but noisy real-world PD voice; weak labels. |
| **Nemours** (11 male dysarthric) | Historic CD-ROM via Nemours/duPont Institute; no web endpoint | Small classic dysarthric set — effectively unavailable. |
| **AVFAD** (Portuguese, 346 path/363 HC) | Email `ieeta-acsa@ua.pt` (name+postal address) → OneDrive access granted | Portuguese voice-pathology (CAPE-V + clinical). |
| **Haskins HPRC** (8 spk EMA, 2 rates) | Canonical Yale Box link **404/dead** → email **Mark Tiede / Haskins** | Parallel acoustic–EMA at two speech rates. |
| **EMA-MAE** (40 spk, 3D EMA) | Email **Dr. Mike Johnson** (`mike.johnson@marquette.edu`) — no download button (`speechlab.eece.mu.edu/emamae`) | L1-vs-L2 3D EMA for inversion/accent. |
| **mngu0** (1 spk, EMA + MRI) | Email **Korin Richmond** (CSTR) → **manually-issued UserID** (`homepages.inf.ed.ac.uk/korin/...`) | Classic single-speaker EMA+MRI benchmark. |
| **FROST-EMA** (Finnish/Russian, 1250 Hz EMA) | **Not yet released** (planned 2026; no repo URL as of Jul 2026; arXiv 2506.08981) | Cross-lingual high-rate EMA — watch for release. |

Also confirmed **empty**: **OpenSLR** hosts *no* pathological/dysarthric/PD/ALS/clinical or articulatory corpus (whole index checked). **Kaggle** TORGO/dysarthria/Sakar sets are BORDERLINE (public pages, but download needs a free Kaggle account + API token) and are binary-labeled re-uploads — HF is easier.

---

## 3. HONEST VERDICT

### (a) Can direct-download data break the n=15 (TORGO) dysarthria-SEVERITY ceiling? — **YES, roughly 10×, but with caveats.**
Severity/intelligibility-labeled pathological speakers obtainable **without a human gate**:

| Source | New labeled speakers | Severity label | Access |
|---|---|---|---|
| VOC-ALS | **102 ALS** | ALSFRS-R ordinal (severe→none) | borderline (free Synapse acct) |
| EasyCall | **31 dysarthric** | TOM (clinician) | direct |
| IPVS | **28 PD** | 4 severity bins | direct |
| MDVR-KCL | **16 PD** | H&Y / UPDRS | direct |
| UASpeech (mirror/on-disk) | ~**15 dysarthric** | intelligibility bands | direct (re-upload risk) |

That takes labeled *motor-speech* speakers from ~8 dysarthric (TORGO) to **≈150-190 speakers across 4 etiologies (ALS, cerebral palsy, Parkinson's, mixed Italian dysarthria) and 3 languages (English, Italian, PD-Italian)** — a genuine order-of-magnitude break and, more importantly, real **cross-corpus / cross-language / cross-etiology** validation.
**Blunt caveats:** (1) labels are *heterogeneous* — ALSFRS-R vs TOM vs UPDRS vs intelligibility-% — so they need harmonizing into a common ordinal/z-scored axis, not naive pooling. (2) VOC-ALS needs a free Synapse account (instant, no human). (3) The single biggest set, **SAP**, is DUA-gated — the ceiling break via *direct* download tops out at the ~190-speaker aggregate above; SAP would add thousands but requires the DUA. (4) SVD/VOICED/PVQD add ~2500 more *pathological-voice* speakers but those are laryngeal **voice disorders, not dysarthria** — use them as manifold/specificity controls, not severity targets.

### (b) Can direct-download data add DYSARTHRIC articulatory ground truth? — **Essentially NO. Confirmed gap.**
There is **no new adult dysarthric EMA available by direct download.** The only measured articulation on pathological speakers that is public is:
- **TORGO EMA** — AG500 EMA on ~7-8 dysarthric speakers. Directly downloadable (Toronto tarballs) **and already on disk** — so it adds *nothing new* (same speakers).
- **UltraSuite (UXSSD/UPX/Cleft)** — direct via rsync, but it is **pediatric speech-sound-disorder / cleft ultrasound**, *not* adult neuromotor dysarthria, and it is **ultrasound, not EMA** (harder to compare to SPARC's EMA-space inversion).
- **AUSpeech** (43 healthy + 11 dysarthric, Mandarin) — ScienceDB, but **~676 GB and ultrasound** → impractical and wrong modality.

Bottom line: **direct-download does not add clean new dysarthric EMA.** Measured dysarthric articulation beyond TORGO does not exist as an open EMA download; the nearest gated option (Haskins/EMA-MAE) is healthy anyway. Grounding the *anti-normalization / pathological* inversion behaviour must lean on TORGO's on-disk dysarthric EMA.

### (c) Can direct-download data add HEALTHY articulatory data to ground inversion? — **YES, abundantly. Fully achievable now.**
- **USC-TIMIT** (Zenodo, 8.5 GB) ⭐ — rtMRI **and** EMA on the same 10 speakers/460 sentences: validate SPARC inversion against two independent modalities. **Grab first.**
- **MOCHA-TIMIT** (CSTR, ~1.6 GB) — the standard EMA↔audio inversion benchmark; smallest, fastest, zero friction.
- **XRMB** (Berkeley Box) — ~50-speaker flesh-point trajectories for speaker-generalization (borderline: browser Box download).
- **USC 75-speaker rtMRI** (figshare) — 75-speaker whole-tract imaging (start with the 2.5 GB video subset; full set is 570 GB).
- **UltraSuite / TaL** (rsync) — large healthy tongue(/lip) ultrasound.

**Immediate plan:** MOCHA-TIMIT + USC-TIMIT cover healthy EMA-inversion validation today with ~10 GB and no gate. XRMB/USC-75 extend to many speakers when needed.

---

### One-paragraph bottom line
Grab, in order: **USC-TIMIT + MOCHA-TIMIT** (healthy EMA/rtMRI, direct, ~10 GB — grounds SPARC now); **VOC-ALS** (Synapse acct) + **EasyCall** + **IPVS** + **MDVR-KCL** (direct, ~2 GB — take dysarthria/PD *severity* from n≈8 to n≈190 across 4 etiologies/3 languages). Treat SVD/VOICED/PVQD as voice-disorder controls. Accept two hard limits: **no new dysarthric articulatory (EMA) data exists by direct download** (TORGO on-disk is the ceiling), and **the largest dysarthria-severity corpus (SAP) is DUA-gated**, so the direct-download severity break caps around ~190 speakers.

# Datasets — direct links + baby-step downloads

Three external corpora to acquire for the world-class roadmap. Tier 1 = open
direct download. Tier 2 = email request, expect 1–3 weeks lead time.

All datasets land under `data/raw/<corpus>/`. Embeddings under
`data/embeddings/<corpus>/`. Both directories are gitignored — never commit
audio or embeddings to the repo.

---

## 1. VCTK (open, ~10 GB) — neurotypical English baseline

**Why:** clean multi-speaker English controls. Used by Track C as a
neurotypical-baseline source for perturbation references and by Track D as
the cross-corpus English manifold check.

### Baby steps
1. Open browser → https://datashare.ed.ac.uk/handle/10283/3443
2. Click **"Download all files"** → `VCTK-Corpus-0.92.zip` (~10.4 GB).
3. While it downloads, create the destination:
   ```bash
   mkdir -p "c:/Users/Dell/Desktop/Projects/Dysar/data/raw/vctk"
   ```
4. Move the zip there, then unzip:
   ```bash
   cd "c:/Users/Dell/Desktop/Projects/Dysar/data/raw/vctk"
   unzip VCTK-Corpus-0.92.zip
   ```
5. Verify expected layout:
   ```
   data/raw/vctk/
       wav48_silence_trimmed/        ← 110 speaker subdirs (p225, p226, …)
       txt/                          ← per-utterance transcripts
       speaker-info.txt              ← speaker metadata
   ```
6. Smoke check:
   ```bash
   ls data/raw/vctk/wav48_silence_trimmed | head
   ```
   Should list ≥110 speaker IDs (`p225, p226, …, p376`). If empty, the
   archive layout changed — re-check the DataShare page.

### License
ODC-BY (text) + CC0 (audio). Cite Veaux et al. 2017.

### Direct mirror (optional, faster in some regions)
http://www.udialogue.org/download/VCTK-Corpus.tar.gz (older 0.80 — prefer 0.92).

---

## 2. EasyCall corpus (Italian dysarthric, ~21k recordings, 24 healthy + 31 dysarthric)

**Why:** independent dysarthric corpus, different language, different
acoustic conditions. Strongest cross-corpus claim for Track D.

**Status:** searched Zenodo + ResearchGate + ISCA archive — **no public
direct download URL surfaced**. Paper is open access on arXiv but the
data-availability statement is in the PDF only. Likely gated via author
contact.

### Baby steps
1. Open https://arxiv.org/abs/2104.02542 → click **"Download PDF"**.
2. Search the PDF (Ctrl+F) for "available", "data", "dataset", "download",
   "request", "zenodo", "github". The data-availability statement should
   surface a URL or contact email.
3. If a Zenodo/GitHub URL exists, paste it back to me and I'll write the
   baby-step download for that specific URL.
4. If only an email is listed, send the same template as PC-GITA below
   (substitute corpus name + cite Turrisi et al. 2021 Interspeech).
5. Authors' affiliations: Università di Ferrara + Istituto Italiano di
   Tecnologia. PI candidates: **Leonardo Badino** (IIT) and **Luciano
   Fadiga** (UniFE). Email pattern usually `firstname.lastname@iit.it` or
   `@unife.it`.
6. Once data lands, place under `data/raw/easycall/`.

### Citation
Turrisi, R., Braccia, A., Emanuele, M., Giulietti, S., Pugliatti, M.,
Sensi, M., Fadiga, L., Badino, L. (2021). *EasyCall Corpus: A Dysarthric
Speech Dataset.* Proc. Interspeech 2021. DOI:
10.21437/Interspeech.2021-549.

---

## 3. PC-GITA (gated, Spanish Parkinson's speech, 50 PD + 50 controls)

**Why:** the cross-disorder anchor. Reframes ATD as a motor-speech-disorder
metric, not dysarthria-only. Required for Track D Hypothesis 1.

**Status:** gated. Email request to PI. Expect 1–3 weeks for response.
**Send NOW** so it arrives in parallel with our local cleanup.

### Baby steps
1. Find the PI's current email. Two routes:
   - https://scholar.google.com/citations?user=MqUjowUAAAAJ&hl=es →
     "Verified email at udea.edu.co" (his Universidad de Antioquia faculty
     email). Most recent paper PDF will list it explicitly.
   - Search "Juan Rafael Orozco-Arroyave Universidad de Antioquia email"
     and pick the most recent listing.
2. Compose email. Subject: `Research access request — PC-GITA database`.
3. Body (copy-paste, replace `[YOUR_NAME]` and `[INSTITUTION_OR_INDEPENDENT]`):

```
Dear Prof. Orozco-Arroyave,

I am [YOUR_NAME], a researcher at [INSTITUTION_OR_INDEPENDENT] working on a
zero-shot dysarthric speech severity metric (Articulatory Trajectory
Deviation, ATD) built on frozen WavLM-Large representations. We have
validated the metric on TORGO and UASpeech corpora and are preparing a
manuscript targeting INTERSPEECH 2027.

To extend the work to motor speech disorders beyond English dysarthria,
I am requesting research access to the PC-GITA database. Specifically, I
would use the sustained-vowel and DDK subsets to test whether the
articulatory manifold learned by self-supervised models generalises to
Parkinsonian dysarthria.

I commit to:
  (1) using the data exclusively for non-commercial academic research;
  (2) not redistributing the audio or any speaker-identifying derivatives;
  (3) citing Orozco-Arroyave et al. 2014 (PC-GITA paper) and any
      requested additional references in resulting publications;
  (4) signing any data-use agreement that GITA Lab requires.

Please let me know the formal request process and any documents I should
sign. Happy to share a draft of the ATD manuscript once it is submitted.

Thank you very much for your time.

Best regards,
[YOUR_NAME]
[Affiliation, Email, ORCID if available]
```

4. Send. Paste the reply (or any portal link / DUA they send) back to me
   when it arrives.
5. On receipt, place archive under `data/raw/pc_gita/`. Expected structure:
   ```
   data/raw/pc_gita/
       50_PD/                  ← Parkinson speakers
       50_HC/                  ← healthy controls
       PD_subjects.xlsx        ← UPDRS-III speech subscale per speaker
   ```

### Citation
Orozco-Arroyave, J. R., Arias-Londoño, J. D., Vargas-Bonilla, J. F.,
González-Rátiva, M. C., Nöth, E. (2014). *New Spanish speech corpus
database for the analysis of people suffering from Parkinson's disease.*
Proc. LREC 2014.

---

## Optional Tier 0 — already on disk

| Corpus | Location | Status |
|---|---|---|
| TORGO | `F/`, `FC/`, `M/`, `MC/` at repo root (gitignored) | ✓ ready |
| UASpeech | external (parsed via `src/uaspeech_parser.py`) | ✓ ready, embeddings layer 1+9 done |

---

## Why these three

| Track | Needs | From |
|---|---|---|
| A — streaming | none new | TORGO controls only |
| B — phoneme localization | none new | TORGO + MFA |
| C — perturbation | clean speech (TORGO already enough) | optional VCTK |
| D — cross-disorder | independent dysarthria + Parkinson's | **EasyCall + PC-GITA (critical)** |
| E — clinical anchoring | UASpeech 3-listener intelligibility | UASpeech (already on disk) |

Without PC-GITA, Track D drops to within-corpus-only (ATD on Italian
dysarthria via EasyCall) — still a paper-worthy contribution but the
"motor-speech-disorders" framing weakens. With PC-GITA, the headline
becomes "ATD generalises across languages AND disorders".

## Storage budget

- TORGO + UASpeech embeddings already use 66 GB of `data/embeddings`.
- VCTK + EasyCall + PC-GITA WavLM-Large + XLS-R-300m embeddings (layers
  1, 9, 12 only): estimated **+45 GB**.
- Plan disk free: ≥120 GB before kicking off cross-corpus extraction.

`shutil.disk_usage(PROJECT_ROOT)` (in
`experiments/01_verify_environment.py`) will warn at <50 GB free.

# VERITAS pitfalls paper — Submission Checklist

*Negative-results / methodology paper. Every number traces to a committed
`results/*.json`; this checklist tracks what is done and what remains before a
venue submission.*

---

## 1. What this paper is (be honest about it)

This is a **negative-results + pitfalls / methodology** paper, **not** a new-SOTA
claim. It shows that four popular frozen-SSL clinical-speech metric families
each fail a rigorous validity check (speaker leakage, silence/energy confounds,
cross-corpus collapse, absent pathological ground truth). The contribution is
the diagnostics + the LOCO reference protocol + a reproducible harness. Frame it
that way in the cover letter — reviewers should not expect a leaderboard win.

## 2. Target venues (ranked, honest fit)

| Venue | Fit | Format | Notes |
|---|---|---|---|
| **Interspeech** (main) — "Pathological speech & language" / "Analysis" area | Strong | 4 pp text + 1 pp refs, 2-col, double-blind | Analysis/negative-results papers are in scope; the real-EMA angle is distinctive. |
| **"Insights from Negative Results in NLP"** workshop (co-located ACL/EMNLP) | Very strong | short (4–8 pp) | Purpose-built for rigorous negative results; least reviewer-mismatch risk. |
| **ML4H** (Machine Learning for Health) — Findings track | Strong | short/Findings | Clinical audience; values validity/confound critiques. |
| **SLPAT** (Speech & Language Proc. for Assistive Tech.) workshop | Good | short | Clinical-speech community; smaller venue, faster turnaround. |
| **IEEE SLT / ICASSP** | Moderate | 4 pp incl. refs | Possible but the format is tight and negative results are a harder sell here. |

**Recommendation:** primary = Interspeech (main track, pathological-speech area);
fallback = *Insights* negative-results workshop. Both are 2-column and
double-blind, which the current template already matches.

## 3. Page-limit note

- Interspeech: **4 pages** content (incl. figures/tables) **+ 1 page references
  only**. ICASSP: 4 pages incl. refs (+ optional 5th refs-only page).
- Current draft (`veritas_pitfalls.tex`) has abstract + 10 sections +
  reproducibility + limitations + **2 tables + 3 figures**. In 10pt 2-column
  this will run **~4.5–5 pages** — **slightly over** the Interspeech 4-page text
  limit. Trim options (in priority order) if needed:
  1. Merge Findings 4 + 5 into one "inversion is unreliable" section.
  2. Cut Table 1 (Fig. 1 already carries it) **or** cut Fig. 1 (Table 1 carries
     it) — keep one.
  3. Shorten the Introduction's second paragraph and the Mirror section (§8).
- The reference list (10 entries) fits comfortably in the extra refs page.

## 4. Anonymization steps (double-blind)

- [x] Author block is `Anonymous Submission / Paper ID XXXX` (no names/affils).
- [x] Paper body never uses the internal codename "VERITAS" or personal handles.
- [ ] **Released-artifact URLs**: §8 (Gradio demo) and Reproducibility (harness)
      must use an **anonymized** repo/demo link for review
      (e.g. anonymous.4open.science) — do **not** paste the real GitHub URL.
- [ ] Strip any identifying strings from figure metadata before upload
      (the PDFs are matplotlib output; re-export is clean, but double-check).
- [ ] No Acknowledgements/funding lines in the anonymized version (none present
      now — keep it that way until camera-ready).
- [ ] Internal file paths in footnotes (`results/...`, `experiments/...`) are
      non-identifying and may stay, but confirm none embed a username.

## 5. Figure & caption pass

- [x] 3 figures generated **from committed JSON** by
      `figures/make_figures.py` (re-runnable; prints every plotted value).
- [x] Vector **PDF** (for LaTeX) + PNG (preview) for each.
- [x] Colorblind-safe **Okabe–Ito** palette; value labels on bars; each has a
      self-contained caption.
- [x] Fig. 1 within-vs-LOCO; Fig. 2 layer×reference leakage (with bootstrap
      CIs); Fig. 3 cycle-consistency energy-artifact.
- [ ] At camera-ready: confirm fonts embed in the final PDF (venue PDF-check)
      and that column-width rendering stays legible at print size.

## 6. Citations / bib

- [x] `refs.bib` — 10 entries, **all cited**, no uncited padding (verified).
- [x] All `\cite` keys resolve (verified programmatically).
- [ ] **Complete the two 2026 frozen-SSL preprint entries** (`frozenssl2026a` =
      arXiv:2604.10123, `frozenssl2026b` = arXiv:2604.21706): fill exact
      author list + final title from the arXiv abstract pages. They currently
      carry `[author list to be completed]` placeholders and descriptive titles
      (flagged in `refs.bib`). Everything else (WavLM, HuBERT, SPARC, TORGO,
      EasyCall, IPVS, WavRx, MFA) has full metadata.
- [ ] Swap `\bibliographystyle{IEEEtran}` if the venue mandates a different
      style (Interspeech uses IEEEtran; some workshops use ACL/`acl_natbib`).

## 7. LaTeX / template

- [x] Self-contained 2-column `article` preamble that compiles on stock TeX
      Live (no Interspeech `.sty` was in the repo).
- [x] Well-formedness verified **programmatically** (no local LaTeX toolchain):
      balanced braces, 20 balanced `\begin`/`\end`, skeleton present, all
      figure paths exist. Re-run:
      `python paper/veritas/figures/make_figures.py` then the tex checker.
- [ ] **Before submission**: drop the venue class into `paper/veritas/` and
      swap only the preamble (comment at the top of the `.tex` explains it) —
      body/tables/figures/refs port unchanged. Then do one real `pdflatex →
      bibtex → pdflatex ×2` pass on a machine with LaTeX and eyeball the PDF.

## 8. Numbers integrity

- [x] Every headline number re-verified against its JSON (not trusted from the
      markdown draft).
- [x] **Correction applied:** the markdown draft's Finding 3 claim
      *"~70–76% of the distance mass falls in silence"* is **not present in any
      committed JSON** and was **removed**. Finding 3 now reports the
      JSON-backed decomposition from `results/silence_confound.json`: full
      ρ=0.756 → speech-only ρ=0.181 (n.s., p=0.52); silence-only ρ=0.775;
      speech-only+LOCO ρ=0.644. No other discrepancies found.

## 9. Optional strengthening experiment (1)

- [ ] **VOC-ALS graded cross-etiology arm.** The strongest single addition is a
      *graded* cross-corpus test: train severity on the current corpora and
      transfer to **VOC-ALS** (graded ALS via ALSFRS-R / bulbar subscore). This
      directly tests the falsifiable claim in §10/Limitations — whether a graded
      cross-etiology transfer can ever succeed. **Blocked on data access**
      (VOC-ALS download + per-speaker ALSFRS-R; pending Synapse/DUA). The harness
      already stubs it (`cross_corpus_severity.json → skipped_corpora → vocals`,
      with the exact activation steps). This is a *nice-to-have*, not required:
      the paper's thesis stands on the five committed negative findings.

---

### Quick pre-flight (run before every submission build)

```
python paper/veritas/figures/make_figures.py    # regenerate figs from JSON
#  then, on a LaTeX machine:
cd paper/veritas && pdflatex veritas_pitfalls && bibtex veritas_pitfalls \
  && pdflatex veritas_pitfalls && pdflatex veritas_pitfalls
```

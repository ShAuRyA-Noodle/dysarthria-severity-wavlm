# PC-GITA Email — ready to send

**TO:** `rafael.orozco@udea.edu.co`
**CC:** `rafael.orozco@i5.informatik.uni-erlangen.de`
**Subject:** `Research access request — PC-GITA database`

---

## Option A — copy-paste body

```
Dear Prof. Orozco-Arroyave,

I am Shaurya, a researcher working on a zero-shot dysarthric speech
severity metric (Articulatory Trajectory Deviation, ATD) built on frozen
WavLM-Large representations. We have validated the metric on TORGO and
UASpeech corpora and are preparing a manuscript targeting INTERSPEECH
2027.

To extend the work to motor speech disorders beyond English dysarthria,
I am requesting research access to the PC-GITA database. Specifically,
I would use the sustained-vowel and DDK subsets to test whether the
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
Shaurya
GitHub: https://github.com/ShAuRyA-Noodle/ShAuRyA_Dysar
```

## Option B — clickable mailto

The full URL-encoded `mailto:` link is in [`docs/pcgita_mailto.txt`](pcgita_mailto.txt).
Open that file, copy the entire one-line URL, paste into your browser
address bar, and your default mail client (Gmail / Outlook / Apple
Mail) opens with everything pre-filled. Hit send.

## What to expect back

- Typical response time: 1–3 weeks.
- Likely workflow: PI emails a Data Use Agreement (PDF) → you sign + scan
  → PI emails back a download URL or sends a hard drive (rare).
- Some requests are routed to GITA Lab admin, not PI directly.

## Once data arrives

1. Place archive under `data/raw/pc_gita/` (do **not** unzip into repo
   root — `.gitignore` covers `data/raw/` already).
2. Tell me, and I will write `src/cross_corpus/pcgita_parser.py` mirroring
   `ipvs_parser.py` and wire PC-GITA into the cross-corpus pipeline
   (Track D Hypothesis H1).

## Related citation

Orozco-Arroyave, J. R., Arias-Londoño, J. D., Vargas-Bonilla, J. F.,
González-Rátiva, M. C., Nöth, E. (2014). *New Spanish speech corpus
database for the analysis of people suffering from Parkinson's disease.*
Proc. LREC 2014.

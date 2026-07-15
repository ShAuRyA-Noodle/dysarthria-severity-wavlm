# VERITAS-Bench — full-stack showcase

A real full-stack web app that **demos the actual build**, not a static page.

```
webapp/
├── backend/          FastAPI · a thin, honest API over src/veritas/bench/
│   └── main.py         GET /api/report-cards  → the real committed verdicts
│                       POST /api/audit        → live PASS/WARN/FAIL from the
│                                                real thresholds.py policy
└── frontend/         React + Vite + TypeScript
    └── src/            react-three-fiber (3-D signal field) · framer-motion
                        • ReportCards.tsx  — fetches /api/report-cards live
                        • AuditConsole.tsx — POSTs to /api/audit, real verdicts
```

## Run it (two terminals)

**1 · Backend** (Python — same env as the research build)
```bash
cd webapp
../.conda/envs/dysarthria/python.exe -m pip install -r backend/requirements.txt
../.conda/envs/dysarthria/python.exe -m uvicorn backend.main:app --reload --port 8000
# → http://localhost:8000/api/health
```
On Windows the interpreter is `C:/Users/Dell/.conda/envs/dysarthria/python.exe`.

**2 · Frontend** (Node 18+)
```bash
cd webapp/frontend
npm install
npm run dev
# → http://localhost:5173
```
`npm run dev` runs even before a full type-check (Vite/esbuild). For a production
build + type-check: `npm run build`.

Point the frontend at a different backend by setting `VITE_API_URL` in `frontend/.env`.

## What is genuinely "real" here
- **`POST /api/audit`** imports and runs `src/veritas/bench/thresholds.py` — the exact
  PASS/WARN/FAIL policy the benchmark uses. Enter a metric's numbers (or a preset) and
  you get the same verdict the paper reports.
- **`GET /api/report-cards`** serves `results/veritas/veritas_bench_report.json` +
  the SOTA-audit card — the committed results, unmodified.
- Every number rendered on the page traces to a committed `results/` file.

## Deploy (optional)
- **Backend** → Render / Railway / Fly.io (`uvicorn backend.main:app`).
- **Frontend** → Vercel / Netlify (`npm run build`; set `VITE_API_URL` to the deployed backend).

## Design
"The Validity Instrument" — dark teal-biased instrument panel, PASS/WARN/FAIL semantic
colour system, mono-forward readout typography, an animated 3-D speech-signal field in the
hero. Light + dark themes, reduced-motion aware, keyboard accessible.

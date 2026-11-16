import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { runAudit, type AuditInput, type AuditResult } from "../api";

type Form = Required<Omit<AuditInput, "name">>;

const EMPTY: Form = {
  kind: "reference",
  predictive_rho: 0.77,
  predictive_p: 0.001,
  leakage_delta: 0.0,
  silence_retention: 0.24,
  silence_fraction: 0.63,
  energy_retention: 0.96,
  transfer_rho: 0.43,
  transfer_p: 0.11,
  identity_auc: 0.996,
};

// Real committed metric profiles — one click to reproduce a published verdict.
const PRESETS: Record<string, Form> = {
  "ATD · L9": { kind: "reference", predictive_rho: 0.772, predictive_p: 0.0008, leakage_delta: 0.0, silence_retention: 0.235, silence_fraction: 0.63, energy_retention: 0.96, transfer_rho: 0.432, transfer_p: 0.108, identity_auc: 0.996 },
  "Pooled probe": { kind: "probe", predictive_rho: 0.745, predictive_p: 0.0015, leakage_delta: 0, silence_retention: 1.0, silence_fraction: 0.63, energy_retention: 0.8, transfer_rho: 0.432, transfer_p: 0.108, identity_auc: 0.996 },
  "Phono-subspace (SOTA)": { kind: "reference", predictive_rho: 0.741, predictive_p: 0.0016, leakage_delta: 0.0, silence_retention: 1.0, silence_fraction: 0.63, energy_retention: 0.88, transfer_rho: 0.467, transfer_p: 0.079, identity_auc: 0.999 },
  "Ideal metric": { kind: "probe", predictive_rho: 0.8, predictive_p: 0.001, leakage_delta: 0, silence_retention: 0.95, silence_fraction: 0.3, energy_retention: 0.9, transfer_rho: 0.6, transfer_p: 0.01, identity_auc: 0.6 },
};

const FIELDS: { key: keyof Form; label: string; unit?: string }[] = [
  { key: "predictive_rho", label: "Predictive ρ (within-corpus LOSO)" },
  { key: "predictive_p", label: "Predictive p" },
  { key: "leakage_delta", label: "Leakage Δ (naive − LOCO)" },
  { key: "silence_retention", label: "Silence retention", unit: "0–1" },
  { key: "silence_fraction", label: "Silence fraction", unit: "0–1" },
  { key: "energy_retention", label: "Energy retention", unit: "0–1" },
  { key: "transfer_rho", label: "Cross-corpus ρ" },
  { key: "transfer_p", label: "Cross-corpus p" },
  { key: "identity_auc", label: "Identity AUC", unit: "0.5–1" },
];

const NAME: Record<string, string> = {
  predictive_validity: "Predictive validity",
  leakage_loco: "Leakage (LOCO)",
  silence_confound: "Silence",
  energy_confound: "Energy",
  cross_corpus_transfer: "Cross-corpus",
  leakage_by_layer: "Identity AUC",
};

export default function AuditConsole() {
  const [form, setForm] = useState<Form>({ ...PRESETS["ATD · L9"] });
  const [res, setRes] = useState<AuditResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function set<K extends keyof Form>(k: K, v: Form[K]) {
    setForm((f) => ({ ...f, [k]: v }));
  }

  async function run() {
    setBusy(true);
    setErr(null);
    try {
      const out = await runAudit({ name: "your_metric", ...form });
      setRes(out);
    } catch (e) {
      setErr("Backend unreachable. Start it: uvicorn backend.main:app --port 8000");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="console">
      <div>
        <div className="presets">
          {Object.keys(PRESETS).map((k) => (
            <button key={k} className="chip" onClick={() => { setForm({ ...PRESETS[k] }); setRes(null); }}>
              {k}
            </button>
          ))}
        </div>
        <div className="field">
          <label>Metric kind</label>
          <select
            value={form.kind}
            onChange={(e) => set("kind", e.target.value as Form["kind"])}
            style={{ fontFamily: "var(--mono)", fontSize: 13, padding: "9px 11px", background: "var(--panel-2)", color: "var(--ink)", border: "1px solid var(--line-strong)", borderRadius: 4 }}
          >
            <option value="reference">reference</option>
            <option value="probe">probe</option>
            <option value="intrinsic">intrinsic</option>
          </select>
        </div>
        {FIELDS.map((f) => (
          <div className="field" key={f.key}>
            <label>{f.label}{f.unit && <span className="u">{f.unit}</span>}</label>
            <input
              type="number"
              step="0.001"
              value={form[f.key] as number}
              onChange={(e) => set(f.key, parseFloat(e.target.value) as Form[typeof f.key])}
            />
          </div>
        ))}
        <button className="runbtn" onClick={run} disabled={busy}>
          {busy ? "Auditing…" : "Run the battery →"}
        </button>
      </div>

      <div className="out">
        {err && <div className="err">{err}</div>}
        {!err && !res && (
          <div className="loading">
            Enter a metric's numbers (or pick a preset) and run the audit. The verdict comes from the
            real threshold policy in <span className="mono">src/veritas/bench/thresholds.py</span>, live.
          </div>
        )}
        <AnimatePresence mode="wait">
          {res && (
            <motion.div
              key={res.overall + JSON.stringify(res.results).length}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.35 }}
            >
              <div className={`verdict ${res.overall === "N/A" ? "" : res.overall}`}>
                <div className="big">{res.overall}</div>
                <small>
                  {res.overall === "PASS"
                    ? "Survives every applicable validity control."
                    : res.overall === "WARN"
                    ? "A confound is present but not disqualifying."
                    : "Fails at least one validity control — the headline number hides a confound."}
                </small>
              </div>
              <div className="rows">
                {res.results.map((r) => (
                  <div className="row" key={r.control}>
                    <span className="c">{NAME[r.control] ?? r.control}</span>
                    <span style={{ display: "flex", gap: 12, alignItems: "center" }}>
                      <span className="s">{r.stat}</span>
                      <span className={`badge ${r.outcome === "N/A" ? "NA" : r.outcome}`}>{r.outcome}</span>
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

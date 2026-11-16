import { useEffect, useState } from "react";
import { getReportCards } from "../api";

const METRIC_LABEL: Record<string, string> = {
  atd_wavlm_l9: "ATD · L9",
  atd_wavlm_l1: "ATD · L1",
  pooled_wavlm_probe: "Pooled probe",
  phono_subspace_l9: "Phono-sub",
};

const CONTROLS: { key: string; label: string }[] = [
  { key: "predictive_validity", label: "Predictive validity" },
  { key: "leakage_loco", label: "Leakage (LOCO Δ)" },
  { key: "silence_confound", label: "Silence (kept)" },
  { key: "energy_confound", label: "Energy (kept)" },
  { key: "cross_corpus_transfer", label: "Cross-corpus" },
  { key: "leakage_by_layer", label: "Identity AUC" },
];

function shortVal(key: string, test: any): string {
  if (!test || test.outcome === "N/A") return test?.outcome === "N/A" ? "" : "";
  const s = test.stats || {};
  switch (key) {
    case "predictive_validity": return s.rho != null ? s.rho.toFixed(2).replace(/^0/, "") : "";
    case "leakage_loco": return s.delta_rho != null ? s.delta_rho.toFixed(2).replace(/^0/, "") : "";
    case "silence_confound": return s.retention != null ? Math.round(s.retention * 100) + "%" : "";
    case "energy_confound": return s.retention != null ? Math.round(s.retention * 100) + "%" : "";
    case "cross_corpus_transfer": return (s.p != null && s.p >= 0.05) ? "n.s." : (s.transfer_rho != null ? s.transfer_rho.toFixed(2).replace(/^0/, "") : "");
    case "leakage_by_layer": return s.speaker_identity_auc != null ? s.speaker_identity_auc.toFixed(3).replace(/^0/, "") : "";
    default: return "";
  }
}

function cls(o?: string) {
  return o === "PASS" ? "pass" : o === "WARN" ? "warn" : o === "FAIL" ? "fail" : "na";
}

export default function ReportCards() {
  const [cards, setCards] = useState<any[] | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    getReportCards().then((d) => setCards(d.cards)).catch(() => setErr(true));
  }, []);

  if (err)
    return <div className="err">Backend unreachable. Start it: <span className="mono">uvicorn backend.main:app --port 8000</span></div>;
  if (!cards) return <div className="loading">Fetching report cards from the API…</div>;

  return (
    <div className="cardwrap">
      <div className="cardtop">
        <div className="t">VERITAS-Bench · TORGO · seed 42 · <span style={{ color: "var(--signal)" }}>live from /api/report-cards</span></div>
        <div className="legend">
          <span><i className="dot p" />PASS</span><span><i className="dot w" />WARN</span><span><i className="dot f" />FAIL</span>
        </div>
      </div>
      <div className="tblscroll">
        <table className="rc">
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>Control</th>
              {cards.map((c) => <th key={c.metric_name}>{METRIC_LABEL[c.metric_name] ?? c.metric_name}</th>)}
            </tr>
          </thead>
          <tbody>
            {CONTROLS.map((ctrl) => (
              <tr key={ctrl.key}>
                <th scope="row">{ctrl.label}</th>
                {cards.map((c) => {
                  const test = c.tests?.[ctrl.key];
                  if (!test) return <td key={c.metric_name}><span className="dash">—</span></td>;
                  return (
                    <td key={c.metric_name}>
                      <span className={`v ${cls(test.outcome)}`}>
                        <span className="tag2">{test.outcome}</span>
                        {shortVal(ctrl.key, test) && <span className="val">{shortVal(ctrl.key, test)}</span>}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
            <tr className="overall">
              <th scope="row">Overall</th>
              {cards.map((c) => (
                <td key={c.metric_name}><span className={`v ${cls(c.overall)}`}><span className="tag2">{c.overall}</span></span></td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

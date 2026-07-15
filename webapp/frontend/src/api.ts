// Client for the VERITAS-Bench FastAPI backend.
const BASE = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

export type Outcome = "PASS" | "WARN" | "FAIL" | "N/A";

export interface AuditInput {
  name: string;
  kind: "reference" | "probe" | "intrinsic";
  predictive_rho?: number | null;
  predictive_p?: number | null;
  leakage_delta?: number | null;
  silence_retention?: number | null;
  silence_fraction?: number | null;
  energy_retention?: number | null;
  transfer_rho?: number | null;
  transfer_p?: number | null;
  identity_auc?: number | null;
}

export interface AuditResult {
  name: string;
  kind: string;
  results: { control: string; outcome: Outcome; stat: string }[];
  overall: Outcome;
}

async function j<T>(p: Promise<Response>): Promise<T> {
  const r = await p;
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

export interface Summary {
  n_metrics: number; n_pass_all: number;
  rho_min: number; rho_max: number; auc_min: number; auc_max: number;
  atd_silence_lost_pct: number; atd_identity_auc: number; atd_transfer_outcome: string;
}

export const getReportCards = () => j<{ cards: any[]; corpus: string; seed: number }>(fetch(`${BASE}/api/report-cards`));
export const getFindings = () => j<{ findings: any[] }>(fetch(`${BASE}/api/findings`));
export const getSummary = () => j<Summary>(fetch(`${BASE}/api/summary`));
export const runAudit = (body: AuditInput) =>
  j<AuditResult>(fetch(`${BASE}/api/audit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }));

export const API_BASE = BASE;

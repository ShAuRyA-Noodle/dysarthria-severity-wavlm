import { useEffect, useState } from "react";
import { getFindings } from "../api";

// All findings come live from the backend (/api/findings) — nothing hard-coded.
export default function Findings() {
  const [items, setItems] = useState<any[] | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    getFindings().then((d) => setItems(d.findings)).catch(() => setErr(true));
  }, []);

  if (err) return <div className="err">Backend unreachable — start it on :8000.</div>;
  if (!items) return <div className="loading">Fetching findings from the API…</div>;

  return (
    <div className="findings">
      {items.map((x) => (
        <div className="finding" key={x.key}>
          <div className="fstat num">{x.stat}</div>
          <h3>{x.title}</h3>
          <p>{x.detail}</p>
          <div className="src">{x.source}</div>
        </div>
      ))}
    </div>
  );
}

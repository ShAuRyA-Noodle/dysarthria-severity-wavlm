import { useState, type ReactNode } from "react";
import { motion } from "framer-motion";
import Scene from "./components/Scene";
import ReportCards from "./components/ReportCards";
import AuditConsole from "./components/AuditConsole";

function Reveal({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.6, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </motion.div>
  );
}

function Nav({ onToggle }: { onToggle: () => void }) {
  return (
    <div className="bar">
      <div className="wrap">
        <div className="brand" aria-label="VERITAS-Bench">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <path d="M3 16 H9 L12.5 7 L19 25 L22 16 H29" fill="none" stroke="var(--signal)" strokeWidth="2.1" strokeLinejoin="round" strokeLinecap="round" />
          </svg>
          VERITAS<b>-BENCH</b>
        </div>
        <nav className="navlinks">
          <a href="#battery">Battery</a>
          <a href="#reportcard">Report card</a>
          <a href="#console">Live audit</a>
          <a href="#papers">Papers</a>
          <button className="toggle" onClick={onToggle} aria-label="Toggle theme">Theme</button>
        </nav>
      </div>
    </div>
  );
}

const CONTROLS = [
  ["01", "Predictive validity", "Within-corpus leave-one-speaker-out Spearman ρ against clinical severity. Does it recover severity at all?", "PASS ρ ≥ 0.50, p < 0.05"],
  ["02", "Speaker-identity leakage", "Naive reference vs a leave-one-control-out reference. A reference containing the test speaker's own data inflates the score.", "PASS Δ ≤ 0.05"],
  ["03", "Silence confound", "Does the signal survive silence removal? Retention of ρ on speech-only frames, plus the silence fraction.", "PASS retain ≥ 70% & silence < 50%"],
  ["04", "Energy confound", "Partial Spearman controlling for mean speech energy. If loudness explains it, the metric is an energy artifact.", "PASS partial ≥ 70%"],
  ["05", "Cross-corpus transfer", "Train a probe on the other corpora, test held-out. Single-corpus success that doesn't transfer is overfitting.", "PASS ρ ≥ 0.50, p < 0.05"],
  ["06", "Representation identifiability", "Speaker-classification AUC of the features the metric uses. High identifiability is a leakage risk.", "PASS AUC ≤ 0.70"],
];

const FINDINGS = [
  ["−0.001", "Cycle-consistency isn't reliability", "Energy-controlled partial Spearman vs measured EMA inversion error over 22,854 frames.", "h1_faithfulness_vs_inversion_error.json"],
  ["0.775", "Pausing, not articulation", "Silence frames alone reproduce the full metric's severity correlation. Real signal (0.644) only survives silence-trim + de-leak.", "silence_confound.json"],
  ["0.089", "Early layers leak identity", "A naive early-layer reference drops by this much under a leave-one-control-out reference. Mid-network layers are robust.", "deleak_report_fast.json"],
  ["0.39", "Inversion is unstable on disordered input", "SSL-inversion kinematics don't track measured EMA (n.s.); the 'significant' severity correlation points the clinically wrong way.", "m2_articulatory_kinematics.json"],
  ["n ≈ 2", "The ground truth barely exists", "Usable dysarthric electromagnetic articulography sessions in public data — a hard ceiling on validating pathological inversion.", "h1_dysarthric_inversion.json"],
  ["metric-dependent", "Transfer must be shown, not assumed", "A well-designed metric transfers at scale; naive trajectory-distance at n=15 does not. Never assume generalisation.", "cross_corpus_severity.json"],
];

export default function App() {
  const [theme, setTheme] = useState<string>(() => document.documentElement.getAttribute("data-theme") || "dark");
  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    setTheme(next);
  }

  return (
    <>
      <Nav onToggle={toggle} />

      <header className="hero">
        <div className="canvaswrap"><Scene /></div>
        <div className="wrap">
          <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease: [0.2, 0.7, 0.2, 1] }}>
            <div className="tag"><i />Validity, not accuracy</div>
            <h1 className="title">A single correlation is not <span className="b">validation</span>.</h1>
            <p className="lede">
              Training-free metrics on frozen self-supervised speech models score dysarthria severity with no labels and no fine-tuning — judged by one number on one corpus. <b>VERITAS-Bench</b> audits any such metric against six confound controls and returns a live PASS / WARN / FAIL verdict. Every metric we tested passes its headline number — and <b>fails at least one validity control.</b>
            </p>
            <div className="cta">
              <a className="btn primary" href="#console">Run a live audit</a>
              <a className="btn" href="#reportcard">See the report card</a>
            </div>
          </motion.div>
        </div>
      </header>

      <div className="strip">
        <div className="cell"><div className="k">4</div><div className="l">metric families audited on real dysarthric speech</div></div>
        <div className="cell"><div className="k fail">0</div><div className="l">pass all six validity controls</div></div>
        <div className="cell"><div className="k sig num">.74–.77</div><div className="l">headline Spearman ρ — all look successful</div></div>
        <div className="cell"><div className="k num">.996–.999</div><div className="l">speaker-identity AUC — every metric</div></div>
      </div>

      <section id="problem">
        <div className="wrap">
          <Reveal>
            <div className="head">
              <div className="eyebrow">The problem</div>
              <h2>The number says it works. The confounds say what it measures.</h2>
              <p>On the trajectory-distance metric, the single reported ρ = 0.77 hides three confounds at once.</p>
            </div>
          </Reveal>
          <div className="grid3">
            <Reveal><div className="card"><div className="h">Silence</div><div className="big fail">76%</div><p>of the severity signal disappears when silence is removed. The metric measures <em>pausing</em>, not articulation.</p></div></Reveal>
            <Reveal><div className="card"><div className="h">Identity</div><div className="big fail">0.996</div><p>speaker-classification AUC (chance 0.5). A score that encodes <em>who</em> the speaker is tracks any speaker-correlated label.</p></div></Reveal>
            <Reveal><div className="card"><div className="h">Transfer</div><div className="big warn">n.s.</div><p>graded cross-corpus transfer is an underpowered null at n = 15 — not proof of absence, but not generalisation either.</p></div></Reveal>
          </div>
        </div>
      </section>

      <section id="battery">
        <div className="wrap">
          <Reveal><div className="head"><div className="eyebrow">The instrument</div><h2>Six controls. One verdict.</h2><p>Feed any per-speaker metric and a corpus. A metric doesn't win by scoring highest — it wins by surviving the controls.</p></div></Reveal>
          <Reveal>
            <div className="controls">
              {CONTROLS.map(([no, h, p, thr]) => (
                <div className="ctrl" key={no}><div className="no">{no}</div><div><h3>{h}</h3><p>{p}</p><div className="thr">{thr}</div></div></div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      <section id="reportcard">
        <div className="wrap">
          <Reveal><div className="head"><div className="eyebrow">The verdict</div><h2>The report card, on real dysarthric speech.</h2><p>Four metric families on TORGO (n = 15 graded, with articulography). Read down each column — every metric fails at least one control. Fetched live from the backend.</p></div></Reveal>
          <Reveal><ReportCards /></Reveal>
          <Reveal><p className="readout"><b>The battery discriminates.</b> The trajectory-distance metric fails on silence and identity; the pooled probe is silence-robust yet still fails transfer and identity; the phonological-subspace SOTA passes silence, energy and leakage — the instrument certifies good design where it exists, and surfaces the identity confound its authors never checked.</p></Reveal>
        </div>
      </section>

      <section id="console">
        <div className="wrap">
          <Reveal><div className="head"><div className="eyebrow">Live demo · demoing the real build</div><h2>Audit your own metric.</h2><p>Enter a metric's numbers — or load one of the published profiles — and run the battery. The verdict is computed by the <em>real</em> threshold policy in <span className="mono">src/veritas/bench/thresholds.py</span>, served by the FastAPI backend.</p></div></Reveal>
          <Reveal><AuditConsole /></Reveal>
        </div>
      </section>

      <section id="findings">
        <div className="wrap">
          <Reveal><div className="head"><div className="eyebrow">Grounded in measurement</div><h2>Findings, each with a real diagnostic.</h2></div></Reveal>
          <div className="findings">
            {FINDINGS.map(([stat, h, p, src]) => (
              <Reveal key={h as string}><div className="finding"><div className="fstat num">{stat}</div><h3>{h}</h3><p>{p}</p><div className="src">{src}</div></div></Reveal>
            ))}
          </div>
        </div>
      </section>

      <section id="explainer">
        <div className="wrap">
          <Reveal>
            <div className="head" style={{ marginBottom: 26 }}>
              <div className="eyebrow">Never seen any of this before?</div>
              <h2>The whole project, explained like you're ten.</h2>
              <p>Every piece — how it started, what's new, the novelty, and how each part was coded — in the simplest words, with lots of examples. And nothing here is mocked: every number traces to a committed <span className="mono">results/</span> file, and the Audit Console above runs the actual <span className="mono">thresholds.py</span> code, live.</p>
            </div>
          </Reveal>
          <Reveal>
            <a className="btn primary" href="https://claude.ai/code/artifact/b1ae4d1e-8857-47fc-8223-9df047197fbb" target="_blank" rel="noreferrer" style={{ textDecoration: "none", display: "inline-block" }}>
              Read the baby-step explainer →
            </a>
          </Reveal>
        </div>
      </section>

      <section id="papers">
        <div className="wrap">
          <Reveal><div className="head"><div className="eyebrow">Outputs</div><h2>Two papers, one released instrument.</h2><p>Every number traces to a committed result file. Seed 42; frozen weights; speaker-disjoint splits.</p></div></Reveal>
          <div className="papers">
            <Reveal><div className="paper"><div className="k">Paper A · audit</div><h3>When Frozen SSL Speech Metrics Don't Measure What They Claim</h3><p>The empirical validity audit: five confound findings on real dysarthric speech, grounded against articulography.</p><div className="m">IEEE · under submission</div></div></Reveal>
            <Reveal><div className="paper"><div className="k">Paper B · tool</div><h3>VERITAS-Bench: A Validity Test-Suite for Clinical-Speech Metrics</h3><p>The instrument — six controls, a documented threshold policy, a plug-in API, and report cards including the SOTA audit.</p><div className="m">IEEE / Springer · under submission</div></div></Reveal>
            <Reveal><div className="paper"><div className="k">Release</div><h3>Plug in your own metric</h3><p>Implement one method, receive a machine-readable PASS/WARN/FAIL card. A metric's validity becomes a regression test.</p><div className="m">src/veritas/bench · reproducible</div></div></Reveal>
          </div>
        </div>
      </section>

      <footer>
        <div className="wrap">
          <div className="foot">
            <div><span className="mono">VERITAS-BENCH</span> — validity, not accuracy, for training-free clinical-speech metrics.</div>
            <div className="mono">React · Three.js · Framer Motion · FastAPI</div>
          </div>
        </div>
      </footer>
    </>
  );
}

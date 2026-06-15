import React from "react";
import { Spinner, fmt, duration } from "../ui.jsx";

// Cap how many per-structure cells we draw, so a very large run can't flood the
// DOM. The segmented bar + phase tally still summarise every structure, and the
// Results table lists them all.
const GRID_CAP = 480;

// Turn the engine's low-level stage words into a clear, ordered pipeline that a
// biologist can follow at a glance — no run-log reading required. Each mode has
// a fixed sequence of named phases; the current engine stage maps to one of them
// and everything up to it is shown as complete.

const PREDICT_PHASES = [
  { key: "prepare", label: "Prepare", activity: "Preparing your input…" },
  { key: "msa", label: "Sequence search (MSA)", activity: "Searching databases for related sequences — the multiple-sequence alignment (MSA)…" },
  { key: "fold", label: "Fold", activity: "Folding the 3D structure…" },
  { key: "score", label: "Score", activity: "Scoring confidence and binding affinity…" },
  { key: "save", label: "Finish", activity: "Writing the final structure…" },
];
const PREDICT_STAGE_KEY = {
  loading: "prepare", featuriz: "prepare", prep: "prepare", start: "prepare",
  msa: "msa",
  trunk: "fold", pairformer: "fold", diffusion: "fold", sampling: "fold",
  affinity: "score", confidence: "score",
  writing: "save", saving: "save",
};

const DESIGN_PHASES = [
  { key: "design", label: "Generate", activity: "Generating candidate binders…" },
  { key: "inverse_folding", label: "Design sequence", activity: "Designing amino-acid sequences for the binders…" },
  { key: "folding", label: "Re-fold", activity: "Re-folding each design to check it holds its shape…" },
  { key: "analysis", label: "Analyze", activity: "Scoring each design against the target…" },
  { key: "filtering", label: "Rank", activity: "Ranking and keeping the most promising designs…" },
];
const DESIGN_STAGE_KEY = {
  design: "design", inverse_folding: "inverse_folding",
  folding: "folding", design_folding: "folding",
  analysis: "analysis", filtering: "filtering",
};

function phaseModel(job) {
  const isDesign = job.kind === "design";
  let phases = isDesign ? DESIGN_PHASES : PREDICT_PHASES.slice();
  // ESMFold-2 Fast folds single-sequence — no MSA step to show.
  if (!isDesign && job.model === "esmfold2-fast") phases = phases.filter((p) => p.key !== "msa");
  const keyMap = isDesign ? DESIGN_STAGE_KEY : PREDICT_STAGE_KEY;
  const curKey = job.stage != null ? keyMap[job.stage] : null;
  let index = curKey ? Math.max(0, phases.findIndex((p) => p.key === curKey)) : 0;
  if (job.status === "succeeded") index = phases.length; // everything done
  return { phases, index };
}

// For a run with many structures, a single linear stepper can't represent them
// (each is folded on its own device, in its own stage). Show overall progress
// plus a live per-structure grid instead.
const MULTI_VERB = {
  prepare: "Preparing inputs…",
  msa: "Searching for related sequences (MSA)…",
  fold: "Folding structures…",
  score: "Scoring confidence & binding affinity…",
  save: "Finishing up…",
};

// The most telling single number for a finished structure, for the chip tooltip.
function chipMetric(r) {
  if (r.plddt != null) return `pLDDT ${fmt(r.plddt, 2)}`;
  if (r.confidence_score != null) return `confidence ${fmt(r.confidence_score, 2)}`;
  if (r.ptm != null) return `pTM ${fmt(r.ptm, 2)}`;
  return null;
}

const STATE_ICON = { done: "✓", failed: "✗", running: "⟳", queued: "•" };
const STAGE_LABEL = { prepare: "preparing", msa: "MSA search", fold: "folding", score: "scoring", save: "finishing" };
const STAGE_ORDER = ["prepare", "msa", "fold", "score", "save"];

// "3 folding · 2 scoring · 1 MSA search" — the running inputs by phase.
function runningSummary(cells) {
  const by = {};
  cells.forEach((c) => { if (c.state === "running") { const l = STAGE_LABEL[c.stage] || "running"; by[l] = (by[l] || 0) + 1; } });
  const order = STAGE_ORDER.map((s) => STAGE_LABEL[s]).concat("running");
  return order.filter((l) => by[l]).map((l) => `${by[l]} ${l}`).join(" · ");
}

function MultiTargetProgress({ job, results }) {
  const rows = (results && results.rows) || [];
  const byId = {};
  rows.forEach((r) => { if (r && r.id != null) byId[r.id] = r; });
  const done = job.status === "succeeded";
  const stopped = job.status === "failed" || job.status === "canceled";
  const waiting = job.status === "queued";
  const active = !done && !stopped;

  // One cell per structure. While running, the server tells us each input's
  // live state (done / running) plus a queued count; once finished (or with no
  // live feed) we derive it from the result rows. Stable, and scales to many.
  let cells;
  if (job.targets) {
    cells = job.targets.map((t) => ({ id: t.id, state: t.state, stage: t.stage, row: byId[t.id] }));
  } else {
    cells = rows.map((r) => ({
      id: r.id, row: r, state: r.status && r.status !== "ok" ? "failed" : "done",
    }));
    for (let i = 0; i < Math.max(0, (job.total || 0) - rows.length); i++) cells.push({ state: "queued" });
  }
  const total = cells.length;
  const count = (s) => cells.reduce((a, c) => a + (c.state === s), 0);
  const doneN = count("done"), failN = count("failed"), runN = count("running"), queuedN = count("queued");
  const finished = doneN + failN;

  const { phases, index } = phaseModel(job);
  const curKey = phases[Math.min(index, phases.length - 1)]?.key;
  const verb = waiting ? "Waiting for a free device…" : (MULTI_VERB[curKey] || "Working…");
  const compact = total > 30; // dot grid scales to hundreds; chips stay readable below

  const tip = (c) => {
    if (!c.id) return "Queued";
    if (c.state === "failed") return `${c.id} · failed`;
    if (c.state === "running") return `${c.id} · ${STAGE_LABEL[c.stage] || "running"}`;
    if (c.state === "queued") return `${c.id} · queued`;
    const m = c.row ? chipMetric(c.row) : null;
    return `${c.id}${m ? ` · ${m}` : ""}`;
  };

  return (
    <div className="mtp">
      <div className="mtp-head">
        {active ? <span className="mtp-verb"><Spinner /> {verb}</span>
          : done ? <span className="jp-done">✓ {doneN} of {total} structure{total > 1 ? "s" : ""} complete{failN ? ` · ${failN} failed` : ""}</span>
          : <span className="muted">{job.status === "canceled" ? "Canceled." : "Stopped before finishing."}</span>}
        <span className="spacer" />
        {!done && <span className="mtp-count">{finished} / {total}</span>}
        <span className="muted small">{duration(job)}</span>
      </div>
      {/* Proportional segmented bar — one element regardless of N, so the
          overall make-up (done/running/queued/failed) reads at a glance whether
          there are 3 inputs or 3000. */}
      {active && (
        <div className="mtp-seg" title={`${doneN} done · ${runN} running · ${queuedN} queued${failN ? ` · ${failN} failed` : ""}`}>
          {[["done", doneN], ["failed", failN], ["running", runN], ["queued", queuedN]].map(
            ([k, v]) => (v > 0 ? <span key={k} className={`seg ${k}`} style={{ flexGrow: v }} /> : null))}
        </div>
      )}
      {(runN > 0 || queuedN > 0 || failN > 0) && (
        <div className="mtp-tally">
          {doneN > 0 && <span className="t done">{doneN} done</span>}
          {runN > 0 && <span className="t running">{runningSummary(cells)}</span>}
          {queuedN > 0 && <span className="t queued">{queuedN} queued</span>}
          {failN > 0 && <span className="t failed">{failN} failed</span>}
        </div>
      )}
      {/* Per-structure detail. Chips (named + phase) stay readable up to ~30;
          beyond that a dot heatmap. Cap the rendered cells so a huge run can't
          flood the DOM — the bar + tally above already cover every structure,
          and the searchable Results table gives full per-structure access. */}
      <div className={`mtp-grid ${compact ? "compact" : ""}`}>
        {cells.slice(0, GRID_CAP).map((c, i) => (
          compact
            ? <span key={c.id || `q${i}`} className={`mtp-dot ${c.state}`} title={tip(c)} />
            : <span key={c.id || `q${i}`} className={`mtp-chip ${c.state}`} title={tip(c)}>
                <span className="mtp-ic">{STATE_ICON[c.state]}</span>
                {c.id && <span className="mtp-name">{c.id}</span>}
                {c.state === "running" && c.stage && <span className="mtp-sub">{STAGE_LABEL[c.stage]}</span>}
              </span>
        ))}
      </div>
      {total > GRID_CAP && (
        <p className="hint" style={{ marginTop: 8 }}>
          Showing {GRID_CAP} of {total} structures above · the bar and tally cover all of them, and the Results table lists every one.
        </p>
      )}
    </div>
  );
}

export default function JobProgress({ job, results }) {
  // Many structures at once → per-structure overview; one structure (or a design
  // run, which is a single ranked-design pipeline) → the linear stepper.
  if (job.kind === "predict" && (job.total || 0) > 1) {
    return <MultiTargetProgress job={job} results={results} />;
  }
  const { phases, index } = phaseModel(job);
  const queued = job.status === "queued";
  const done = job.status === "succeeded";
  const failed = job.status === "failed" || job.status === "canceled";
  const cur = phases[Math.min(index, phases.length - 1)];
  const running = !queued && !done && !failed;
  // Within-stage progress (e.g. diffusion 150/200), if the engine is reporting it.
  const sp = running && job.stage_progress && job.stage_progress.total > 1 ? job.stage_progress : null;

  return (
    <div className="jp">
      <ol className="jp-steps">
        {phases.map((p, i) => {
          // Mark the exact phase a failed run stopped at, so the user can see
          // how far it got (a plain "stopped" run leaves every later step
          // pending). Cancel is user-initiated, so we don't flag a phase for it.
          const failedHere = job.status === "failed" && i === index;
          const state = failedHere ? "failed"
            : done || i < index ? "done"
            : (i === index && !queued && !failed ? "active" : "pending");
          return (
            <li key={p.key} className={`jp-step ${state}`}>
              <span className="jp-dot">{state === "done" ? "✓" : state === "failed" ? "✗" : i + 1}</span>
              <span className="jp-label">{p.label}</span>
            </li>
          );
        })}
      </ol>
      <div className="jp-activity">
        {queued ? (
          <span className="muted">Waiting for a free device…</span>
        ) : failed ? (
          <span className="muted">{job.status === "canceled" ? "Canceled." : "This run stopped before finishing."}</span>
        ) : done ? (
          <span className="jp-done">✓ Complete</span>
        ) : (
          <span><Spinner /> {cur.activity}{sp ? ` · ${sp.step}/${sp.total}` : ""}</span>
        )}
        <span className="spacer" />
        <span className="muted small">{duration(job)}</span>
      </div>
      {sp && (
        <div className="jp-subbar" title={`${sp.stage} ${sp.step}/${sp.total}`}>
          <div className="jp-subbar-fill" style={{ width: `${Math.round(100 * sp.step / sp.total)}%` }} />
        </div>
      )}
    </div>
  );
}

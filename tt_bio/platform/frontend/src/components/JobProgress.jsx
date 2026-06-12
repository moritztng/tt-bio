import React from "react";
import { Spinner, Progress, fmt, duration } from "../ui.jsx";

// Turn the engine's low-level stage words into a clear, ordered pipeline that a
// biologist can follow at a glance — no run-log reading required. Each mode has
// a fixed sequence of named phases; the current engine stage maps to one of them
// and everything up to it is shown as complete.

const PREDICT_PHASES = [
  { key: "prepare", label: "Prepare", activity: "Preparing your input…" },
  { key: "msa", label: "Find relatives", activity: "Searching for evolutionary relatives (MSA)…" },
  { key: "fold", label: "Fold", activity: "Folding the 3-D structure…" },
  { key: "score", label: "Score", activity: "Scoring confidence and binding affinity…" },
  { key: "save", label: "Finish", activity: "Writing the final structure…" },
];
const PREDICT_STAGE_KEY = {
  loading: "prepare", featuriz: "prepare", prep: "prepare", start: "prepare",
  msa: "msa",
  trunk: "fold", pairformer: "fold", diffusion: "fold", sampling: "fold",
  affinity: "score",
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
  msa: "Finding evolutionary relatives (MSA)…",
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

function MultiTargetProgress({ job, results }) {
  const rows = (results && results.rows) || [];
  const completed = rows.length;
  const failed = rows.filter((r) => r.status && r.status !== "ok").length;
  const ok = completed - failed;
  const total = Math.max(job.total || 0, completed);
  const pending = Math.max(0, total - completed);
  const queued = job.status === "queued";
  const done = job.status === "succeeded";
  const stopped = job.status === "failed" || job.status === "canceled";
  const active = !done && !stopped;

  const { phases, index } = phaseModel(job);
  const curKey = phases[Math.min(index, phases.length - 1)]?.key;
  const verb = queued ? "Waiting for a free device on the cluster…" : (MULTI_VERB[curKey] || "Working…");

  return (
    <div className="mtp">
      <div className="mtp-head">
        {active ? (
          <span className="mtp-verb"><Spinner /> {verb}</span>
        ) : done ? (
          <span className="jp-done">✓ {ok} of {total} structure{total > 1 ? "s" : ""} complete{failed ? ` · ${failed} failed` : ""}</span>
        ) : (
          <span className="muted">{job.status === "canceled" ? "Canceled." : "Stopped before finishing."}</span>
        )}
        <span className="spacer" />
        {!done && <span className="mtp-count">{completed} / {total}</span>}
        <span className="muted small">{duration(job)}</span>
      </div>
      {active && <div className="mtp-bar"><Progress value={total ? completed / total : null} /></div>}
      <div className="mtp-grid">
        {rows.map((r) => {
          const bad = r.status && r.status !== "ok";
          const m = chipMetric(r);
          return (
            <span key={r.id} className={`mtp-chip ${bad ? "failed" : "done"}`}
                  title={`${r.id}${bad ? " · failed" : m ? ` · ${m}` : ""}`}>
              <span className="mtp-ic">{bad ? "✗" : "✓"}</span>
              <span className="mtp-name">{r.id}</span>
            </span>
          );
        })}
        {Array.from({ length: pending }).map((_, i) => (
          <span key={`p${i}`} className="mtp-chip pending" title={active ? "Folding…" : "Did not run"}>
            <span className="mtp-ic">•</span>
          </span>
        ))}
      </div>
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
          <span className="muted">Waiting for a free device on the cluster…</span>
        ) : failed ? (
          <span className="muted">{job.status === "canceled" ? "Canceled." : "This run stopped before finishing."}</span>
        ) : done ? (
          <span className="jp-done">✓ Complete</span>
        ) : (
          <span><Spinner /> {cur.activity}</span>
        )}
        {job.kind === "predict" && job.total > 1 && !done && !failed && (
          <span className="muted"> · {job.done}/{job.total} structures</span>
        )}
        <span className="spacer" />
        <span className="muted small">{duration(job)}</span>
      </div>
    </div>
  );
}

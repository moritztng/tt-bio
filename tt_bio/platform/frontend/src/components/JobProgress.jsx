import React from "react";
import { Spinner, duration } from "../ui.jsx";

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

export default function JobProgress({ job }) {
  const { phases, index } = phaseModel(job);
  const queued = job.status === "queued";
  const done = job.status === "succeeded";
  const failed = job.status === "failed" || job.status === "canceled";
  const cur = phases[Math.min(index, phases.length - 1)];

  return (
    <div className="jp">
      <ol className="jp-steps">
        {phases.map((p, i) => {
          const state = done || i < index ? "done" : (i === index && !queued && !failed ? "active" : "pending");
          return (
            <li key={p.key} className={`jp-step ${state}`}>
              <span className="jp-dot">{state === "done" ? "✓" : i + 1}</span>
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

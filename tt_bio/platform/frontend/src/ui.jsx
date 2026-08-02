// Small shared UI helpers.
import React from "react";

export function Badge({ status }) {
  return <span className={`badge ${status}`}>{status}</span>;
}

export function Progress({ value }) {
  // value: 0..1, or null for indeterminate
  const indet = value == null;
  return (
    <div className={`progress ${indet ? "indet" : ""}`}>
      <span style={{ width: `${Math.round((indet ? 0.35 : value) * 100)}%` }} />
    </div>
  );
}

export function Spinner() {
  return <span className="spin" />;
}

export function fmt(v, digits = 3) {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (Number.isNaN(n)) return String(v);
  return n.toFixed(digits);
}

// Display names for the design engines, matching the CLI's `--model` vocabulary.
export const DESIGN_MODEL_LABEL = { boltzgen: "BoltzGen", rfd3: "RFdiffusion3" };

// The design model that produced/will produce a job, from the server-computed
// `engine` field (jobs.py derives it from the protocol). The prefix fallback
// covers a job payload from before the field existed.
export function designEngine(job) {
  if (job?.engine) return job.engine;
  return (job?.protocol || "").startsWith("rfd3") ? "rfd3" : "boltzgen";
}

export function designModelLabel(job) {
  return DESIGN_MODEL_LABEL[designEngine(job)] || "BoltzGen";
}

// Job-list / job-detail tag for a design job: "RFdiffusion3 · Motif scaffolding".
// Falls back to the raw protocol id when the catalog hasn't loaded yet.
export function designTag(catalog, job) {
  const proto = catalog?.protocols?.find((p) => p.id === job.protocol)?.name || job.protocol;
  return `${designModelLabel(job)} · ${proto || "design"}`;
}

export function timeAgo(ts) {
  if (!ts) return "";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function duration(job) {
  if (!job.started_at) return "";
  const end = job.finished_at || Date.now() / 1000;
  const s = Math.max(0, end - job.started_at);
  if (s < 60) return `${s.toFixed(0)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

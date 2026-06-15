import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { Badge, Spinner, duration, timeAgo } from "../ui.jsx";
import ResultsPredict from "./ResultsPredict.jsx";
import ResultsDesign from "./ResultsDesign.jsx";
import JobProgress from "./JobProgress.jsx";
import LogPanel from "./LogPanel.jsx";

export default function JobDetail({ jobId, onDeleted, onError }) {
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const load = () =>
      api.job(jobId).then((d) => {
        if (cancelled) return;
        setJob(d);
        const active = d.status === "running" || d.status === "queued";
        timer = setTimeout(load, active ? 1500 : 6000);
      }).catch(() => { if (!cancelled) timer = setTimeout(load, 4000); });
    load();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [jobId]);

  if (!job) return <div className="empty"><Spinner /> Loading…</div>;

  const active = job.status === "running" || job.status === "queued";
  const results = job.results || {};

  const cancel = async () => { setBusy(true); try { await api.cancel(jobId); } finally { setBusy(false); } };
  const remove = async () => {
    if (!confirm("Delete this job and its outputs?")) return;
    setBusy(true);
    try { await api.remove(jobId); onDeleted(); } catch (e) { onError(e.message); } finally { setBusy(false); }
  };

  return (
    <div>
      <div className="panel">
        <div className="flex-between">
          <div>
            <div className="flex">
              <h2 className="mb0">{job.name}</h2>
              <Badge status={job.status} />
            </div>
            <div className="ji-meta mt8">
              <span className="tag">{job.kind === "design" ? job.protocol : job.model}</span>
              <span>{job.kind === "design" ? "BoltzGen design" : "Structure prediction"}</span>
              <span className="sep">·</span>
              <span>submitted {timeAgo(job.created_at)}</span>
              {job.started_at && <><span className="sep">·</span><span>{duration(job)}</span></>}
            </div>
          </div>
          <div className="flex">
            {active && <button className="btn sm" disabled={busy} onClick={cancel}>Cancel</button>}
            {!active && <button className="btn sm danger" disabled={busy} onClick={remove}>Delete</button>}
          </div>
        </div>

        {(active || job.status === "succeeded") && (
          <div className="mt16"><JobProgress job={job} results={results} /></div>
        )}

        {job.status === "failed" && job.error && (
          <div className="mt16">
            <JobProgress job={job} results={results} />
            <div className="errbox mt16">
              <strong>This run couldn't finish.</strong>
              <p className="mt8">{_friendlyError(job.error)}</p>
              <details className="collapse mt8">
                <summary>Technical detail</summary>
                <pre className="log mt8">{job.error}</pre>
              </details>
            </div>
          </div>
        )}
      </div>

      {results.ready && (
        <div className="panel">
          <p className="section-title">Results</p>
          {job.kind === "predict"
            ? <ResultsPredict jobId={jobId} results={results} />
            : <ResultsDesign jobId={jobId} results={results} />}
        </div>
      )}

      {!results.ready && !active && job.status === "succeeded" && (
        <div className="panel"><div className="empty">No results were produced.</div></div>
      )}

      {/* The raw engine log is for advanced/debugging use only — collapsed by
          default so biologists see the visual progress + results, not a wall
          of technical output. */}
      <div className="panel">
        <details className="collapse">
          <summary>Technical log (advanced)</summary>
          <div className="mt8"><LogPanel jobId={jobId} live={active} /></div>
        </details>
      </div>
    </div>
  );
}

// Translate the most common engine failures into one plain sentence a biologist
// can act on; otherwise fall back to the first line of the raw error.
function _friendlyError(err) {
  const e = String(err || "");
  const has = (s) => e.toLowerCase().includes(s);
  if (has("msa") && (has("server") || has("colabfold") || has("timeout") || has("connection")))
    return "The MSA service couldn't be reached. This is usually temporary — please try again in a minute.";
  if (has("every target failed"))
    return "None of the inputs could be folded — check that the sequences are valid.";
  if (has("out of memory") || has("oom"))
    return "The input was too large to fit in memory for this demo. Try a smaller construct.";
  if (has("no protein") || has("no sequences") || has("invalid"))
    return "The input couldn't be read — please check the sequence and try again.";
  const first = e.split("\n").map((l) => l.trim()).filter(Boolean)[0] || "The run failed.";
  return first.length > 200 ? first.slice(0, 200) + "…" : first;
}

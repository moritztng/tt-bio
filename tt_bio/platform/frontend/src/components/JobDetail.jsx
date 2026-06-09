import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { Badge, Progress, Spinner, duration, timeAgo } from "../ui.jsx";
import ResultsPredict from "./ResultsPredict.jsx";
import ResultsDesign from "./ResultsDesign.jsx";
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
              <span className="muted small">{job.kind === "design" ? "BoltzGen design" : "Structure prediction"}</span>
              <span className="muted small">· submitted {timeAgo(job.created_at)}</span>
              {job.started_at && <span className="muted small">· {duration(job)}</span>}
            </div>
          </div>
          <div className="flex">
            {active && <button className="btn sm" disabled={busy} onClick={cancel}>Cancel</button>}
            {!active && <button className="btn sm danger" disabled={busy} onClick={remove}>Delete</button>}
          </div>
        </div>

        {active && (
          <div className="mt16">
            <Progress value={job.progress} />
            <div className="flex small muted mt8">
              <Spinner />
              <span>
                {job.kind === "predict" && job.total
                  ? `${job.done} / ${job.total} targets`
                  : job.status === "queued" ? "Queued…" : "Running…"}
                {job.stage ? ` · ${job.stage}` : ""}
              </span>
            </div>
          </div>
        )}

        {job.status === "failed" && job.error && (
          <pre className="log mt16" style={{ background: "#2a1416", color: "#ffd7d7" }}>{job.error}</pre>
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

      {!results.ready && !active && job.status !== "failed" && (
        <div className="panel"><div className="empty">No results were produced. Check the log below.</div></div>
      )}

      <div className="panel">
        <details className="collapse" open={active || job.status === "failed"}>
          <summary>Run log</summary>
          <div className="mt8"><LogPanel jobId={jobId} live={active} /></div>
        </details>
      </div>
    </div>
  );
}

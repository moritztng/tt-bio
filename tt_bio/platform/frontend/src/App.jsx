import React, { useEffect, useState, useCallback } from "react";
import { api } from "./api.js";
import { Badge, Progress, timeAgo } from "./ui.jsx";
import NewJob from "./components/NewJob.jsx";
import JobDetail from "./components/JobDetail.jsx";

export default function App() {
  const [catalog, setCatalog] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [selected, setSelected] = useState(null); // job id or null (= new job)
  const [error, setError] = useState(null);
  const [cluster, setCluster] = useState(null);

  useEffect(() => {
    api.catalog().then(setCatalog).catch((e) => setError(e.message));
  }, []);

  const refresh = useCallback(() => {
    api.jobs().then((d) => setJobs(d.jobs)).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 2500);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    const pull = () => api.cluster().then(setCluster).catch(() => {});
    pull();
    const t = setInterval(pull, 4000);
    return () => clearInterval(t);
  }, []);

  const onSubmitted = (job) => {
    refresh();
    setSelected(job.id);
  };

  const showError = (msg) => {
    setError(msg);
    setTimeout(() => setError(null), 5000);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="wordmark">
          <span>ai<span className="amp">&amp;</span></span>
          <span className="sub">Bio</span>
        </div>
        <span className="tagline">Drug discovery on sovereign compute</span>
        <div className="spacer" />
        <span className="pill">Boltz-2 · ESMFold-2 · BoltzGen</span>
      </header>

      <div className="main">
        <aside className="sidebar">
          <div className="sidebar-head">
            <button
              className={`btn primary block ${selected === null ? "" : ""}`}
              onClick={() => setSelected(null)}
            >
              + New prediction
            </button>
          </div>
          <div className="joblist">
            {jobs.length === 0 && (
              <div className="empty small">No jobs yet.<br />Submit one to get started.</div>
            )}
            {jobs.map((j) => (
              <button
                key={j.id}
                className={`jobitem ${selected === j.id ? "active" : ""}`}
                onClick={() => setSelected(j.id)}
              >
                <div className="ji-top">
                  <span className="ji-name">{j.name}</span>
                  <Badge status={j.status} />
                </div>
                <div className="ji-meta">
                  <span className="tag">{j.kind === "design" ? j.protocol : j.model}</span>
                  <span>{timeAgo(j.created_at)}</span>
                </div>
                {j.status === "running" && (
                  <div className="mt8"><Progress value={j.progress} /></div>
                )}
              </button>
            ))}
          </div>
          <Fleet cluster={cluster} />
        </aside>

        <main className="content">
          <div className="content-inner">
            {selected === null ? (
              <NewJob catalog={catalog} onSubmitted={onSubmitted} onError={showError} />
            ) : (
              <JobDetail
                jobId={selected}
                onDeleted={() => { setSelected(null); refresh(); }}
                onError={showError}
              />
            )}
          </div>
        </main>
      </div>

      {error && <div className="toast">{error}</div>}
    </div>
  );
}

// Live fleet status: how many galaxies/devices back the platform, how much is
// running right now, and the one command to add another galaxy to the pool.
function Fleet({ cluster }) {
  const [showJoin, setShowJoin] = useState(false);
  const [copied, setCopied] = useState(false);
  if (!cluster || !cluster.enabled) return null;
  const alive = cluster.controller_alive;
  const online = cluster.online_workers || 0;
  const hosts = cluster.hosts || [];
  const running = (cluster.jobs && cluster.jobs.running) || 0;
  const copy = () => {
    navigator.clipboard?.writeText(cluster.join_command).then(
      () => { setCopied(true); setTimeout(() => setCopied(false), 1500); }, () => {});
  };
  return (
    <div className="fleet">
      <div className="fleet-head">
        <span className={`fleet-dot ${alive ? "ok" : "off"}`} />
        <strong>Fleet</strong>
        <span className="spacer" />
        <span className="fleet-count">{online} device{online === 1 ? "" : "s"}</span>
      </div>
      <div className="fleet-sub">
        {hosts.length || (alive ? 0 : "—")} galax{hosts.length === 1 ? "y" : "ies"}
        {" · "}{running} running
      </div>
      {hosts.length > 0 && (
        <div className="fleet-hosts">
          {hosts.map((h) => (
            <div key={h.host} className="fleet-host" title={(h.accelerators || []).join(", ")}>
              <span className="fh-name">{h.host}{h.is_master ? " ★" : ""}</span>
              <span className="fh-dev">{h.devices}×</span>
            </div>
          ))}
        </div>
      )}
      <button className="btn ghost sm block" onClick={() => setShowJoin((s) => !s)}>
        {showJoin ? "Hide" : "+ Add a galaxy"}
      </button>
      {showJoin && (
        <div className="fleet-join">
          <div className="hint">Run this on another galaxy (reachable over the network) to add its devices:</div>
          <code className="join-cmd" onClick={copy} title="Click to copy">{cluster.join_command}</code>
          {copied && <div className="hint" style={{ color: "var(--accent)" }}>Copied ✓</div>}
        </div>
      )}
    </div>
  );
}

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
  const [navOpen, setNavOpen] = useState(false);  // mobile: job-list drawer open

  // Pick a job (or the new-job form) and close the mobile drawer behind it.
  const choose = (id) => { setSelected(id); setNavOpen(false); };

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
        <button className="nav-toggle" onClick={() => setNavOpen((o) => !o)}
                aria-label="Toggle job list" aria-expanded={navOpen}>
          <span /><span /><span />
        </button>
        <div className="wordmark">
          <img className="brand-logo" src="/aiand-logo.svg" alt="ai&amp;" />
          <span className="sub">Drug Discovery</span>
        </div>
        <span className="tagline">Sovereign compute in Japan</span>
        <div className="spacer" />
        <a className="powered" href="https://tenstorrent.com" target="_blank" rel="noopener noreferrer"
           title="Powered by Tenstorrent AI Processors">
          <span>Powered by</span>
          <img className="tt-logo" src="/tenstorrent-logo.svg" alt="Tenstorrent" />
          <span>AI&nbsp;Processors</span>
        </a>
      </header>

      <div className="main">
        <aside className={`sidebar ${navOpen ? "open" : ""}`}>
          <div className="sidebar-head">
            <button
              className="btn primary block"
              onClick={() => choose(null)}
            >
              + New job
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
                onClick={() => choose(j.id)}
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
        </aside>
        {navOpen && <div className="nav-backdrop" onClick={() => setNavOpen(false)} />}

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

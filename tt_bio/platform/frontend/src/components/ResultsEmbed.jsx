import React from "react";
import { api } from "../api.js";

export default function ResultsEmbed({ jobId, results }) {
  const seqs = results.sequences || [];

  if (!seqs.length) return <div className="empty">No embeddings were produced.</div>;

  return (
    <div>
      <div className="kv" style={{ marginBottom: 16 }}>
        <span className="k">Model</span><span className="metric">{results.model}</span>
        <span className="k">Pooling</span><span className="metric">{results.pool}</span>
        <span className="k">Embedding size</span><span className="metric">{results.d_model}</span>
        <span className="k">Format</span><span className="metric">{results.format}</span>
      </div>
      <p className="hint" style={{ marginTop: 0 }}>
        {results.format === "npz"
          ? "One .npz file per sequence — per-residue [length, d_model] and pooled [d_model] float32 arrays."
          : "One embeddings.parquet table — a pooled vector per sequence."}
      </p>

      <div className="results-toolbar">
        <span className="muted small">{seqs.length.toLocaleString()} sequence{seqs.length === 1 ? "" : "s"}</span>
        <div className="spacer" />
        <a className="btn sm" href={api.artifactUrl(jobId, "manifest.json")}>manifest.json</a>
        <a className="btn sm" href={api.archiveUrl(jobId)}>Download all (.zip)</a>
      </div>

      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>ID</th>
              <th>Length</th>
              <th>File</th>
            </tr>
          </thead>
          <tbody>
            {seqs.map((s) => (
              <tr key={s.id}>
                <td><strong>{s.id}</strong></td>
                <td className="metric">{s.length}</td>
                <td>{s.file ? <a href={api.artifactUrl(jobId, s.file)}>{s.file}</a> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

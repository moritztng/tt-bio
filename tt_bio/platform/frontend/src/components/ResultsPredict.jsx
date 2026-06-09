import React, { useState, useMemo } from "react";
import { api } from "../api.js";
import { fmt } from "../ui.jsx";
import StructureViewer from "./StructureViewer.jsx";

// Which metric columns to show, in order — only those present are rendered.
const COLS = [
  ["confidence_score", "Confidence"],
  ["plddt", "pLDDT"],
  ["complex_plddt", "Complex pLDDT"],
  ["ptm", "pTM"],
  ["iptm", "ipTM"],
  ["runtime_s", "Time (s)"],
];

function affinityLabel(row) {
  if (row.affinity_pred_value == null) return null;
  const log = Number(row.affinity_pred_value);
  const ic50 = Math.pow(10, log); // value is log10(IC50 in µM)
  const prob = row.affinity_probability_binary;
  return { log: log.toFixed(2), ic50: ic50 < 1 ? ic50.toFixed(3) : ic50.toFixed(1), prob };
}

export default function ResultsPredict({ jobId, results }) {
  const rows = results.rows || [];
  const structures = results.structures || {};
  const [selId, setSelId] = useState(rows[0]?.id);

  const presentCols = useMemo(
    () => COLS.filter(([k]) => rows.some((r) => r[k] != null)),
    [rows]
  );

  const sel = rows.find((r) => r.id === selId) || rows[0];
  const selFiles = (sel && structures[sel.id]) || [];
  const [fileIdx, setFileIdx] = useState(0);
  const file = selFiles[fileIdx];
  const aff = sel && affinityLabel(sel);

  return (
    <div>
      <table className="data">
        <thead>
          <tr>
            <th>Target</th>
            <th>Status</th>
            {presentCols.map(([k, label]) => <th key={k}>{label}</th>)}
            {rows.some((r) => r.affinity_pred_value != null) && <th>Affinity (log IC₅₀)</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const a = affinityLabel(r);
            return (
              <tr
                key={r.id}
                className={`clickable ${r.id === sel?.id ? "sel" : ""}`}
                onClick={() => { setSelId(r.id); setFileIdx(0); }}
              >
                <td><strong>{r.id}</strong></td>
                <td>{r.status === "ok" || !r.status ? "✓" : <span style={{ color: "var(--err)" }}>{r.status}</span>}</td>
                {presentCols.map(([k]) => <td key={k} className="metric">{fmt(r[k], k === "runtime_s" ? 1 : 3)}</td>)}
                {rows.some((x) => x.affinity_pred_value != null) && (
                  <td className="metric">{a ? `${a.log} (${a.prob != null ? `p=${fmt(a.prob, 2)}` : "—"})` : "—"}</td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>

      {sel && (
        <div className="mt16">
          <div className="flex-between mb0" style={{ marginBottom: 10 }}>
            <p className="section-title mb0">Structure — {sel.id}</p>
            {selFiles.length > 1 && (
              <select className="btn sm" value={fileIdx} onChange={(e) => setFileIdx(Number(e.target.value))} style={{ padding: "5px 9px" }}>
                {selFiles.map((f, i) => <option key={f} value={i}>{f}</option>)}
              </select>
            )}
          </div>
          {aff && (
            <div className="kv mt8" style={{ marginBottom: 12 }}>
              <span className="k">Predicted IC₅₀</span><span className="metric">≈ {aff.ic50} µM</span>
              <span className="k">log₁₀(IC₅₀)</span><span className="metric">{aff.log}</span>
              {aff.prob != null && (<><span className="k">Binding probability</span><span className="metric">{fmt(aff.prob, 2)}</span></>)}
            </div>
          )}
          {file ? (
            <StructureViewer
              url={api.structureUrl(jobId, file)}
              format={file.toLowerCase().endsWith(".pdb") ? "pdb" : "cif"}
              downloadName={file}
            />
          ) : (
            <div className="empty">No structure file (the target may have failed — check the log).</div>
          )}
        </div>
      )}
    </div>
  );
}

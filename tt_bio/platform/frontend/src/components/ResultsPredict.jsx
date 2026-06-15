import React, { useState, useMemo } from "react";
import { api } from "../api.js";
import { fmt } from "../ui.jsx";
import StructureViewer from "./StructureViewerLazy.jsx";

const COLS = [
  ["confidence_score", "Confidence", "Overall model confidence in the prediction (0–1; higher is better)."],
  ["plddt", "pLDDT", "Per-residue confidence averaged over the structure (0–100; >70 confident, >90 very high)."],
  ["complex_plddt", "Complex pLDDT", "Confidence averaged across the whole complex (0–100; higher is better)."],
  ["ptm", "pTM", "Predicted TM-score for the overall fold (0–1; >0.5 is a confident fold)."],
  ["iptm", "ipTM", "Predicted interface confidence between chains (0–1; higher is better for complexes)."],
  ["runtime_s", "Time (s)", "Wall-clock time to predict this target."],
];

const CAP = 300; // rows rendered at once; search narrows beyond this

function affinityLabel(row) {
  if (row.affinity_pred_value == null) return null;
  const log = Number(row.affinity_pred_value);
  const ic50 = Math.pow(10, log);
  return { log: log.toFixed(2), ic50: ic50 < 1 ? ic50.toFixed(3) : ic50.toFixed(1), prob: row.affinity_probability_binary };
}

export default function ResultsPredict({ jobId, results }) {
  const allRows = results.rows || [];
  const structures = results.structures || {};
  const [selId, setSelId] = useState(allRows[0]?.id);
  const [fileIdx, setFileIdx] = useState(0);
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState(-1);

  const hasAffinity = allRows.some((r) => r.affinity_pred_value != null);
  const presentCols = useMemo(() => COLS.filter(([k]) => allRows.some((r) => r[k] != null)), [allRows]);

  const filtered = useMemo(() => {
    let rows = allRows;
    if (query.trim()) {
      const q = query.toLowerCase();
      rows = rows.filter((r) => String(r.id).toLowerCase().includes(q));
    }
    if (sortKey) {
      rows = [...rows].sort((a, b) => {
        const av = Number(a[sortKey]); const bv = Number(b[sortKey]);
        if (Number.isNaN(av)) return 1;
        if (Number.isNaN(bv)) return -1;
        return (av - bv) * sortDir;
      });
    }
    return rows;
  }, [allRows, query, sortKey, sortDir]);

  const shown = filtered.slice(0, CAP);
  const sel = allRows.find((r) => r.id === selId) || allRows[0];
  const selFiles = (sel && structures[sel.id]) || [];
  const file = selFiles[fileIdx];
  const aff = sel && affinityLabel(sel);

  const toggleSort = (k) => {
    if (sortKey === k) setSortDir((d) => -d);
    else { setSortKey(k); setSortDir(-1); }
  };
  const sortArrow = (k) => (sortKey === k ? (sortDir === -1 ? " ↓" : " ↑") : "");

  // Quote any cell with a comma/quote/newline (target ids and names are
  // user-controlled, so an unescaped comma would shift every later column).
  const csvCell = (v) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const downloadCsv = () => {
    const cols = ["id", "status", ...presentCols.map(([k]) => k), ...(hasAffinity ? ["affinity_pred_value", "affinity_probability_binary"] : [])];
    const lines = [cols.map(csvCell).join(",")];
    for (const r of filtered) lines.push(cols.map((c) => csvCell(r[c])).join(","));
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "results.csv"; a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div>
      <div className="results-toolbar">
        <input type="text" placeholder={`Search ${allRows.length.toLocaleString()} targets…`} value={query} onChange={(e) => setQuery(e.target.value)} />
        <span className="muted small">
          {filtered.length.toLocaleString()} shown{filtered.length > CAP ? ` (first ${CAP} rendered — search to narrow)` : ""}
        </span>
        <div className="spacer" />
        <button className="btn sm" onClick={downloadCsv}>Download CSV</button>
        <a className="btn sm" href={api.archiveUrl(jobId)}>Download all (.zip)</a>
      </div>

      <div className="table-scroll">
        <table className="data">
          <thead>
            <tr>
              <th>Target</th>
              <th>Status</th>
              {presentCols.map(([k, label, tip]) => (
                <th key={k} className="sortable" title={tip} onClick={() => toggleSort(k)}>{label}{sortArrow(k)}</th>
              ))}
              {hasAffinity && <th className="sortable" title="Predicted binding affinity as log₁₀(IC₅₀ in µM); lower (more negative) is stronger binding. Binding probability in parentheses." onClick={() => toggleSort("affinity_pred_value")}>Affinity (log IC₅₀){sortArrow("affinity_pred_value")}</th>}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const a = affinityLabel(r);
              return (
                <tr key={r.id} className={`clickable ${r.id === sel?.id ? "sel" : ""}`} onClick={() => { setSelId(r.id); setFileIdx(0); }}>
                  <td><strong>{r.id}</strong></td>
                  <td>{r.status === "ok" || !r.status ? "✓" : <span style={{ color: "var(--err)" }}>{r.status}</span>}</td>
                  {presentCols.map(([k]) => <td key={k} className="metric">{fmt(r[k], k === "runtime_s" ? 1 : 3)}</td>)}
                  {hasAffinity && <td className="metric">{a ? `${a.log} (${a.prob != null ? `p=${fmt(a.prob, 2)}` : "—"})` : "—"}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {sel && (
        <div className="mt16">
          <div className="flex-between" style={{ marginBottom: 10 }}>
            <p className="section-title mb0">Structure — {sel.id}</p>
            {selFiles.length > 1 && (
              <select className="btn sm" value={fileIdx} onChange={(e) => setFileIdx(Number(e.target.value))} style={{ padding: "5px 9px" }}>
                {selFiles.map((f, i) => <option key={f} value={i}>{f}</option>)}
              </select>
            )}
          </div>
          {aff && (
            <div className="kv" style={{ marginBottom: 12 }}>
              <span className="k">Predicted IC₅₀</span><span className="metric">≈ {aff.ic50} µM</span>
              <span className="k">log₁₀(IC₅₀)</span><span className="metric">{aff.log}</span>
              {aff.prob != null && (<><span className="k">Binding probability</span><span className="metric">{fmt(aff.prob, 2)}</span></>)}
            </div>
          )}
          {file ? (
            <StructureViewer url={api.structureUrl(jobId, file)} format={file.toLowerCase().endsWith(".pdb") ? "pdb" : "cif"} downloadName={file} />
          ) : (
            <div className="empty">No structure file (this target may have failed — check the log).</div>
          )}
        </div>
      )}
    </div>
  );
}

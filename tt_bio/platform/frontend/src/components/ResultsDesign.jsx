import React, { useState } from "react";
import { api } from "../api.js";
import { fmt } from "../ui.jsx";
import StructureViewer from "./StructureViewerLazy.jsx";

export default function ResultsDesign({ jobId, results }) {
  const designs = results.designs || [];
  const [selRank, setSelRank] = useState(designs[0]?.final_rank);
  const [copied, setCopied] = useState(null);

  const sel = designs.find((d) => d.final_rank === selRank) || designs[0];

  const copy = (seq, rank) => {
    navigator.clipboard?.writeText(seq);
    setCopied(rank);
    setTimeout(() => setCopied(null), 1200);
  };

  if (!designs.length) return <div className="empty">No ranked designs were produced.</div>;

  return (
    <div>
      <p className="hint" style={{ marginTop: 0 }}>
        {designs.length} candidate binders, ranked by predicted interface confidence. Sequences are ready to order.
      </p>
      <p className="hint metric-legend">
        Higher <strong>iPTM</strong> and <strong>pTM</strong> (0–1) mean more confident binding and folding; lower{" "}
        <strong>PAE</strong> (Å) means a more precisely predicted interface. More interface <strong>H-bonds</strong> suggest tighter contact.
      </p>
      <table className="data">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Designed sequence</th>
            <th title="Predicted interface confidence between the designed binder and your target (0–1; higher is better, ≳0.6 is strong).">iPTM (→target)</th>
            <th title="Predicted fold confidence of the designed binder on its own (0–1; higher is better).">Design pTM</th>
            <th title="Lowest predicted aligned error across the interface, in ångström (lower is better; ≲5 Å is a tight interface).">min PAE</th>
            <th title="Hydrogen bonds at the refolded binder–target interface (more indicates richer contact).">H-bonds</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {designs.map((d) => (
            <tr
              key={d.final_rank}
              className={`clickable ${d.final_rank === sel?.final_rank ? "sel" : ""}`}
              onClick={() => setSelRank(d.final_rank)}
            >
              <td><strong>#{d.final_rank}</strong></td>
              <td className="seqcell">{(d.designed_sequence || "").slice(0, 60)}{(d.designed_sequence || "").length > 60 ? "…" : ""}</td>
              <td className="metric">{fmt(d.design_to_target_iptm, 3)}</td>
              <td className="metric">{fmt(d.design_ptm, 3)}</td>
              <td className="metric">{fmt(d.min_design_to_target_pae, 2)}</td>
              <td className="metric">{d.plip_hbonds_refolded ?? "—"}</td>
              <td>
                <button className="btn ghost sm" onClick={(e) => { e.stopPropagation(); copy(d.designed_sequence, d.final_rank); }}>
                  {copied === d.final_rank ? "Copied ✓" : "Copy"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {sel && (
        <div className="mt16">
          <p className="section-title">Design #{sel.final_rank}</p>
          <div className="field">
            <textarea className="code" rows={3} readOnly value={sel.designed_sequence || ""} />
            <div className="hint">{(sel.designed_sequence || "").length} residues · liability violations: {sel.liability_num_violations ?? "—"}</div>
          </div>
          {sel.structure ? (
            <StructureViewer
              url={api.structureUrl(jobId, sel.structure)}
              format="cif"
              downloadName={`rank${sel.final_rank}.cif`}
            />
          ) : (
            <div className="empty">No structure file for this design.</div>
          )}
        </div>
      )}
    </div>
  );
}

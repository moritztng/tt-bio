import React, { useState, useRef } from "react";
import { api } from "../api.js";
import ParamControls, { defaultsFor } from "./ParamControls.jsx";
import { parseSequences } from "../sequences.js";

let _sid = 1;
const newRow = (id = "", sequence = "") => ({ key: _sid++, id, sequence });

export default function EmbedForm({ catalog, onSubmitted, onError }) {
  const [model, setModel] = useState((catalog.embed_models[0] || {}).id || "esmc-600m");
  const [inputMode, setInputMode] = useState("compose"); // compose | bulk
  const [rows, setRows] = useState([newRow()]);
  const [name, setName] = useState("");
  const [params, setParams] = useState(() => defaultsFor(catalog.embed_params));
  const [submitting, setSubmitting] = useState(false);

  // bulk state — identical pattern to FoldForm's bulk upload
  const [bulk, setBulk] = useState([]); // [{name, sequence}]
  const [bulkText, setBulkText] = useState("");
  const fileRef = useRef(null);

  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));
  const updRow = (i, patch) => setRows((rs) => rs.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));
  const addRow = () => setRows((rs) => [...rs, newRow()]);
  const rmRow = (i) => setRows((rs) => rs.filter((_, idx) => idx !== i));
  const resetForm = () => { setRows([newRow()]); setInputMode("compose"); setName(""); };

  const ingest = (text, fname) => {
    const recs = parseSequences(text, fname);
    if (!recs.length) return onError("No sequences found in that input.");
    setBulk((prev) => {
      const seen = new Set(prev.map((r) => r.name));
      const merged = [...prev];
      for (const r of recs) {
        let nm = r.name; let n = 2;
        while (seen.has(nm)) nm = `${r.name}_${n++}`;
        seen.add(nm); merged.push({ ...r, name: nm });
      }
      return merged;
    });
  };
  const onFiles = async (fileList) => {
    for (const f of Array.from(fileList)) {
      try { ingest(await f.text(), f.name); } catch { onError(`Could not read ${f.name}`); }
    }
  };
  const addPasted = () => { if (bulkText.trim()) { ingest(bulkText, ""); setBulkText(""); } };

  const lim = catalog.limits || {};
  const maxSeqs = lim.max_embed_sequences || 50;
  const maxResidues = lim.max_embed_sequence_residues || 2000;

  const composeRecords = () => rows
    .filter((r) => r.sequence.trim())
    .map((r, i) => ({ id: (r.id || "").trim() || `seq_${i + 1}`, sequence: r.sequence.trim().replace(/\s+/g, "") }));
  const records = inputMode === "bulk"
    ? bulk.map((r) => ({ id: r.name, sequence: r.sequence }))
    : composeRecords();

  const oversized = records.some((r) => r.sequence.length > maxResidues);
  const tooMany = records.length > maxSeqs;

  const submit = async () => {
    if (!records.length) return onError("Add at least one protein sequence.");
    if (tooMany) return onError(`The free demo embeds at most ${maxSeqs} sequences per run.`);
    if (oversized) return onError(`Keep each sequence to ${maxResidues} residues or fewer.`);
    setSubmitting(true);
    try {
      const job = await api.submit({ kind: "embed", name: name.trim(), model, sequences: records, params });
      onSubmitted(job);
    } catch (e) {
      onError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <div className="panel">
        <p className="section-title">Model</p>
        <p className="section-sub">Protein-language-model embeddings (ESMC, SaProt) — no folding, no MSA.</p>
        <div className="cardgrid models">
          {catalog.embed_models.map((m) => (
            <button key={m.id} className={`selcard ${model === m.id ? "active" : ""}`}
              title={m.blurb} onClick={() => setModel(m.id)}>
              <div className="t">{m.name}</div>
              <div className="s">{m.tagline}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="flex-between">
          <p className="section-title mb0">Sequences</p>
          <div className="flex">
            <button className={`btn sm ${inputMode === "compose" ? "primary" : "ghost"}`} onClick={() => setInputMode("compose")}>Compose</button>
            <button className={`btn sm ${inputMode === "bulk" ? "primary" : "ghost"}`} onClick={() => setInputMode("bulk")}>Bulk upload</button>
          </div>
        </div>

        {inputMode === "bulk" ? (
          <BulkPanel bulk={bulk} setBulk={setBulk} bulkText={bulkText} setBulkText={setBulkText}
            addPasted={addPasted} onFiles={onFiles} fileRef={fileRef} />
        ) : (
          <div className="mt16">
            {rows.map((r, i) => (
              <div className="chain" key={r.key}>
                <div className="chain-head">
                  <label className="chain-id-tag" title="A short name for this sequence — used in the results and download files">
                    ID
                    <input className="id" type="text" value={r.id} placeholder={`seq_${i + 1}`} onChange={(e) => updRow(i, { id: e.target.value })} />
                  </label>
                  <div className="spacer" />
                  {rows.length > 1 && <button className="btn ghost sm" onClick={() => rmRow(i)}>Remove</button>}
                </div>
                <textarea className="code" rows={3} value={r.sequence} onChange={(e) => updRow(i, { sequence: e.target.value })}
                  placeholder="Paste a protein sequence — e.g. MKTAYIAKQR…" spellCheck={false} />
              </div>
            ))}
            <div className="flex">
              <button className="btn ghost sm" disabled={rows.length >= maxSeqs} onClick={addRow}>+ Sequence</button>
              <button className="btn sm" onClick={resetForm}>↺ Clear</button>
            </div>
          </div>
        )}
      </div>

      <div className="panel">
        <details className="collapse">
          <summary>Advanced settings</summary>
          <div className="mt8">
            <ParamControls params={catalog.embed_params} values={params} onChange={setParam} />
          </div>
        </details>
      </div>

      {oversized && (
        <div className="panel" style={{ borderColor: "var(--warn)", background: "rgba(201,138,0,0.06)" }}>
          <strong style={{ color: "var(--warn)" }}>⚠ Sequence too long for the free demo.</strong> Keep each sequence
          to <strong>{maxResidues} residues</strong> or fewer.
        </div>
      )}

      {tooMany && (
        <div className="panel" style={{ borderColor: "var(--warn)", background: "rgba(201,138,0,0.06)" }}>
          <strong style={{ color: "var(--warn)" }}>⚠ Too many sequences for the free demo.</strong> The demo embeds
          at most <strong>{maxSeqs}</strong> sequences per run.
        </div>
      )}

      <div className="flex-between">
        <div className="field" style={{ flex: 1, marginRight: 16, marginBottom: 0 }}>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Job name (optional)" />
        </div>
        <button className="btn primary" disabled={submitting || oversized || tooMany || !records.length} onClick={submit}>
          {submitting ? "Submitting…" : records.length > 1 ? `Embed ${records.length} sequences →` : "Embed sequence →"}
        </button>
      </div>
    </>
  );
}

function BulkPanel({ bulk, setBulk, bulkText, setBulkText, addPasted, onFiles, fileRef }) {
  const [drag, setDrag] = useState(false);
  return (
    <div className="mt16">
      <div
        className={`dropzone ${drag ? "drag" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); onFiles(e.dataTransfer.files); }}
        onClick={() => fileRef.current?.click()}
      >
        <input ref={fileRef} type="file" multiple accept=".fasta,.fa,.fas,.faa,.txt,.csv" style={{ display: "none" }}
          onChange={(e) => { onFiles(e.target.files); e.target.value = ""; }} />
        <strong>Drop FASTA / CSV files here</strong> or click to browse
        <div className="hint">Multi-record FASTA or CSV (<code>name,sequence</code>) — one embedding per sequence.</div>
      </div>

      <details className="collapse mt8">
        <summary>…or paste sequences</summary>
        <div className="mt8">
          <textarea className="code" rows={6} value={bulkText} onChange={(e) => setBulkText(e.target.value)}
            placeholder={">seq_1\nMVTPEG...\n>seq_2\nQLEDSE..."} spellCheck={false} />
          <button className="btn sm mt8" onClick={addPasted}>Add to batch</button>
        </div>
      </details>

      {bulk.length > 0 && (
        <div className="panel mt16" style={{ marginBottom: 0, background: "var(--surface-2)" }}>
          <div className="flex-between">
            <strong>{bulk.length.toLocaleString()} sequence{bulk.length === 1 ? "" : "s"} queued</strong>
            <button className="btn ghost sm" onClick={() => setBulk([])}>Clear</button>
          </div>
          <div className="hint mt8">
            {bulk.slice(0, 6).map((r) => r.name).join(", ")}{bulk.length > 6 ? `, +${bulk.length - 6} more` : ""}
          </div>
        </div>
      )}
    </div>
  );
}

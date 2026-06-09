import React, { useMemo, useState, useRef } from "react";
import { api } from "../api.js";
import ParamControls, { defaultsFor } from "./ParamControls.jsx";
import { parseSequences, recordToTarget } from "../sequences.js";

let _tid = 1;
const newTarget = (content = "", name = "") => ({ key: _tid++, name, content });

function genYaml(chains, affinityBinder) {
  const lines = ["version: 1", "sequences:"];
  for (const c of chains) {
    const id = c.id.trim() || "A";
    if (c.type === "ligand") {
      lines.push("  - ligand:");
      lines.push(`      id: ${id}`);
      if (c.ligandMode === "ccd") lines.push(`      ccd: ${c.ccd.trim()}`);
      else lines.push(`      smiles: '${c.smiles.trim()}'`);
    } else {
      lines.push(`  - ${c.type}:`);
      lines.push(`      id: ${id}`);
      lines.push(`      sequence: ${c.sequence.trim().replace(/\s+/g, "")}`);
    }
  }
  if (affinityBinder) {
    lines.push("properties:", "  - affinity:", `      binder: ${affinityBinder}`);
  }
  return lines.join("\n") + "\n";
}

export default function FoldForm({ catalog, onSubmitted, onError }) {
  const [model, setModel] = useState("boltz2");
  const [inputMode, setInputMode] = useState("compose"); // compose | bulk
  const [format, setFormat] = useState("yaml");
  const [targets, setTargets] = useState([newTarget()]);
  const [active, setActive] = useState(0);
  const [name, setName] = useState("");
  const [params, setParams] = useState(() => ({ ...defaultsFor(catalog.predict_params), use_msa_server: true }));
  const [showBuilder, setShowBuilder] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  // bulk state
  const [bulk, setBulk] = useState([]); // [{name, sequence}]
  const [bulkText, setBulkText] = useState("");
  const fileRef = useRef(null);

  // quick-builder state (compose mode)
  const [chains, setChains] = useState([{ type: "protein", id: "A", sequence: "", smiles: "", ccd: "", ligandMode: "smiles" }]);
  const [affinity, setAffinity] = useState(false);
  const [affinityBinder, setAffinityBinder] = useState("");

  const modelInfo = useMemo(() => catalog.models.find((m) => m.id === model), [catalog, model]);
  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));
  const updTarget = (i, patch) => setTargets((ts) => ts.map((t, idx) => (idx === i ? { ...t, ...patch } : t)));

  const onModelChange = (m) => {
    setModel(m);
    const info = catalog.models.find((x) => x.id === m);
    setParam("use_msa_server", !!info?.needs_msa);
    if (m !== "boltz2") setFormat("yaml");
  };

  const ligandChains = chains.filter((c) => c.type === "ligand");
  const applyBuilder = () => {
    updTarget(active, { content: genYaml(chains, affinity ? (affinityBinder || ligandChains[0]?.id) : null) });
    setFormat("yaml");
  };
  const addChain = (type) => setChains((cs) => [...cs, { type, id: String.fromCharCode(65 + cs.length), sequence: "", smiles: "", ccd: "", ligandMode: "smiles" }]);
  const updChain = (i, patch) => setChains((cs) => cs.map((c, idx) => (idx === i ? { ...c, ...patch } : c)));
  const rmChain = (i) => setChains((cs) => cs.filter((_, idx) => idx !== i));

  const loadExample = (id) => {
    const ex = catalog.examples.find((e) => e.id === id);
    if (!ex) return;
    // Load the input only — keep the model the user picked. (Examples used to
    // silently switch the model card, which was confusing.)
    setFormat(ex.format || "yaml");
    setInputMode("compose");
    setTargets((ts) => {
      const next = [...ts.filter((t) => t.content.trim()), newTarget(ex.content, ex.name)];
      setActive(next.length - 1);
      return next;
    });
  };

  // ---- bulk handlers ----
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

  const submit = async () => {
    setSubmitting(true);
    try {
      let payloadTargets; let inputFormat;
      if (inputMode === "bulk") {
        if (!bulk.length) { onError("Upload or paste some sequences first."); setSubmitting(false); return; }
        inputFormat = model.startsWith("esmfold") ? "fasta" : "yaml";
        payloadTargets = bulk.map((r) => recordToTarget(r, model));
      } else {
        const clean = targets.filter((t) => t.content.trim());
        if (!clean.length) { onError("Add at least one input."); setSubmitting(false); return; }
        inputFormat = format;
        payloadTargets = clean.map((t, i) => ({ name: t.name.trim() || `target_${i + 1}`, content: t.content }));
      }
      const job = await api.submit({
        kind: "predict", name: name.trim(), model, input_format: inputFormat,
        targets: payloadTargets, params,
      });
      onSubmitted(job);
    } catch (e) {
      onError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const predictExamples = catalog.examples.filter((e) => e.kind === "predict");
  const bigBatch = inputMode === "bulk" && bulk.length > 200 && model === "boltz2" && params.use_msa_server;

  return (
    <>
      <div className="panel">
        <p className="section-title">Model</p>
        <div className="cardgrid">
          {catalog.models.map((m) => (
            <button key={m.id} className={`selcard ${model === m.id ? "active" : ""}`} onClick={() => onModelChange(m.id)}>
              <div className="t">{m.name}</div>
              <div className="s">{m.tagline}</div>
            </button>
          ))}
        </div>
        {modelInfo && <div className="hint mt8">{modelInfo.blurb}</div>}
      </div>

      <div className="panel">
        <div className="flex-between">
          <div className="flex">
            <button className={`btn sm ${inputMode === "compose" ? "primary" : "ghost"}`} onClick={() => setInputMode("compose")}>Compose</button>
            <button className={`btn sm ${inputMode === "bulk" ? "primary" : "ghost"}`} onClick={() => setInputMode("bulk")}>Bulk upload</button>
          </div>
          {inputMode === "compose" && (
            <div className="flex">
              <select value={format} onChange={(e) => setFormat(e.target.value)} className="btn sm" style={{ padding: "6px 10px" }}>
                <option value="yaml">YAML</option>
                <option value="fasta">FASTA</option>
              </select>
              <select className="btn sm" style={{ padding: "6px 10px" }} value="" onChange={(e) => loadExample(e.target.value)}>
                <option value="">Load example…</option>
                {predictExamples.map((e) => <option key={e.id} value={e.id}>{e.name}</option>)}
              </select>
            </div>
          )}
        </div>

        {inputMode === "bulk" ? (
          <BulkPanel
            bulk={bulk} setBulk={setBulk} bulkText={bulkText} setBulkText={setBulkText}
            addPasted={addPasted} onFiles={onFiles} fileRef={fileRef} bigBatch={bigBatch} model={model}
          />
        ) : (
          <ComposePanel
            targets={targets} setTargets={setTargets} active={active} setActive={setActive}
            updTarget={updTarget} format={format} newTarget={newTarget}
            showBuilder={showBuilder} setShowBuilder={setShowBuilder}
            chains={chains} addChain={addChain} updChain={updChain} rmChain={rmChain}
            ligandChains={ligandChains} modelInfo={modelInfo} affinity={affinity} setAffinity={setAffinity}
            affinityBinder={affinityBinder} setAffinityBinder={setAffinityBinder} applyBuilder={applyBuilder}
          />
        )}
      </div>

      <div className="panel">
        <details className="collapse">
          <summary>Advanced settings</summary>
          <div className="mt8">
            <ParamControls params={catalog.predict_params} values={params} onChange={setParam} />
          </div>
        </details>
      </div>

      <div className="flex-between">
        <div className="field" style={{ flex: 1, marginRight: 16, marginBottom: 0 }}>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Job name (optional)" />
        </div>
        <button className="btn primary" disabled={submitting} onClick={submit}>
          {submitting ? "Submitting…" : inputMode === "bulk" && bulk.length ? `Run ${bulk.length} predictions →` : "Run prediction →"}
        </button>
      </div>
    </>
  );
}

function BulkPanel({ bulk, setBulk, bulkText, setBulkText, addPasted, onFiles, fileRef, bigBatch, model }) {
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
        <div className="hint">Multi-record FASTA or CSV (<code>name,sequence</code>). Upload thousands at once — one structure per sequence.</div>
      </div>

      <details className="collapse mt8">
        <summary>…or paste sequences</summary>
        <div className="mt8">
          <textarea className="code" rows={6} value={bulkText} onChange={(e) => setBulkText(e.target.value)}
            placeholder={">protein_1|protein\nMVTPEG...\n>protein_2|protein\nQLEDSE..."} spellCheck={false} />
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
          {bigBatch && (
            <div className="hint mt8" style={{ color: "var(--warn)" }}>
              That's a large batch with MSA generation on — it may be slow / rate-limited. For very large
              runs, ESMFold-2 Fast (no MSA) is far faster, or turn off “Generate MSA” in Advanced settings.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ComposePanel(p) {
  const { targets, setTargets, active, setActive, updTarget, format, newTarget } = p;
  return (
    <>
      <div className="flex mt16" style={{ flexWrap: "wrap" }}>
        {targets.map((t, i) => (
          <button key={t.key} className={`btn sm ${active === i ? "primary" : "ghost"}`} onClick={() => setActive(i)}>
            {t.name.trim() || `Target ${i + 1}`}
            {targets.length > 1 && (
              <span onClick={(e) => { e.stopPropagation(); setTargets((ts) => ts.filter((_, x) => x !== i)); setActive(0); }} style={{ marginLeft: 4, opacity: 0.7 }}>✕</span>
            )}
          </button>
        ))}
        <button className="btn sm ghost" onClick={() => { setTargets((ts) => [...ts, newTarget()]); setActive(targets.length); }}>+ Add target</button>
      </div>

      <div className="field mt16">
        <label>Target name</label>
        <input type="text" value={targets[active]?.name || ""} placeholder={`target_${active + 1}`} onChange={(e) => updTarget(active, { name: e.target.value })} />
      </div>

      <div className="field">
        <label>{format === "fasta" ? "FASTA" : "Input (YAML)"}</label>
        <textarea className="code" rows={10} spellCheck={false} value={targets[active]?.content || ""}
          placeholder={format === "fasta" ? ">A|protein\nMVTPEG..." : "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MVTPEG..."}
          onChange={(e) => updTarget(active, { content: e.target.value })} />
        <div className="hint">Full tt-bio input schema — chains, ligands, constraints, templates, modifications, custom MSA. Use the quick builder or write YAML directly.</div>
      </div>

      <details className="collapse" open={p.showBuilder} onToggle={(e) => p.setShowBuilder(e.target.open)}>
        <summary>Quick builder</summary>
        <div className="mt8">
          {p.chains.map((c, i) => (
            <div className="chain" key={i}>
              <div className="chain-head">
                <select value={c.type} onChange={(e) => p.updChain(i, { type: e.target.value })}>
                  <option value="protein">Protein</option>
                  <option value="dna">DNA</option>
                  <option value="rna">RNA</option>
                  <option value="ligand">Ligand</option>
                </select>
                <input className="id" type="text" value={c.id} onChange={(e) => p.updChain(i, { id: e.target.value })} placeholder="id" />
                <div className="spacer" />
                {p.chains.length > 1 && <button className="btn ghost sm" onClick={() => p.rmChain(i)}>Remove</button>}
              </div>
              {c.type === "ligand" ? (
                <div className="row">
                  <select value={c.ligandMode} onChange={(e) => p.updChain(i, { ligandMode: e.target.value })} style={{ maxWidth: 110 }}>
                    <option value="smiles">SMILES</option>
                    <option value="ccd">CCD</option>
                  </select>
                  {c.ligandMode === "ccd"
                    ? <input type="text" value={c.ccd} onChange={(e) => p.updChain(i, { ccd: e.target.value })} placeholder="e.g. ATP" />
                    : <input type="text" value={c.smiles} onChange={(e) => p.updChain(i, { smiles: e.target.value })} placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O" />}
                </div>
              ) : (
                <textarea className="code" rows={2} value={c.sequence} onChange={(e) => p.updChain(i, { sequence: e.target.value })} placeholder="sequence" spellCheck={false} />
              )}
            </div>
          ))}
          <div className="flex">
            <button className="btn sm" onClick={() => p.addChain("protein")}>+ Protein</button>
            <button className="btn sm" onClick={() => p.addChain("ligand")}>+ Ligand</button>
            <button className="btn sm" onClick={() => p.addChain("dna")}>+ DNA</button>
            <button className="btn sm" onClick={() => p.addChain("rna")}>+ RNA</button>
          </div>
          {p.modelInfo?.supports_affinity && p.ligandChains.length > 0 && (
            <div className="checkline mt16">
              <input type="checkbox" id="aff" checked={p.affinity} onChange={(e) => p.setAffinity(e.target.checked)} />
              <div className="cl-body">
                <label className="cl-label" htmlFor="aff">Predict binding affinity</label>
                {p.affinity && (
                  <select className="mt8" value={p.affinityBinder || p.ligandChains[0]?.id} onChange={(e) => p.setAffinityBinder(e.target.value)} style={{ maxWidth: 200 }}>
                    {p.ligandChains.map((c) => <option key={c.id} value={c.id}>binder: {c.id}</option>)}
                  </select>
                )}
              </div>
            </div>
          )}
          <button className="btn primary sm mt16" onClick={p.applyBuilder}>Apply to input ↑</button>
        </div>
      </details>
    </>
  );
}

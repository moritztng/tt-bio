import React, { useMemo, useState } from "react";
import { api } from "../api.js";
import ParamControls, { defaultsFor } from "./ParamControls.jsx";

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
    lines.push("properties:");
    lines.push("  - affinity:");
    lines.push(`      binder: ${affinityBinder}`);
  }
  return lines.join("\n") + "\n";
}

export default function FoldForm({ catalog, onSubmitted, onError }) {
  const [model, setModel] = useState("boltz2");
  const [format, setFormat] = useState("yaml");
  const [targets, setTargets] = useState([newTarget()]);
  const [active, setActive] = useState(0);
  const [name, setName] = useState("");
  const [params, setParams] = useState(() => ({ ...defaultsFor(catalog.predict_params), use_msa_server: true }));
  const [showBuilder, setShowBuilder] = useState(true);
  const [submitting, setSubmitting] = useState(false);

  // Quick-builder state (applies to the active target).
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
    if (m !== "boltz2") setFormat("yaml"); // builder still emits yaml; fasta is a manual choice
  };

  const ligandChains = chains.filter((c) => c.type === "ligand");

  const applyBuilder = () => {
    const yaml = genYaml(chains, affinity ? (affinityBinder || ligandChains[0]?.id) : null);
    updTarget(active, { content: yaml });
    setFormat("yaml");
  };

  const addChain = (type) =>
    setChains((cs) => [...cs, { type, id: nextId(cs), sequence: "", smiles: "", ccd: "", ligandMode: "smiles" }]);
  const nextId = (cs) => String.fromCharCode(65 + cs.length); // A, B, C…
  const updChain = (i, patch) => setChains((cs) => cs.map((c, idx) => (idx === i ? { ...c, ...patch } : c)));
  const rmChain = (i) => setChains((cs) => cs.filter((_, idx) => idx !== i));

  const loadExample = (id) => {
    const ex = catalog.examples.find((e) => e.id === id);
    if (!ex) return;
    onModelChange(ex.model || "boltz2");
    setFormat(ex.format || "yaml");
    setTargets((ts) => {
      const next = [...ts.filter((t) => t.content.trim()), newTarget(ex.content, ex.name)];
      setActive(next.length - 1);
      return next.length ? next : [newTarget(ex.content, ex.name)];
    });
  };

  const submit = async () => {
    const clean = targets.filter((t) => t.content.trim());
    if (!clean.length) return onError("Add at least one input.");
    setSubmitting(true);
    try {
      const job = await api.submit({
        kind: "predict",
        name: name.trim(),
        model,
        input_format: format,
        targets: clean.map((t, i) => ({ name: t.name.trim() || `target_${i + 1}`, content: t.content })),
        params,
      });
      onSubmitted(job);
    } catch (e) {
      onError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const predictExamples = catalog.examples.filter((e) => e.kind === "predict");

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
          <p className="section-title mb0">Input</p>
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
        </div>

        {/* target tabs (batch) */}
        <div className="flex mt8" style={{ flexWrap: "wrap" }}>
          {targets.map((t, i) => (
            <button key={t.key} className={`btn sm ${active === i ? "primary" : "ghost"}`} onClick={() => setActive(i)}>
              {t.name.trim() || `Target ${i + 1}`}
              {targets.length > 1 && (
                <span
                  onClick={(e) => { e.stopPropagation(); setTargets((ts) => ts.filter((_, x) => x !== i)); setActive(0); }}
                  style={{ marginLeft: 4, opacity: 0.7 }}
                >✕</span>
              )}
            </button>
          ))}
          <button className="btn sm ghost" onClick={() => { setTargets((ts) => [...ts, newTarget()]); setActive(targets.length); }}>
            + Add target
          </button>
        </div>

        <div className="field mt16">
          <label>Target name</label>
          <input
            type="text"
            value={targets[active]?.name || ""}
            placeholder={`target_${active + 1}`}
            onChange={(e) => updTarget(active, { name: e.target.value })}
          />
        </div>

        <div className="field">
          <label>{format === "fasta" ? "FASTA" : "Input (YAML)"}</label>
          <textarea
            className="code"
            rows={10}
            spellCheck={false}
            value={targets[active]?.content || ""}
            placeholder={format === "fasta"
              ? ">my_protein|protein\nMVTPEG..."
              : "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MVTPEG..."}
            onChange={(e) => updTarget(active, { content: e.target.value })}
          />
          <div className="hint">
            Full tt-bio input schema is supported — chains, ligands, constraints, templates,
            modifications, custom MSA. Use the quick builder below or write YAML directly.
          </div>
        </div>

        {/* Quick builder */}
        <details className="collapse" open={showBuilder} onToggle={(e) => setShowBuilder(e.target.open)}>
          <summary>Quick builder</summary>
          <div className="mt8">
            {chains.map((c, i) => (
              <div className="chain" key={i}>
                <div className="chain-head">
                  <select value={c.type} onChange={(e) => updChain(i, { type: e.target.value })}>
                    <option value="protein">Protein</option>
                    <option value="dna">DNA</option>
                    <option value="rna">RNA</option>
                    <option value="ligand">Ligand</option>
                  </select>
                  <input className="id" type="text" value={c.id} onChange={(e) => updChain(i, { id: e.target.value })} placeholder="id" />
                  <div className="spacer" />
                  {chains.length > 1 && <button className="btn ghost sm" onClick={() => rmChain(i)}>Remove</button>}
                </div>
                {c.type === "ligand" ? (
                  <div className="row">
                    <select value={c.ligandMode} onChange={(e) => updChain(i, { ligandMode: e.target.value })} style={{ maxWidth: 110 }}>
                      <option value="smiles">SMILES</option>
                      <option value="ccd">CCD</option>
                    </select>
                    {c.ligandMode === "ccd"
                      ? <input type="text" value={c.ccd} onChange={(e) => updChain(i, { ccd: e.target.value })} placeholder="e.g. ATP" />
                      : <input type="text" value={c.smiles} onChange={(e) => updChain(i, { smiles: e.target.value })} placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O" />}
                  </div>
                ) : (
                  <textarea className="code" rows={2} value={c.sequence} onChange={(e) => updChain(i, { sequence: e.target.value })} placeholder="sequence" spellCheck={false} />
                )}
              </div>
            ))}
            <div className="flex">
              <button className="btn sm" onClick={() => addChain("protein")}>+ Protein</button>
              <button className="btn sm" onClick={() => addChain("ligand")}>+ Ligand</button>
              <button className="btn sm" onClick={() => addChain("dna")}>+ DNA</button>
              <button className="btn sm" onClick={() => addChain("rna")}>+ RNA</button>
            </div>
            {modelInfo?.supports_affinity && ligandChains.length > 0 && (
              <div className="checkline mt16">
                <input type="checkbox" id="aff" checked={affinity} onChange={(e) => setAffinity(e.target.checked)} />
                <div className="cl-body">
                  <label className="cl-label" htmlFor="aff">Predict binding affinity</label>
                  {affinity && (
                    <select className="mt8" value={affinityBinder || ligandChains[0]?.id} onChange={(e) => setAffinityBinder(e.target.value)} style={{ maxWidth: 200 }}>
                      {ligandChains.map((c) => <option key={c.id} value={c.id}>binder: {c.id}</option>)}
                    </select>
                  )}
                </div>
              </div>
            )}
            <button className="btn primary sm mt16" onClick={applyBuilder}>Apply to input ↑</button>
          </div>
        </details>
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
          {submitting ? "Submitting…" : "Run prediction →"}
        </button>
      </div>
    </>
  );
}

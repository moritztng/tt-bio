import React, { useMemo, useState, useRef } from "react";
import { api } from "../api.js";
import ParamControls, { defaultsFor } from "./ParamControls.jsx";
import { parseSequences, recordToTarget } from "../sequences.js";

let _tid = 1;
const newTarget = (content = "", name = "") => ({ key: _tid++, name, content });

// Type-aware placeholder for the sequence box, so it matches the chain type
// (protein / DNA / RNA) instead of always saying "protein".
const SEQ_HINT = {
  protein: { noun: "protein", ex: "MKTAYIAKQR…" },
  dna: { noun: "DNA", ex: "ATGCATGC…" },
  rna: { noun: "RNA", ex: "AUGCAUGC…" },
};
const seqPlaceholder = (type, single) => {
  const h = SEQ_HINT[type] || { noun: "", ex: "" };
  return single ? `Paste a ${h.noun} sequence — e.g. ${h.ex}` : `${h.noun} sequence`;
};

// Which capabilities an input actually exercises — so we can refuse it on a
// model that lacks them (e.g. ESMFold silently drops ligands/nucleic/affinity).
const CAP_LABEL = {
  ligands: "ligands", nucleic: "nucleic acids (DNA/RNA)",
  affinity: "binding affinity", constraints: "constraints",
};
function inputCaps(content) {
  const caps = new Set();
  if (/(^|\n)\s*-?\s*ligand\s*:/i.test(content)) caps.add("ligands");
  if (/(^|\n)\s*-?\s*(dna|rna)\s*:/i.test(content)) caps.add("nucleic");
  if (/affinity\s*:/i.test(content)) caps.add("affinity");
  if (/(^|\n)\s*constraints\s*:/i.test(content)) caps.add("constraints");
  return caps;
}

// A ligand only makes sense bound to something — folding one on its own isn't a
// structure-prediction task. Protein / DNA / RNA on their own are fine.
function isLigandOnly(content) {
  const hasLigand = /(^|\n)\s*-?\s*ligand\s*:/i.test(content);
  const hasPolymer = /(^|\n)\s*-?\s*(protein|dna|rna)\s*:/i.test(content);
  return hasLigand && !hasPolymer;
}

// Single-quote a value as a YAML scalar (escaping ' as '') so user-entered
// fields with ':', '#', '[' etc. can't break or inject into the generated YAML.
const yq = (v) => `'${String(v).replace(/'/g, "''")}'`;

function genYaml(chains, affinityBinder) {
  const lines = ["version: 1", "sequences:"];
  for (const c of chains) {
    const id = c.id.trim() || "A";
    if (c.type === "ligand") {
      lines.push("  - ligand:");
      lines.push(`      id: ${yq(id)}`);
      if (c.ligandMode === "ccd") lines.push(`      ccd: ${yq(c.ccd.trim())}`);
      else lines.push(`      smiles: ${yq(c.smiles.trim())}`);
    } else {
      lines.push(`  - ${c.type}:`);
      lines.push(`      id: ${yq(id)}`);
      lines.push(`      sequence: ${yq(c.sequence.trim().replace(/\s+/g, ""))}`);
    }
  }
  if (affinityBinder) {
    lines.push("properties:", "  - affinity:", `      binder: ${yq(affinityBinder)}`);
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
  const [composeMode, setComposeMode] = useState("form"); // form (simple, default) | yaml (advanced)
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
  const caps = useMemo(() => new Set(modelInfo?.caps || []), [modelInfo]);
  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));
  const updTarget = (i, patch) => setTargets((ts) => ts.map((t, idx) => (idx === i ? { ...t, ...patch } : t)));

  const onModelChange = (m) => {
    setModel(m);
    const info = catalog.models.find((x) => x.id === m);
    setParam("use_msa_server", !!info?.needs_msa);
    if (m !== "boltz2") setFormat("yaml");
  };

  const ligandChains = chains.filter((c) => c.type === "ligand");
  // In the simple "form" mode the builder *is* the input — it generates the YAML
  // on submit, so beginners never see raw YAML.
  const formContent = () => genYaml(chains, affinity ? (affinityBinder || ligandChains[0]?.id) : null);
  const formEmpty = () => chains.every((c) => !((c.type === "ligand" ? (c.smiles || c.ccd) : c.sequence) || "").trim());
  const editAsYaml = () => { updTarget(active, { content: formContent() }); setFormat("yaml"); setComposeMode("yaml"); };
  const addChain = (type) => setChains((cs) => [...cs, { type, id: String.fromCharCode(65 + cs.length), sequence: "", smiles: "", ccd: "", ligandMode: "smiles" }]);
  const updChain = (i, patch) => setChains((cs) => cs.map((c, idx) => (idx === i ? { ...c, ...patch } : c)));
  const rmChain = (i) => setChains((cs) => cs.filter((_, idx) => idx !== i));

  const loadExample = (id) => {
    const ex = catalog.examples.find((e) => e.id === id);
    if (!ex) return;
    if ((ex.requires || []).some((c) => !caps.has(c))) {
      onError(`The "${ex.name}" example needs features ${modelInfo.name} doesn't support. Switch to Boltz-2 to use it.`);
      return;
    }
    // Load the input only — keep the model the user picked. (Examples used to
    // silently switch the model card, which was confusing.) Examples are raw
    // templates, so they open the YAML editor.
    setFormat(ex.format || "yaml");
    setInputMode("compose");
    setComposeMode("yaml");
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
      } else if (composeMode === "form") {
        if (formEmpty()) { onError("Enter a sequence (or a ligand)."); setSubmitting(false); return; }
        inputFormat = "yaml";
        payloadTargets = [{ name: name.trim() || "target", content: formContent() }];
      } else {
        const clean = targets.filter((t) => t.content.trim());
        if (!clean.length) { onError("Add at least one input."); setSubmitting(false); return; }
        inputFormat = format;
        payloadTargets = clean.map((t, i) => ({ name: t.name.trim() || `target_${i + 1}`, content: t.content }));
      }
      const missing = [...new Set(payloadTargets.flatMap((t) => [...inputCaps(t.content)]))].filter((c) => !caps.has(c));
      if (missing.length) {
        onError(`${modelInfo.name} doesn't support ${missing.map((c) => CAP_LABEL[c]).join(", ")} — it would silently ignore that. Switch to a model that does (e.g. Boltz-2).`);
        setSubmitting(false); return;
      }
      if (payloadTargets.some((t) => isLigandOnly(t.content))) {
        onError("A ligand can't be folded on its own — add a protein, DNA, or RNA chain for it to bind.");
        setSubmitting(false); return;
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
  const exampleOk = (e) => (e.requires || []).every((c) => caps.has(c));
  const composeContent = inputMode !== "bulk"
    ? (composeMode === "form" ? [formContent()] : targets.map((t) => t.content))
    : [];
  const missingCaps = [...new Set(composeContent.flatMap((c) => (c && c.trim() ? [...inputCaps(c)] : [])))]
    .filter((c) => !caps.has(c));
  const esmMismatch = missingCaps.length > 0;
  const ligandOnly = composeContent.some((c) => c && c.trim() && isLigandOnly(c));

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

      {inputMode === "compose" && (
        <div className="examples-row">
          <span className="examples-label">Start from an example:</span>
          {predictExamples.map((e) => (
            <button key={e.id} className="chip" disabled={!exampleOk(e)}
              title={exampleOk(e) ? "Load this example" : "Needs Boltz-2"}
              onClick={() => loadExample(e.id)}>
              {e.name}
            </button>
          ))}
        </div>
      )}

      <div className="panel">
        <div className="flex-between">
          <div className="flex">
            <button className={`btn sm ${inputMode === "compose" ? "primary" : "ghost"}`} onClick={() => setInputMode("compose")}>Compose</button>
            <button className={`btn sm ${inputMode === "bulk" ? "primary" : "ghost"}`} onClick={() => setInputMode("bulk")}>Bulk upload</button>
          </div>
          {inputMode === "compose" && (
            <div className="flex">
              {composeMode === "yaml" && (
                <select value={format} onChange={(e) => setFormat(e.target.value)} className="btn sm" style={{ padding: "6px 10px" }}>
                  <option value="yaml">YAML</option>
                  <option value="fasta">FASTA</option>
                </select>
              )}
              <button className="btn ghost sm" onClick={() => (composeMode === "form" ? editAsYaml() : setComposeMode("form"))}>
                {composeMode === "form" ? "Edit as YAML ⟩" : "⟨ Simple form"}
              </button>
            </div>
          )}
        </div>

        {inputMode === "bulk" ? (
          <BulkPanel
            bulk={bulk} setBulk={setBulk} bulkText={bulkText} setBulkText={setBulkText}
            addPasted={addPasted} onFiles={onFiles} fileRef={fileRef} bigBatch={bigBatch} model={model}
          />
        ) : composeMode === "form" ? (
          <FormBuilder
            chains={chains} addChain={addChain} updChain={updChain} rmChain={rmChain}
            caps={caps} modelInfo={modelInfo} ligandChains={ligandChains}
            affinity={affinity} setAffinity={setAffinity}
            affinityBinder={affinityBinder} setAffinityBinder={setAffinityBinder}
          />
        ) : (
          <YamlEditor
            targets={targets} setTargets={setTargets} active={active} setActive={setActive}
            updTarget={updTarget} format={format} newTarget={newTarget}
          />
        )}
      </div>

      <div className="panel">
        <details className="collapse">
          <summary>Advanced settings</summary>
          <div className="mt8">
            <ParamControls params={catalog.predict_params} values={params} onChange={setParam} caps={caps} modelName={modelInfo?.name} />
          </div>
        </details>
      </div>

      {esmMismatch && (
        <div className="panel" style={{ borderColor: "var(--warn)", background: "rgba(201,138,0,0.06)" }}>
          <strong style={{ color: "var(--warn)" }}>⚠ Model / input mismatch.</strong> This input uses{" "}
          <strong>{missingCaps.map((c) => CAP_LABEL[c]).join(", ")}</strong>, which <strong>{modelInfo?.name}</strong> doesn't
          support — it would silently ignore them. Switch to <strong>Boltz-2</strong> for ligands, nucleic acids, binding affinity, or constraints.
        </div>
      )}

      {ligandOnly && (
        <div className="panel" style={{ borderColor: "var(--warn)", background: "rgba(201,138,0,0.06)" }}>
          <strong style={{ color: "var(--warn)" }}>⚠ Ligand needs a target.</strong> A ligand can't be folded on its own —
          add a <strong>protein, DNA, or RNA</strong> chain for it to bind.
        </div>
      )}

      <div className="flex-between">
        <div className="field" style={{ flex: 1, marginRight: 16, marginBottom: 0 }}>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Job name (optional)" />
        </div>
        <button className="btn primary" disabled={submitting || esmMismatch || ligandOnly} onClick={submit}>
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

// Advanced editor: raw YAML/FASTA with multi-target batch tabs.
function YamlEditor({ targets, setTargets, active, setActive, updTarget, format, newTarget }) {
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
        <textarea className="code" rows={12} spellCheck={false} value={targets[active]?.content || ""}
          placeholder={format === "fasta" ? ">A|protein\nMVTPEG..." : "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: MVTPEG..."}
          onChange={(e) => updTarget(active, { content: e.target.value })} />
        <div className="hint">Full tt-bio input schema — chains, ligands, constraints, templates, modifications, custom MSA.</div>
      </div>
    </>
  );
}

// Simple, beginner-first builder (the default). Starts as one protein-sequence
// box; ligands / extra chains / affinity appear only as you add them.
function FormBuilder(p) {
  const single = p.chains.length === 1;
  return (
    <div className="mt16">
      {p.chains.map((c, i) => (
        <div className="chain" key={i}>
          <div className="chain-head">
            <select value={c.type} onChange={(e) => p.updChain(i, { type: e.target.value })}>
              <option value="protein">Protein</option>
              <option value="dna" disabled={!p.caps.has("nucleic")}>DNA</option>
              <option value="rna" disabled={!p.caps.has("nucleic")}>RNA</option>
              <option value="ligand" disabled={!p.caps.has("ligands")}>Ligand</option>
            </select>
            {!single && <input className="id" type="text" value={c.id} onChange={(e) => p.updChain(i, { id: e.target.value })} placeholder="id" />}
            <div className="spacer" />
            {!single && <button className="btn ghost sm" onClick={() => p.rmChain(i)}>Remove</button>}
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
            <textarea className="code" rows={3} value={c.sequence} onChange={(e) => p.updChain(i, { sequence: e.target.value })}
              placeholder={seqPlaceholder(c.type, single)} spellCheck={false} />
          )}
        </div>
      ))}
      <div className="flex" style={{ flexWrap: "wrap" }}>
        <button className="btn ghost sm" onClick={() => p.addChain("protein")}>+ Protein</button>
        <button className="btn ghost sm" disabled={!p.caps.has("ligands")} title={p.caps.has("ligands") ? "" : `${p.modelInfo?.name} doesn't support ligands`} onClick={() => p.addChain("ligand")}>+ Ligand</button>
        <button className="btn ghost sm" disabled={!p.caps.has("nucleic")} title={p.caps.has("nucleic") ? "" : `${p.modelInfo?.name} doesn't support nucleic acids`} onClick={() => p.addChain("dna")}>+ DNA</button>
        <button className="btn ghost sm" disabled={!p.caps.has("nucleic")} title={p.caps.has("nucleic") ? "" : `${p.modelInfo?.name} doesn't support nucleic acids`} onClick={() => p.addChain("rna")}>+ RNA</button>
      </div>
      {p.caps.has("affinity") && p.ligandChains.length > 0 && (
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
    </div>
  );
}

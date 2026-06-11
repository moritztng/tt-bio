import React, { useMemo, useState } from "react";
import { api } from "../api.js";
import ParamControls, { defaultsFor } from "./ParamControls.jsx";

const DEFAULT_LEN = {
  "protein-anything": "80..120",
  "peptide-anything": "12..25",
  "nanobody-anything": "110..130",
  "antibody-anything": "110..130",
  "protein-small_molecule": "80..120",
  "protein-redesign": "80..120",
};

// 'protein-small_molecule' designs a binder against a small molecule, so its
// target is a ligand; every other protocol targets a protein.
const isLigandProtocol = (p) => p === "protein-small_molecule";

// Single-quote a value as a YAML scalar (escaping ' as '') so user-entered
// fields with ':', '#', '[' etc. can't break or inject into the generated spec.
const yq = (v) => `'${String(v).replace(/'/g, "''")}'`;

// Build a BoltzGen entities spec from the simple-form fields. The binder is a
// length range to design; the target is either a protein sequence or a ligand.
function genSpec({ isLigand, target, ligand, ligandMode, targetId, binderId, lengthRange }) {
  const lines = [
    "entities:",
    "  - protein:",
    `      id: ${yq(binderId || "B")}`,
    `      sequence: ${yq(lengthRange || "80..120")}`,
  ];
  if (isLigand) {
    lines.push("  - ligand:");
    lines.push(`      id: ${yq(targetId || "A")}`);
    if (ligandMode === "ccd") lines.push(`      ccd: ${yq((ligand || "").trim())}`);
    else lines.push(`      smiles: ${yq((ligand || "").trim())}`);
  } else {
    lines.push("  - protein:");
    lines.push(`      id: ${yq(targetId || "A")}`);
    lines.push("      msa: empty");
    lines.push(`      sequence: ${yq((target || "").trim().replace(/\s+/g, ""))}`);
  }
  return lines.join("\n") + "\n";
}

const specHasLigand = (s) => /(^|\n)\s*-?\s*ligand\s*:/i.test(s || "");

export default function DesignForm({ catalog, onSubmitted, onError }) {
  const [protocol, setProtocol] = useState("protein-anything");
  const [name, setName] = useState("");
  const [params, setParams] = useState(() => defaultsFor(catalog.design_params));
  const [submitting, setSubmitting] = useState(false);
  // Simple form is the default; raw YAML is an optional advanced surface, just
  // like the Fold tab. Beginners never have to see BoltzGen YAML.
  const [specMode, setSpecMode] = useState("form"); // form (simple, default) | yaml (advanced)
  const [spec, setSpec] = useState("");

  // builder state
  const [target, setTarget] = useState("");       // protein-target sequence
  const [ligand, setLigand] = useState("");        // small-molecule target
  const [ligandMode, setLigandMode] = useState("smiles");
  const [targetId, setTargetId] = useState("A");
  const [binderId, setBinderId] = useState("B");
  const [lengthRange, setLengthRange] = useState(DEFAULT_LEN["protein-anything"]);

  const protoInfo = useMemo(() => catalog.protocols.find((p) => p.id === protocol), [catalog, protocol]);
  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));
  const isLigand = isLigandProtocol(protocol);

  const onProtocol = (id) => {
    setProtocol(id);
    setLengthRange(DEFAULT_LEN[id] || "80..120");
  };

  const builderArgs = { isLigand, target, ligand, ligandMode, targetId, binderId, lengthRange };

  // Turn the simple form into editable YAML and switch to the advanced view —
  // one obvious click, mirroring the Fold tab's "YAML" toggle.
  const editAsYaml = () => {
    setSpec(genSpec(builderArgs));
    setSpecMode("yaml");
  };

  const loadExample = (id) => {
    const ex = catalog.examples.find((e) => e.id === id);
    if (!ex) return;
    if (ex.protocol) onProtocol(ex.protocol);
    // Load examples into the simple form (not raw YAML), keeping everything
    // editable without touching the spec textarea.
    if (ex.builder) {
      const b = ex.builder;
      setTarget(b.target || "");
      setLigand(b.ligand || "");
      setLigandMode(b.ligandMode || "smiles");
      setTargetId(b.targetId || "A");
      setBinderId(b.binderId || "B");
      if (b.lengthRange) setLengthRange(b.lengthRange);
      setSpec("");
      setSpecMode("form");
    } else if (ex.content) {
      setSpec(ex.content);
      setSpecMode("yaml");
    }
  };

  // "Actually I don't want the example" — back to a blank simple form.
  const resetForm = () => {
    setTarget(""); setLigand(""); setLigandMode("smiles");
    setTargetId("A"); setBinderId("B");
    setLengthRange(DEFAULT_LEN[protocol] || "80..120");
    setSpec(""); setSpecMode("form"); setName("");
  };

  const designExamples = catalog.examples.filter((e) => e.kind === "design");
  // The simple form always builds the right target type for the protocol, so a
  // mismatch is only possible if someone hand-edits the YAML into a protein
  // target under the small-molecule protocol.
  const effectiveSpec = specMode === "yaml" ? spec : genSpec(builderArgs);
  const designMismatch = isLigand && specMode === "yaml" && !!effectiveSpec.trim() && !specHasLigand(effectiveSpec);

  const submit = async () => {
    let body;
    if (specMode === "yaml") {
      body = spec.trim();
      if (!body) return onError("Add a design spec, or switch to the simple form.");
    } else {
      if (isLigand && !ligand.trim()) return onError("Enter the target small molecule (SMILES or CCD code) first.");
      if (!isLigand && !target.trim()) return onError("Paste a target protein sequence first.");
      body = genSpec(builderArgs);
    }
    if (designMismatch) {
      return onError("The 'Binder + affinity' protocol designs a binder against a small molecule — your spec must include a ligand target (ccd or smiles), not a protein.");
    }
    setSubmitting(true);
    try {
      const job = await api.submit({ kind: "design", name: name.trim(), protocol, spec: body, params });
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
        <p className="section-title">Design protocol</p>
        <div className="cardgrid">
          {catalog.protocols.map((p) => (
            <button key={p.id} className={`selcard ${protocol === p.id ? "active" : ""}`} onClick={() => onProtocol(p.id)}>
              <div className="t">{p.name}</div>
              <div className="s">{p.blurb}</div>
            </button>
          ))}
        </div>
        {protoInfo && <div className="hint mt8">{protoInfo.blurb}</div>}
      </div>

      <div className="examples-row">
        <span className="examples-label">Start from an example:</span>
        {designExamples.map((e) => (
          <button key={e.id} className="chip" title="Load this example into the form" onClick={() => loadExample(e.id)}>{e.name}</button>
        ))}
        <span style={{ flex: 1 }} />
        <button className="btn sm" title="Clear the form and start blank" onClick={resetForm}>↺ Clear form</button>
      </div>

      <div className="panel">
        <div className="flex-between">
          <p className="section-title mb0">Design target</p>
          <div className="flex">
            <button className={`btn sm ${specMode === "form" ? "primary" : "ghost"}`} onClick={() => setSpecMode("form")}>Simple form</button>
            <button className={`btn sm ${specMode === "yaml" ? "primary" : "ghost"}`} onClick={() => specMode === "form" && editAsYaml()}>YAML</button>
          </div>
        </div>

        {specMode === "form" ? (
          <div className="mt16">
            {isLigand ? (
              <div className="field">
                <label>Target small molecule</label>
                <div className="row">
                  <select value={ligandMode} onChange={(e) => setLigandMode(e.target.value)} style={{ maxWidth: 110 }}>
                    <option value="smiles">SMILES</option>
                    <option value="ccd">CCD</option>
                  </select>
                  {ligandMode === "ccd"
                    ? <input type="text" value={ligand} onChange={(e) => setLigand(e.target.value)} placeholder="e.g. ATP" />
                    : <input type="text" value={ligand} onChange={(e) => setLigand(e.target.value)} placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O" />}
                </div>
                <div className="hint">The binder will be designed against this small molecule. Run on sovereign compute — your target never leaves the cluster.</div>
              </div>
            ) : (
              <div className="field">
                <label>Target protein sequence</label>
                <textarea className="code" rows={4} value={target} spellCheck={false}
                  onChange={(e) => setTarget(e.target.value)}
                  placeholder="Paste the amino-acid sequence of the protein you want to bind…" />
                <div className="hint">The binder will be designed against this target. Run on sovereign compute — your target never leaves the cluster.</div>
              </div>
            )}
            <div className="row">
              <div className="field">
                <label>Binder length</label>
                <input type="text" value={lengthRange} onChange={(e) => setLengthRange(e.target.value)} placeholder="80..120" />
                <div className="hint">Range sampled per design.</div>
              </div>
              <div className="field">
                <label>Binder chain ID</label>
                <input type="text" value={binderId} onChange={(e) => setBinderId(e.target.value)} />
              </div>
              <div className="field">
                <label>Target chain ID</label>
                <input type="text" value={targetId} onChange={(e) => setTargetId(e.target.value)} />
              </div>
            </div>
          </div>
        ) : (
          <div className="field mt16">
            <label>Spec (YAML)</label>
            <textarea className="code" rows={11} value={spec} spellCheck={false}
              onChange={(e) => setSpec(e.target.value)}
              placeholder={"entities:\n  - protein:\n      id: B\n      sequence: 80..120\n  - protein:\n      id: A\n      msa: empty\n      sequence: MVTPEG..."} />
            <div className="hint">Full BoltzGen spec — bind to proteins, small molecules, DNA or RNA; fix or redesign residues; add binding-site constraints.</div>
          </div>
        )}
      </div>

      <div className="panel">
        <p className="section-title">Design parameters</p>
        <ParamControls params={catalog.design_params} values={params} onChange={setParam} />
      </div>

      {designMismatch && (
        <div className="panel" style={{ borderColor: "var(--warn)", background: "rgba(201,138,0,0.06)" }}>
          <strong style={{ color: "var(--warn)" }}>⚠ Protocol / target mismatch.</strong> The{" "}
          <strong>Binder + affinity</strong> protocol designs a binder against a <strong>small molecule</strong>, so the
          target must be a ligand (ccd or smiles) — but this spec has a protein target. Edit the spec, or pick a
          protein-target protocol (e.g. Protein binder / Nanobody).
        </div>
      )}

      <div className="flex-between">
        <div className="field" style={{ flex: 1, marginRight: 16, marginBottom: 0 }}>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Job name (optional)" />
        </div>
        <button className="btn primary" disabled={submitting || designMismatch} onClick={submit}>
          {submitting ? "Submitting…" : "Run design →"}
        </button>
      </div>
    </>
  );
}

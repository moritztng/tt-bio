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

// Single-quote a value as a YAML scalar (escaping ' as '') so user-entered
// fields with ':', '#', '[' etc. can't break or inject into the generated spec.
const yq = (v) => `'${String(v).replace(/'/g, "''")}'`;

function genSpec({ target, targetId, binderId, lengthRange }) {
  return [
    "entities:",
    "  - protein:",
    `      id: ${yq(binderId || "B")}`,
    `      sequence: ${yq(lengthRange || "80..120")}`,
    "  - protein:",
    `      id: ${yq(targetId || "A")}`,
    "      msa: empty",
    `      sequence: ${yq((target || "").trim().replace(/\s+/g, ""))}`,
    "",
  ].join("\n");
}

export default function DesignForm({ catalog, onSubmitted, onError }) {
  const [protocol, setProtocol] = useState("protein-anything");
  const [spec, setSpec] = useState("");
  const [name, setName] = useState("");
  const [params, setParams] = useState(() => defaultsFor(catalog.design_params));
  const [submitting, setSubmitting] = useState(false);
  const [showBuilder, setShowBuilder] = useState(true);

  // builder state
  const [target, setTarget] = useState("");
  const [targetId, setTargetId] = useState("A");
  const [binderId, setBinderId] = useState("B");
  const [lengthRange, setLengthRange] = useState(DEFAULT_LEN["protein-anything"]);

  const protoInfo = useMemo(() => catalog.protocols.find((p) => p.id === protocol), [catalog, protocol]);
  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));

  const onProtocol = (id) => {
    setProtocol(id);
    setLengthRange(DEFAULT_LEN[id] || "80..120");
  };

  const applyBuilder = () => {
    if (!target.trim()) return onError("Paste a target protein sequence first.");
    setSpec(genSpec({ target, targetId, binderId, lengthRange }));
  };

  const loadExample = (id) => {
    const ex = catalog.examples.find((e) => e.id === id);
    if (!ex) return;
    if (ex.protocol) onProtocol(ex.protocol);
    setSpec(ex.content);
  };

  const submit = async () => {
    const body = spec.trim() || genSpec({ target, targetId, binderId, lengthRange });
    if (!body.trim() || (!spec.trim() && !target.trim())) return onError("Provide a target sequence or a design spec.");
    if (needsLigandTarget && !specHasLigand) {
      return onError("The 'Binder + affinity' protocol designs a binder against a small molecule — your spec must include a ligand target (ccd or smiles), not a protein. Edit the spec accordingly.");
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

  const designExamples = catalog.examples.filter((e) => e.kind === "design");
  // 'protein-small_molecule' designs a binder against a small molecule, so its
  // target must be a ligand; every other protocol targets a protein.
  const needsLigandTarget = protocol === "protein-small_molecule";
  const effectiveSpec = spec.trim() || (target.trim() ? genSpec({ target, targetId, binderId, lengthRange }) : "");
  const specHasLigand = /(^|\n)\s*-?\s*ligand\s*:/i.test(effectiveSpec);
  const designMismatch = needsLigandTarget && effectiveSpec && !specHasLigand;

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
          <button key={e.id} className="chip" title="Load this example" onClick={() => loadExample(e.id)}>{e.name}</button>
        ))}
      </div>

      <div className="panel">
        <div className="flex-between">
          <p className="section-title mb0">Design specification</p>
        </div>

        <details className="collapse" open={showBuilder} onToggle={(e) => setShowBuilder(e.target.open)}>
          <summary>Quick builder</summary>
          <div className="mt8">
            <div className="field">
              <label>Target protein sequence</label>
              <textarea className="code" rows={4} value={target} spellCheck={false}
                onChange={(e) => setTarget(e.target.value)}
                placeholder="Paste the amino-acid sequence of the protein you want to bind…" />
              <div className="hint">The binder will be designed against this target. Run on sovereign compute — the target never leaves the cluster.</div>
            </div>
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
            <button className="btn primary sm" onClick={applyBuilder}>Apply to spec ↓</button>
          </div>
        </details>

        <div className="field mt16">
          <label>Spec (YAML)</label>
          <textarea className="code" rows={9} value={spec} spellCheck={false}
            onChange={(e) => setSpec(e.target.value)}
            placeholder={"entities:\n  - protein:\n      id: B\n      sequence: 80..120\n  - protein:\n      id: A\n      msa: empty\n      sequence: MVTPEG..."} />
          <div className="hint">Full BoltzGen spec supported — bind to proteins, small molecules, DNA or RNA; fix or redesign residues; add binding-site constraints.</div>
        </div>
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

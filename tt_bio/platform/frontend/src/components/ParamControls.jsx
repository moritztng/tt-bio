import React from "react";

// Renders a list of catalog param descriptors into form controls, writing into
// the `values` object via onChange(key, value). Powers progressive disclosure:
// the common case stays simple; every tt-bio knob is reachable here.
//
// A param may declare a `cap`; if the selected model lacks that capability the
// control is greyed out and inert (e.g. MSA options on ESMFold-2 Fast, affinity
// options on a model with no affinity head) so it can't be set to no effect.
export default function ParamControls({ params, values, onChange, caps, modelName }) {
  const control = (p) => {
    const v = values[p.key];
    if (p.type === "bool") {
      return (
        <div className="checkline">
          <input type="checkbox" id={`p-${p.key}`} checked={!!v}
            onChange={(e) => onChange(p.key, e.target.checked)} />
          <div className="cl-body">
            <label className="cl-label" htmlFor={`p-${p.key}`}>{p.label}</label>
            {p.help && <div className="hint">{p.help}</div>}
          </div>
        </div>
      );
    }
    if (p.type === "enum") {
      return (
        <div className="field">
          <label>{p.label}</label>
          <select value={v ?? p.default ?? ""} onChange={(e) => onChange(p.key, e.target.value)}>
            {p.choices.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          {p.help && <div className="hint">{p.help}</div>}
        </div>
      );
    }
    if (p.type === "multienum") {
      const set = new Set(v ?? p.default ?? []);
      return (
        <div className="field">
          <label>{p.label}</label>
          <div className="flex" style={{ flexWrap: "wrap" }}>
            {p.choices.map((c) => (
              <label key={c} className="tag" style={{ cursor: "pointer", padding: "4px 10px" }}>
                <input type="checkbox" style={{ width: "auto", marginRight: 6 }} checked={set.has(c)}
                  onChange={(e) => {
                    const next = new Set(set);
                    e.target.checked ? next.add(c) : next.delete(c);
                    onChange(p.key, p.choices.filter((x) => next.has(x)));
                  }} />
                {c}
              </label>
            ))}
          </div>
          {p.help && <div className="hint">{p.help}</div>}
        </div>
      );
    }
    // int / float / text
    return (
      <div className="field">
        <label>{p.label}</label>
        <input type={p.type === "text" ? "text" : "number"} value={v ?? ""}
          placeholder={p.default != null ? String(p.default) : ""}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") return onChange(p.key, undefined);
            onChange(p.key, p.type === "text" ? raw : Number(raw));
          }} />
        {p.help && <div className="hint">{p.help}</div>}
      </div>
    );
  };

  return (
    <div>
      {params.map((p) => {
        const off = !!(p.cap && caps && !caps.has(p.cap));
        return (
          <div key={p.key} style={off ? { opacity: 0.45, pointerEvents: "none" } : undefined}>
            {control(p)}
            {off && <div className="hint">Not used by {modelName || "this model"}.</div>}
          </div>
        );
      })}
    </div>
  );
}

export function defaultsFor(params) {
  const out = {};
  for (const p of params) if (p.default !== undefined && p.default !== null) out[p.key] = p.default;
  return out;
}

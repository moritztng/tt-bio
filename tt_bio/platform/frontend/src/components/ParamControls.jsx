import React from "react";

// Renders a list of catalog param descriptors into form controls, writing into
// the `values` object via onChange(key, value). Powers progressive disclosure:
// the common case stays simple; every tt-bio knob is reachable here.
//
// A param may declare a `cap`; if the selected model lacks that capability the
// control is greyed out and inert (e.g. affinity options on a model with no
// affinity head). `locks` pins a specific param to a forced value with a
// reason shown on hover (e.g. "Generate MSA" is required for Boltz-2 and
// impossible for ESMFold-2 Fast).
export default function ParamControls({ params, values, onChange, caps, modelName, locks }) {
  const control = (p, disabled, forced) => {
    const v = forced !== undefined ? forced : values[p.key];
    if (p.type === "bool") {
      return (
        <div className="checkline">
          <input type="checkbox" id={`p-${p.key}`} checked={!!v} disabled={disabled}
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
          <select value={v ?? p.default ?? ""} disabled={disabled} onChange={(e) => onChange(p.key, e.target.value)}>
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
                <input type="checkbox" style={{ width: "auto", marginRight: 6 }} checked={set.has(c)} disabled={disabled}
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
    const isNum = p.type !== "text";
    return (
      <div className="field">
        <label>{p.label}</label>
        <input type={p.type === "text" ? "text" : "number"} value={v ?? ""} disabled={disabled}
          min={isNum && p.min != null ? p.min : undefined}
          max={isNum && p.max != null ? p.max : undefined}
          placeholder={p.default != null ? String(p.default) : ""}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") return onChange(p.key, undefined);
            if (p.type === "text") return onChange(p.key, raw);
            // Clamp into the allowed range so a demo limit can't be typed past.
            let num = Number(raw);
            if (Number.isNaN(num)) return;
            if (p.min != null) num = Math.max(p.min, num);
            if (p.max != null) num = Math.min(p.max, num);
            onChange(p.key, num);
          }} />
        {p.help && <div className="hint">{p.help}</div>}
      </div>
    );
  };

  return (
    <div>
      {params.map((p) => {
        const lock = locks && locks[p.key];
        const off = lock ? true : !!(p.cap && caps && !caps.has(p.cap));
        const reason = lock ? lock.reason : (off ? `Not used by ${modelName || "this model"}.` : null);
        // Dim with opacity (not pointer-events:none) so the explanatory tooltip
        // still shows on hover; the inputs themselves are disabled.
        return (
          <div key={p.key} style={off ? { opacity: 0.6 } : undefined} title={reason || undefined}>
            {control(p, off, lock ? lock.value : undefined)}
            {reason && <div className="hint">{reason}</div>}
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

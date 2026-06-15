// Single-quote a value as a YAML scalar (escaping ' as '') so user-entered
// fields with ':', '#', '[' etc. can't break or inject into the generated
// YAML/specs. Single source of truth for both the fold and design forms.
export const yq = (v) => `'${String(v).replace(/'/g, "''")}'`;

// Adapter from the stream to the renderer's two inputs: a topology and frames.
//
// The renderer needs, per atom: name, element, residue index; per residue: name and chain;
// optionally per-residue confidence; and per frame: Float32 xyz for every atom, a time in seconds
// and a progress in [0, 1]. Anything carrying those can drive it. Coordinates arrive either as a
// plain array, as base64 little-endian float32, or as a function that unpacks them when first shown.

export function decodeCoords(c) {
  if (c instanceof Float32Array) return c;
  if (typeof c === 'function') return c();   // a state unpacked on demand
  if (Array.isArray(c)) return Float32Array.from(c.flat ? c.flat() : c);
  const bin = atob(c), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}

export function topologyFrom(t) {
  const natom = t.atom_name.length, nres = t.res_name.length;
  return {
    natom, nres,
    atomName: t.atom_name,
    element: t.element.map(e => e.toUpperCase()),
    atomResidue: Int32Array.from(t.residue_index),
    resName: t.res_name,
    chain: t.chain ?? new Array(nres).fill('A'),
    confidence: t.confidence ? Float32Array.from(t.confidence) : null,
  };
}

export function frameFrom(f, i, n) {
  return { coords: decodeCoords(f.coords), time: f.t ?? i, progress: f.progress ?? (n > 1 ? i / (n - 1) : 1), final: !!f.final, aligned: !!f.aligned };
}

export async function loadTrajectory(url) {
  const j = await (await fetch(url)).json();
  const frames = j.frames.map((f, i) => frameFrom(f, i, j.frames.length));
  frames[frames.length - 1].final = true;
  return { meta: j.meta ?? {}, topo: topologyFrom(j.topology), frames };
}

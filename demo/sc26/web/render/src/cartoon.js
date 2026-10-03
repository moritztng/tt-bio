// Cartoon of the final structure: helices as wide ribbons, strands as flat arrows, everything else
// as a thin tube, the way PyMOL, ChimeraX and Mol* draw a protein.
//
// Secondary structure is assigned from the predicted backbone with the DSSP rules (Kabsch & Sander
// 1983): backbone hydrogen bonds from the electrostatic energy of C=O..H-N, cut at -0.5 kcal/mol;
// an alpha helix where two consecutive i -> i+4 turns start; a strand where residues form parallel
// or antiparallel bridges in a run of two or more; a 3-10 helix where two consecutive i -> i+3 turns
// start, drawn as helix. Pi helices, turns and bends are drawn as coil. The geometry is a Catmull-Rom spline through the C-alpha atoms (strands smoothed so they
// do not zig-zag), oriented by the peptide plane, so it is built from the scored coordinates alone.
//
// Ambient occlusion is baked per vertex: how much of the structure sits within 7 A in front of the
// surface at that point. Atoms that are not part of an amino-acid backbone (ligands, hemes) are not
// in the cartoon; the renderer draws them as balls.

const SUB = 8, RING = 12;
const PROFILE = { C: [0.25, 0.25], H: [1.25, 0.25], E: [1.05, 0.25] };   // half width, half thickness (A)
const ARROW = 1.65;                                                     // half width of a strand's arrow head

export function backbone(topo) {
  const idx = new Map();
  for (let i = 0; i < topo.natom; i++) idx.set(topo.atomResidue[i] * 8 + ['N', 'CA', 'C', 'O'].indexOf(topo.atomName[i]), i);
  const res = [], N = [], CA = [], C = [], O = [];
  const nr = topo.atomResidue.reduce((m, r) => Math.max(m, r), -1) + 1;
  for (let r = 0; r < nr; r++) {
    const g = [0, 1, 2, 3].map(k => idx.get(r * 8 + k));
    if (g.some(v => v === undefined)) continue;
    res.push(r); N.push(g[0]); CA.push(g[1]); C.push(g[2]); O.push(g[3]);
  }
  // atoms of residues that have a backbone are in the cartoon; the rest (ligands) are drawn as balls
  const prot = new Set(res);
  const inCartoon = Uint8Array.from(topo.atomResidue, r => prot.has(r) ? 1 : 0);
  return { res, N, CA, C, O, inCartoon };
}

const v3 = (x, i) => [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const dist = (a, b) => Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);

// One of 'H', 'E', 'C' per backbone residue, and where the chain breaks (break[i]: no peptide bond
// between residue i-1 and i).
export function secondaryStructure(bb, x) {
  const n = bb.res.length;
  const P = (a, i) => v3(x, a[i]);
  const brk = new Uint8Array(n);
  for (let i = 1; i < n; i++) brk[i] = dist(P(bb.C, i - 1), P(bb.N, i)) > 2.0 ? 1 : 0;
  // amide H placed 1.0 A from N, opposite the previous residue's C=O (the DSSP convention)
  const H = Array.from({ length: n }, (_, j) => j === 0 || brk[j] ? null
    : add(P(bb.N, j), norm(sub(P(bb.C, j - 1), P(bb.O, j - 1)))));
  const hb = new Set();   // i*n+j: C=O of i accepts from N-H of j
  const ca = Float64Array.from(bb.CA.flatMap(a => v3(x, a)));
  for (let i = 0; i < n; i++) {
    const c = P(bb.C, i), o = P(bb.O, i);
    for (let j = 0; j < n; j++) {
      if (Math.abs(i - j) < 2 || !H[j]) continue;
      const dx = ca[3 * i] - ca[3 * j], dy = ca[3 * i + 1] - ca[3 * j + 1], dz = ca[3 * i + 2] - ca[3 * j + 2];
      if (dx * dx + dy * dy + dz * dz > 81) continue;
      const nn = P(bb.N, j), h = H[j];
      const e = 0.084 * 332 * (1 / dist(o, nn) + 1 / dist(c, h) - 1 / dist(o, h) - 1 / dist(c, nn));
      if (e < -0.5) hb.add(i * n + j);
    }
  }
  const has = (i, j) => i >= 0 && j >= 0 && i < n && j < n && hb.has(i * n + j);
  const ss = new Array(n).fill('C');
  const chainOK = (a, b) => { for (let k = a + 1; k <= b; k++) if (brk[k]) return false; return true; };
  const turn = (i, k = 4) => has(i, i + k) && chainOK(i, i + k);
  for (let i = 1; i + 4 < n; i++) if (turn(i - 1) && turn(i)) for (let k = i; k < i + 4; k++) ss[k] = 'H';
  const bridge = new Uint8Array(n);
  for (let i = 1; i < n - 1; i++) for (let j = 1; j < n - 1; j++) {
    if (Math.abs(i - j) < 3) continue;
    const par = (has(i - 1, j) && has(j, i + 1)) || (has(j - 1, i) && has(i, j + 1));
    const anti = (has(i, j) && has(j, i)) || (has(i - 1, j + 1) && has(j - 1, i + 1));
    if (par || anti) bridge[i] = 1;
  }
  for (let i = 0; i < n; i++) {
    if (ss[i] !== 'C' || !bridge[i]) continue;
    const run = (i > 0 && bridge[i - 1] && !brk[i]) || (i + 1 < n && bridge[i + 1] && !brk[i + 1]);
    if (run) ss[i] = 'E';
  }
  // 3-10 helices (two consecutive i -> i+3 turns) where nothing else claimed the residues; drawn as
  // helix, as PyMOL and Mol* do
  for (let i = 1; i + 3 < n; i++) if (turn(i - 1, 3) && turn(i, 3) && [0, 1, 2].every(k => ss[i + k] === 'C'))
    for (let k = i; k < i + 3; k++) ss[k] = 'H';
  return { ss, brk };
}

const cr = (p0, p1, p2, p3, t) => {
  const t2 = t * t, t3 = t2 * t;
  return [0, 1, 2].map(k => 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2
    + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3));
};
const smooth = (t) => t * t * (3 - 2 * t);

// -> {pos, nrm, col (rgba8, a = baked occlusion), idx, ss}
export function buildCartoon(topo, bb, x, resColor) {
  const n = bb.res.length;
  if (n < 2) return null;
  const { ss, brk } = secondaryStructure(bb, x);
  // guide points: C-alpha, strands averaged with their neighbours so the pleat does not show
  const CA = bb.CA.map((a, i) => v3(x, a));
  const G = CA.map((p, i) => ss[i] === 'E' && i > 0 && i < n - 1 && !brk[i] && !brk[i + 1]
    ? add(mul(p, 0.5), mul(add(CA[i - 1], CA[i + 1]), 0.25)) : p);
  // peptide-plane direction (C -> O), flipped to stay continuous along the chain
  const D = [];
  for (let i = 0; i < n; i++) {
    let d = norm(sub(v3(x, bb.O[i]), v3(x, bb.C[i])));
    if (i > 0 && !brk[i] && dot(d, D[i - 1]) < 0) d = mul(d, -1);
    D.push(d);
  }
  // segments: runs of residues with no chain break
  const segs = [];
  let s0 = 0;
  for (let i = 1; i <= n; i++) if (i === n || brk[i]) { segs.push([s0, i - 1]); s0 = i; }
  const pos = [], nrm = [], col = [], idx = [];
  for (const [a, b] of segs) {
    if (b - a < 1) continue;
    const rows = (b - a) * SUB + 1, base = pos.length / 3;
    for (let s = 0; s < rows; s++) {
      const seg = Math.min(b - 1, a + Math.floor(s / SUB)), u = (s - (seg - a) * SUB) / SUB;
      const i0 = Math.max(a, seg - 1), i1 = seg, i2 = seg + 1, i3 = Math.min(b, seg + 2);
      const p = cr(G[i0], G[i1], G[i2], G[i3], u);
      const T = norm(sub(cr(G[i0], G[i1], G[i2], G[i3], Math.min(1, u + 0.02)), cr(G[i0], G[i1], G[i2], G[i3], Math.max(0, u - 0.02))));
      let S = add(mul(D[i1], 1 - u), mul(D[i2], u));
      S = norm(sub(S, mul(T, dot(S, T))));
      const B = cross(T, S);
      // profile: blend between the two residues' shapes; a strand's last residue ends in an arrow
      const [w1, h1] = PROFILE[ss[i1]], [w2, h2] = PROFILE[ss[i2]];
      let w = w1 + (w2 - w1) * smooth(u), h = h1 + (h2 - h1) * smooth(u);
      if (ss[i1] === 'E' && ss[i2] !== 'E') { w = Math.max(PROFILE.C[0], ARROW * (1 - u)); h = PROFILE.E[1]; }
      else if (ss[i1] !== 'E' && ss[i2] === 'E') { w = w1 + (PROFILE.E[0] - w1) * smooth(u); }
      const r = u < 0.5 ? i1 : i2, c = resColor(bb.res[r]);
      for (let k = 0; k < RING; k++) {
        const th = 2 * Math.PI * k / RING, cs = Math.cos(th), sn = Math.sin(th);
        pos.push(...add(p, add(mul(S, cs * w), mul(B, sn * h))));
        nrm.push(...norm(add(mul(S, cs * h), mul(B, sn * w))));
        col.push(c[0], c[1], c[2], 255);
      }
    }
    for (let s = 0; s < rows - 1; s++) for (let k = 0; k < RING; k++) {
      const p0 = base + s * RING + k, p1 = base + s * RING + (k + 1) % RING;
      idx.push(p0, p0 + RING, p1, p1, p0 + RING, p1 + RING);
    }
  }
  const out = { pos: new Float32Array(pos), nrm: new Float32Array(nrm), col: new Uint8Array(col), idx: new Uint32Array(idx), ss };
  bakeOcclusion(out, x);
  return out;
}

// Per-vertex openness: atoms within R of a point 2 A out along the normal, weighted by closeness.
function bakeOcclusion(m, x) {
  const R = 7, cell = R, na = x.length / 3, grid = new Map();
  const key = (a, b, c) => (a * 73856093) ^ (b * 19349663) ^ (c * 83492791);
  for (let i = 0; i < na; i++) {
    const k = key(Math.floor(x[3 * i] / cell), Math.floor(x[3 * i + 1] / cell), Math.floor(x[3 * i + 2] / cell));
    let l = grid.get(k); if (!l) grid.set(k, l = []); l.push(i);
  }
  const nv = m.pos.length / 3;
  for (let v = 0; v < nv; v++) {
    const q = [0, 1, 2].map(d => m.pos[3 * v + d] + 2 * m.nrm[3 * v + d]);
    const c = q.map(t => Math.floor(t / cell));
    let w = 0;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let e = -1; e <= 1; e++) {
      const l = grid.get(key(c[0] + a, c[1] + b, c[2] + e)); if (!l) continue;
      for (const i of l) {
        const dx = x[3 * i] - q[0], dy = x[3 * i + 1] - q[1], dz = x[3 * i + 2] - q[2], d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < R * R) w += 1 - Math.sqrt(d2) / R;
      }
    }
    m.col[4 * v + 3] = Math.round(255 * (1 - 0.55 * smooth(Math.min(1, Math.max(0, (w - 6) / 34)))));
  }
}

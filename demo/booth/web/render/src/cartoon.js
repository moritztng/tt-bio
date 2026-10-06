// Cartoon of the final structure: helices as wide ribbons, strands as flat arrows, everything else
// as a thin tube, the way PyMOL, ChimeraX and Mol* draw a protein.
//
// Secondary structure is assigned from the predicted backbone with the DSSP rules (Kabsch & Sander
// 1983): backbone hydrogen bonds from the electrostatic energy of C=O..H-N, cut at -0.5 kcal/mol;
// an alpha helix where two consecutive i -> i+4 turns start; a strand where residues form parallel
// or antiparallel bridges in a run of two or more; a 3-10 helix where two consecutive i -> i+3 turns
// start, drawn as helix. Pi helices, turns and bends are drawn as coil.
//
// The geometry is a cubic B-spline over the C-alpha atoms (strands first averaged with their
// neighbours so the pleat does not show). A B-spline is smooth to the second derivative, so a helix
// comes out as a regular spiral with no corners at the residues. A helix ribbon faces away from its
// own axis (the spline's principal normal), so it winds at constant pitch; a strand faces out of its
// sheet; a coil is a thin round tube carried along by parallel transport. All of it is built from
// the scored coordinates alone.
//
// Ambient occlusion is baked per vertex: how much of the structure sits within 7 A in front of the
// surface at that point. Atoms that are not part of an amino-acid backbone (ligands, hemes) are not
// in the cartoon; they are drawn as thin sticks (sticks() below).

import { elementColor } from './palette.js';

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

const smooth = (t) => t * t * (3 - 2 * t);
const lerp = (a, b, t) => a + (b - a) * t;

// Uniform cubic B-spline through four control points: position, first and second derivative.
// C2-continuous, so the curve has no corners where one residue hands over to the next.
function bspline(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  const b = [(1 - t) ** 3 / 6, (3 * t3 - 6 * t2 + 4) / 6, (-3 * t3 + 3 * t2 + 3 * t + 1) / 6, t3 / 6];
  const d = [-((1 - t) ** 2) / 2, (3 * t2 - 4 * t) / 2, (-3 * t2 + 2 * t + 1) / 2, t2 / 2];
  const e = [1 - t, 3 * t - 2, 1 - 3 * t, t];
  const f = (w) => [0, 1, 2].map(k => w[0] * p0[k] + w[1] * p1[k] + w[2] * p2[k] + w[3] * p3[k]);
  return [f(b), f(d), f(e)];
}

// Cross-sections, in Angstrom: half width (in the ribbon's plane), half thickness, and the
// superellipse exponent (2 a round tube, 4 a flat band with rounded edges). Proportions after
// PyMOL's and Mol*'s defaults: a thin round coil, a wide thin helix ribbon, a slightly narrower
// and thicker strand that ends in an arrow.
const PROFILE = { C: [0.2, 0.2, 2], H: [1.3, 0.2, 4], E: [1.05, 0.28, 4] };
const ARROW = 1.65;
// how far a helix control point moves out, as a multiple of CA - (its neighbours' midpoint): that
// offset is R (1 - cos 100) = 1.17 R, and the point must move 1.64 R - R = 0.64 R
const HELIX_OUT = 0.64 / 1.17;
const SUB = 12, RING = 24;   // samples per residue along the chain, around the cross-section

// -> {pos, nrm, col (rgba8, a = baked occlusion), idx, ss}
export function buildCartoon(topo, bb, x, resColor) {
  const n = bb.res.length;
  if (n < 2) return null;
  const { ss, brk } = secondaryStructure(bb, x);
  const CA = bb.CA.map(a => v3(x, a));
  // control points: C-alpha; in a strand averaged with its neighbours so the pleat does not show.
  // A B-spline over a helix's C-alphas (100 deg apart) runs at (4 + 2 cos 100)/6 = 0.61 of their
  // radius, so inside a helix each control point moves out from the axis by the difference: the
  // curve then winds at the C-alpha radius, as PyMOL's and Mol*'s helices do.
  const inner = (i) => i > 0 && i < n - 1 && !brk[i] && !brk[i + 1];
  const Q = CA.map((p, i) => {
    if (!inner(i)) return p;
    const m = mul(add(CA[i - 1], CA[i + 1]), 0.5);
    if (ss[i] === 'E') return add(mul(p, 0.5), mul(m, 0.5));
    if (ss[i] === 'H' && ss[i - 1] === 'H' && ss[i + 1] === 'H') return add(p, mul(sub(p, m), HELIX_OUT));
    return p;
  });
  // a strand's ribbon lies in the sheet: its face normal is the pleat direction, which alternates
  // residue to residue, so flip it into one continuous direction and average neighbours
  const pleat = CA.map((p, i) => i > 0 && i < n - 1 ? sub(p, mul(add(CA[i - 1], CA[i + 1]), 0.5)) : null);
  const sheetN = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    if (ss[i] !== 'E') continue;
    let v = pleat[i] ?? pleat[i + 1] ?? pleat[i - 1];
    if (!v) continue;
    v = norm(v);
    const prev = i > 0 && !brk[i] ? sheetN[i - 1] : null;
    if (prev && dot(v, prev) < 0) v = mul(v, -1);
    sheetN[i] = v;
  }
  const sheetS = sheetN.map((v, i) => {
    if (!v) return null;
    let s = v;
    for (const j of [i - 1, i + 1]) if (sheetN[j] && (j > i ? !brk[j] : !brk[i])) s = add(s, mul(sheetN[j], 0.5 * Math.sign(dot(sheetN[j], v))));
    return norm(s);
  });
  // segments: runs of residues with no chain break
  const segs = [];
  let s0 = 0;
  for (let i = 1; i <= n; i++) if (i === n || brk[i]) { segs.push([s0, i - 1]); s0 = i; }
  const pos = [], nrm = [], col = [], idx = [];
  const vert = (p, nn, c) => { pos.push(...p); nrm.push(...nn); col.push(c[0], c[1], c[2], 255); };
  for (const [a, b] of segs) {
    if (b - a < 1) continue;
    const ctl = (i) => i < a ? sub(mul(Q[a], 2), Q[Math.min(b, a + 1)]) : i > b ? sub(mul(Q[b], 2), Q[Math.max(a, b - 1)]) : Q[i];
    const rows = (b - a) * SUB + 1, base = pos.length / 3;
    let F = null;   // the ribbon's face normal, carried along the chain
    for (let s = 0; s < rows; s++) {
      const seg = Math.min(b - 1, a + Math.floor(s / SUB)), u = (s - (seg - a) * SUB) / SUB;
      // the curve from residue seg (u = 0) to seg + 1 (u = 1), over control points seg-1 .. seg+2
      const i1 = seg, i2 = seg + 1;
      const [p, d1, d2] = bspline(ctl(i1 - 1), ctl(i1), ctl(i2), ctl(i2 + 1), u);
      const T = norm(d1);
      // what this stretch wants the face normal to be: a helix faces away from its own axis (the
      // curve's principal normal, so the ribbon winds with a constant pitch), a strand faces out of
      // its sheet, a coil has no preference and is carried along by parallel transport
      const want = (i) => ss[i] === 'H' ? norm(sub(d2, mul(T, dot(d2, T)))) : ss[i] === 'E' ? sheetS[i] : null;
      const w1 = want(i1), w2 = want(i2);
      let D = w1 && w2 ? (dot(w1, w2) < 0 ? add(mul(w1, 1 - u), mul(w2, -u)) : add(mul(w1, 1 - u), mul(w2, u)))
        : w1 ?? w2;
      const weight = (w1 ? 1 - u : 0) + (w2 ? u : 0);
      const Ft = F ? sub(F, mul(T, dot(F, T))) : (D ?? perp(T));
      F = norm(Ft);
      if (D) {
        D = norm(sub(D, mul(T, dot(D, T))));
        if (dot(D, F) < 0) D = mul(D, -1);
        F = norm(add(mul(F, 1 - smooth(weight)), mul(D, smooth(weight))));
      }
      const W = cross(F, T);   // across the ribbon
      // profile: blend the two residues' shapes; a strand's last residue ends in an arrow
      // a chain's first and last residue are coil, so a terminal helix or strand tapers into the tube
      const sh = (i) => i === a || i === b ? 'C' : ss[i];
      const P1 = PROFILE[sh(i1)], P2 = PROFILE[sh(i2)], k = smooth(u);
      let [w, h, e] = [0, 1, 2].map(j => lerp(P1[j], P2[j], k));
      if (sh(i1) === 'E' && sh(i2) !== 'E') { w = lerp(ARROW, PROFILE.C[0], u); h = lerp(PROFILE.E[1], PROFILE.C[1], u * u); e = lerp(4, 2, u); }
      const r = u < 0.5 ? i1 : i2, c = resColor(bb.res[r]);
      for (let j = 0; j < RING; j++) {
        const th = 2 * Math.PI * j / RING, cs = Math.cos(th), sn = Math.sin(th);
        const ex = 2 / e, sx = Math.sign(cs) * Math.abs(cs) ** ex, sy = Math.sign(sn) * Math.abs(sn) ** ex;
        const nx = Math.sign(cs) * Math.abs(cs) ** (2 - ex) / w, ny = Math.sign(sn) * Math.abs(sn) ** (2 - ex) / h;
        vert(add(p, add(mul(W, sx * w), mul(F, sy * h))), norm(add(mul(W, nx), mul(F, ny))), c);
      }
    }
    for (let s = 0; s < rows - 1; s++) for (let j = 0; j < RING; j++) {
      const p0 = base + s * RING + j, p1 = base + s * RING + (j + 1) % RING;
      idx.push(p0, p1, p0 + RING, p1, p1 + RING, p0 + RING);   // counter-clockwise seen from outside
    }
    // close both ends of the segment
    for (const [ring, sgn] of [[0, -1], [rows - 1, 1]]) {
      const r0 = base + ring * RING, ctr = [0, 1, 2].map(d => {
        let m = 0; for (let j = 0; j < RING; j++) m += pos[3 * (r0 + j) + d]; return m / RING; });
      const T = norm(sub(v3(pos, base + (ring ? ring : 1) * RING), v3(pos, base + (ring ? ring - 1 : 0) * RING)));
      const nn = mul(T, sgn), c0 = pos.length / 3;
      const c = col.slice(4 * r0, 4 * r0 + 3);
      vert(ctr, nn, c);
      for (let j = 0; j < RING; j++) vert(v3(pos, r0 + j), nn, c);
      for (let j = 0; j < RING; j++) {
        const q0 = c0 + 1 + j, q1 = c0 + 1 + (j + 1) % RING;
        if (sgn > 0) idx.push(c0, q0, q1); else idx.push(c0, q1, q0);
      }
    }
  }
  sticks(topo, bb, x, vert, idx, pos);
  const out = { pos: new Float32Array(pos), nrm: new Float32Array(nrm), col: new Uint8Array(col), idx: new Uint32Array(idx), ss };
  bakeOcclusion(out, x);
  return out;
}

// any unit vector perpendicular to t
function perp(t) {
  const a = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return norm(cross(t, a));
}

// Ligands (a heme, a cofactor) as thin sticks in element colours, the way PyMOL and Mol* draw them
// next to a cartoon: one cylinder per bond, split half and half between the two atoms' colours, and
// a sphere of the same radius at each atom so the joints are round. A lone ion is a small ball.
const COV = { C: 0.76, N: 0.71, O: 0.66, S: 1.05, P: 1.07, FE: 1.32, ZN: 1.22, MG: 1.41, CA: 1.76, CL: 1.02, F: 0.57, BR: 1.2, I: 1.39, SE: 1.2 };
const STICK = 0.17, ION = 0.55, CYL = 14;
function sticks(topo, bb, x, vert, idx, pos) {
  const lig = [];
  for (let i = 0; i < topo.natom; i++) if (!bb.inCartoon[i] && topo.element[i] !== 'H') lig.push(i);
  if (!lig.length) return;
  const colour = (i) => elementColor(topo.element[i]).map(v => Math.round(255 * Math.sqrt(v)));
  const bonded = new Uint8Array(topo.natom);
  for (let p = 0; p < lig.length; p++) for (let q = p + 1; q < lig.length; q++) {
    const i = lig[p], j = lig[q], A = v3(x, i), B = v3(x, j);
    if (topo.atomResidue[i] !== topo.atomResidue[j]) continue;
    if (dist(A, B) > (COV[topo.element[i]] ?? 0.77) + (COV[topo.element[j]] ?? 0.77) + 0.4) continue;
    bonded[i] = bonded[j] = 1;
    const M = mul(add(A, B), 0.5);
    cylinder(A, M, colour(i)); cylinder(M, B, colour(j));
  }
  for (const i of lig) sphere(v3(x, i), bonded[i] ? STICK : ION, colour(i));

  function cylinder(A, B, c) {
    const T = norm(sub(B, A)), U = perp(T), V = cross(T, U), base = pos.length / 3;
    for (const P of [A, B]) for (let j = 0; j < CYL; j++) {
      const th = 2 * Math.PI * j / CYL, nn = add(mul(U, Math.cos(th)), mul(V, Math.sin(th)));
      vert(add(P, mul(nn, STICK)), nn, c);
    }
    for (let j = 0; j < CYL; j++) {
      const a0 = base + j, a1 = base + (j + 1) % CYL;
      idx.push(a0, a1, a0 + CYL, a1, a1 + CYL, a0 + CYL);
    }
  }
  function sphere(P, r, c) {
    const LAT = 8, LON = CYL, base = pos.length / 3;
    for (let a = 0; a <= LAT; a++) for (let o = 0; o < LON; o++) {
      const th = Math.PI * a / LAT, ph = 2 * Math.PI * o / LON;
      const nn = [Math.sin(th) * Math.cos(ph), Math.cos(th), Math.sin(th) * Math.sin(ph)];
      vert(add(P, mul(nn, r)), nn, c);
    }
    for (let a = 0; a < LAT; a++) for (let o = 0; o < LON; o++) {
      const p0 = base + a * LON + o, p1 = base + a * LON + (o + 1) % LON;
      idx.push(p0, p1, p0 + LON, p1, p1 + LON, p0 + LON);
    }
  }
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
    m.col[4 * v + 3] = Math.round(255 * (1 - 0.35 * smooth(Math.min(1, Math.max(0, (w - 6) / 34)))));
  }
}

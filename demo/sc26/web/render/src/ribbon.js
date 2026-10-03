// A flat ribbon through the backbone, for the expert. The ribbon's broad face follows the
// peptide plane (the C=O direction), so helices coil and strands lie flat without a secondary-
// structure assignment. Built on the CPU from the displayed coordinates; ~25k vertices at 400 aa.

const SUB = 8, RING = 10, HALF_W = 1.0, HALF_T = 0.22;

export function ribbonIndex(topo) {
  const ca = [], c = [], o = [], res = [];
  const find = new Map();
  for (let i = 0; i < topo.natom; i++) find.set(topo.atomResidue[i] + ':' + topo.atomName[i], i);
  for (let r = 0; r < topo.nres; r++) {
    const a = find.get(r + ':CA'), b = find.get(r + ':C'), d = find.get(r + ':O');
    if (a === undefined || b === undefined || d === undefined) continue;
    ca.push(a); c.push(b); o.push(d); res.push(r);
  }
  return { ca, c, o, res };
}

// Fixed index buffer for a given residue count: segments are cut where the chain breaks by
// collapsing the gap's triangles in the vertex data, so the topology never changes.
export function ribbonIndices(nres) {
  const rows = Math.max(0, (nres - 1) * SUB + 1), idx = [];
  for (let s = 0; s < rows - 1; s++) for (let k = 0; k < RING; k++) {
    const a = s * RING + k, b = s * RING + (k + 1) % RING, c = a + RING, d = b + RING;
    idx.push(a, c, b, b, c, d);
  }
  return new Uint32Array(idx);
}

const cr = (p0, p1, p2, p3, t) => {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
};

export function buildRibbon(x, ix, rescol, out) {
  const n = ix.ca.length;
  if (n < 2) return 0;
  const P = new Float32Array(n * 3), D = new Float32Array(n * 3), brk = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const a = ix.ca[i] * 3, c = ix.c[i] * 3, o = ix.o[i] * 3;
    P[3 * i] = x[a]; P[3 * i + 1] = x[a + 1]; P[3 * i + 2] = x[a + 2];
    let dx = x[o] - x[c], dy = x[o + 1] - x[c + 1], dz = x[o + 2] - x[c + 2];
    const l = Math.hypot(dx, dy, dz) || 1; dx /= l; dy /= l; dz /= l;
    if (i > 0 && dx * D[3 * i - 3] + dy * D[3 * i - 2] + dz * D[3 * i - 1] < 0) { dx = -dx; dy = -dy; dz = -dz; }
    D[3 * i] = dx; D[3 * i + 1] = dy; D[3 * i + 2] = dz;
    if (i > 0) brk[i] = Math.hypot(P[3 * i] - P[3 * i - 3], P[3 * i + 1] - P[3 * i - 2], P[3 * i + 2] - P[3 * i - 1]) > 4.6 ? 1 : 0;
  }
  const { pos, nrm, col } = out;
  let v = 0;
  const rows = (n - 1) * SUB + 1;
  for (let s = 0; s < rows; s++) {
    const seg = Math.min(n - 2, Math.floor(s / SUB)), t = s / SUB - seg;
    const i0 = Math.max(0, seg - 1), i1 = seg, i2 = seg + 1, i3 = Math.min(n - 1, seg + 2);
    const p = [0, 1, 2].map(k => cr(P[3 * i0 + k], P[3 * i1 + k], P[3 * i2 + k], P[3 * i3 + k], t));
    const q = [0, 1, 2].map(k => cr(P[3 * i0 + k], P[3 * i1 + k], P[3 * i2 + k], P[3 * i3 + k], Math.min(1, t + 0.01)));
    const q0 = [0, 1, 2].map(k => cr(P[3 * i0 + k], P[3 * i1 + k], P[3 * i2 + k], P[3 * i3 + k], Math.max(0, t - 0.01)));
    let T = [q[0] - q0[0], q[1] - q0[1], q[2] - q0[2]];
    let tl = Math.hypot(...T) || 1; T = T.map(c => c / tl);
    let S = [0, 1, 2].map(k => D[3 * i1 + k] * (1 - t) + D[3 * i2 + k] * t);
    const dt = S[0] * T[0] + S[1] * T[1] + S[2] * T[2];
    S = S.map((c, k) => c - T[k] * dt);
    const sl = Math.hypot(...S) || 1; S = S.map(c => c / sl);
    const B = [T[1] * S[2] - T[2] * S[1], T[2] * S[0] - T[0] * S[2], T[0] * S[1] - T[1] * S[0]];
    const gap = brk[i2] && t > 0 ? 0 : 1;  // collapse the tube across a chain break
    const r = ix.res[t < 0.5 ? i1 : i2];
    for (let k = 0; k < RING; k++) {
      const th = 2 * Math.PI * k / RING, cs = Math.cos(th), sn = Math.sin(th);
      for (let j = 0; j < 3; j++) {
        pos[3 * v + j] = p[j] + gap * (S[j] * cs * HALF_W + B[j] * sn * HALF_T);
        nrm[3 * v + j] = S[j] * cs * HALF_T + B[j] * sn * HALF_W;
      }
      const nl = Math.hypot(nrm[3 * v], nrm[3 * v + 1], nrm[3 * v + 2]) || 1;
      nrm[3 * v] /= nl; nrm[3 * v + 1] /= nl; nrm[3 * v + 2] /= nl;
      col[4 * v] = Math.round(Math.sqrt(rescol[3 * r]) * 255);
      col[4 * v + 1] = Math.round(Math.sqrt(rescol[3 * r + 1]) * 255);
      col[4 * v + 2] = Math.round(Math.sqrt(rescol[3 * r + 2]) * 255);
      col[4 * v + 3] = 235;
      v++;
    }
  }
  return v;
}

export function ribbonVertexCount(nres) { return Math.max(0, (nres - 1) * SUB + 1) * RING; }

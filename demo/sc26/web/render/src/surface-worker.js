// Gaussian molecular surface, meshed with naive surface nets, off the main thread.
//
// Density: rho(x) = sum_i exp(-|x - x_i|^2 / (2 s_i^2)), s_i = grow * r_vdw(i) / 1.177, so a lone
// atom at grow = 1 meets the 0.5 isovalue at its van der Waals radius and neighbours fuse into a
// smooth skin (the QuickSurf idea). The Gaussian is separable, so each atom costs three short 1D
// tables and a multiply per cell. Colour is splatted with the same weights; ambient occlusion is
// how much matter sits around a point, read from a blurred half-resolution copy of the density.
//
// Message in:  {id, coords, radii, colors, grow, iso, center, half, h, maxCells}
// Message out: {id, pos, nrm, col (rgba8: rgb + ao), idx, nvert, ntri, ms}

self.onmessage = (e) => {
  const t0 = performance.now();
  const m = e.data, out = mesh(m);
  out.id = m.id; out.ms = performance.now() - t0;
  self.postMessage(out, [out.pos.buffer, out.nrm.buffer, out.col.buffer, out.idx.buffer]);
};

function mesh({ coords, radii, colors, grow, iso, center, half, h: hmin, maxCells }) {
  const natom = radii.length;
  // Grid over the atoms' bounding box (clipped to the framed region), origin snapped to the grid
  // so consecutive frames sample the same lattice and the surface does not shimmer.
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], rmax = 0;
  for (let a = 0; a < natom; a++) {
    rmax = Math.max(rmax, radii[a]);
    for (let k = 0; k < 3; k++) { const v = coords[3 * a + k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
  }
  const pad = 3 * grow * rmax / 1.177 + 1;
  for (let k = 0; k < 3; k++) {
    lo[k] = Math.max(lo[k] - pad, center[k] - half);
    hi[k] = Math.min(hi[k] + pad, center[k] + half);
    if (hi[k] <= lo[k]) hi[k] = lo[k] + 1;
  }
  const vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
  const h = Math.max(hmin, Math.cbrt(vol / maxCells));
  const ox = Math.floor(lo[0] / h) * h, oy = Math.floor(lo[1] / h) * h, oz = Math.floor(lo[2] / h) * h;
  const nx = Math.ceil((hi[0] - ox) / h) + 1, ny = Math.ceil((hi[1] - oy) / h) + 1, nz = Math.ceil((hi[2] - oz) / h) + 1;
  const n = nx, nn = nx * ny, N = nn * nz;

  const rho = new Float32Array(N), cr = new Float32Array(N), cg = new Float32Array(N), cb = new Float32Array(N);
  const ex = new Float32Array(64), ey = new Float32Array(64), ez = new Float32Array(64);

  for (let a = 0; a < natom; a++) {
    const px = coords[3 * a], py = coords[3 * a + 1], pz = coords[3 * a + 2];
    const s = grow * radii[a] / 1.177, inv = -0.5 / (s * s), R = 3 * s;
    const x0 = Math.max(0, Math.ceil((px - R - ox) / h)), x1 = Math.min(nx - 1, Math.floor((px + R - ox) / h));
    const y0 = Math.max(0, Math.ceil((py - R - oy) / h)), y1 = Math.min(ny - 1, Math.floor((py + R - oy) / h));
    const z0 = Math.max(0, Math.ceil((pz - R - oz) / h)), z1 = Math.min(nz - 1, Math.floor((pz + R - oz) / h));
    if (x0 > x1 || y0 > y1 || z0 > z1 || x1 - x0 >= 64 || y1 - y0 >= 64 || z1 - z0 >= 64) continue;
    for (let i = x0; i <= x1; i++) { const d = ox + i * h - px; ex[i - x0] = Math.exp(inv * d * d); }
    for (let j = y0; j <= y1; j++) { const d = oy + j * h - py; ey[j - y0] = Math.exp(inv * d * d); }
    for (let k = z0; k <= z1; k++) { const d = oz + k * h - pz; ez[k - z0] = Math.exp(inv * d * d); }
    const r = colors[3 * a], g = colors[3 * a + 1], b = colors[3 * a + 2];
    for (let k = z0; k <= z1; k++) {
      const wz = ez[k - z0];
      for (let j = y0; j <= y1; j++) {
        const wyz = wz * ey[j - y0], row = k * nn + j * n;
        for (let i = x0; i <= x1; i++) {
          const w = wyz * ex[i - x0], c = row + i;
          rho[c] += w; cr[c] += w * r; cg[c] += w * g; cb[c] += w * b;
        }
      }
    }
  }

  // Ambient occlusion field: density at half resolution, box-blurred three times (~Gaussian).
  const mx = (nx >> 1) + 1, my = (ny >> 1) + 1, mz = (nz >> 1) + 1, mxy = mx * my;
  const occ = new Float32Array(mxy * mz), tmp = new Float32Array(Math.max(mx, my, mz));
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
    const row = k * nn + j * nx, orow = (k >> 1) * mxy + (j >> 1) * mx;
    for (let i = 0; i < nx; i++) occ[orow + (i >> 1)] += rho[row + i] * 0.125;
  }
  const blurR = Math.max(1, Math.round(2.4 / (2 * h)));
  for (let pass = 0; pass < 3; pass++) {
    for (let k = 0; k < mz; k++) for (let j = 0; j < my; j++) boxBlur(occ, tmp, k * mxy + j * mx, 1, mx, blurR);
    for (let k = 0; k < mz; k++) for (let i = 0; i < mx; i++) boxBlur(occ, tmp, k * mxy + i, mx, my, blurR);
    for (let j = 0; j < my; j++) for (let i = 0; i < mx; i++) boxBlur(occ, tmp, j * mx + i, mxy, mz, blurR);
  }

  // Surface nets: one vertex per cube that straddles the isosurface, at the mean edge crossing.
  const vid = new Int32Array(N).fill(-1);
  let cap = 1 << 16, pos = new Float32Array(cap * 3), nv = 0;
  const corner = new Float32Array(8), off = [0, 1, nx, nx + 1, nn, nn + 1, nn + nx, nn + nx + 1];
  for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const c0 = k * nn + j * nx + i;
    let mask = 0;
    for (let q = 0; q < 8; q++) { const v = rho[c0 + off[q]]; corner[q] = v; if (v > iso) mask |= 1 << q; }
    if (mask === 0 || mask === 255) continue;
    let sx = 0, sy = 0, sz = 0, ne = 0;
    for (let e = 0; e < 12; e++) {
      const qa = EDGES[2 * e], qb = EDGES[2 * e + 1];
      if (((mask >> qa) & 1) === ((mask >> qb) & 1)) continue;
      const t = (iso - corner[qa]) / (corner[qb] - corner[qa]);
      sx += (qa & 1) + ((qb & 1) - (qa & 1)) * t;
      sy += ((qa >> 1) & 1) + (((qb >> 1) & 1) - ((qa >> 1) & 1)) * t;
      sz += (qa >> 2) + ((qb >> 2) - (qa >> 2)) * t;
      ne++;
    }
    if (nv === cap) { cap *= 2; const p2 = new Float32Array(cap * 3); p2.set(pos); pos = p2; }
    pos[3 * nv] = i + sx / ne; pos[3 * nv + 1] = j + sy / ne; pos[3 * nv + 2] = k + sz / ne;
    vid[c0] = nv++;
  }

  // Faces: every grid edge with a sign change gets the quad of the four cubes around it.
  let icap = 1 << 17, idx = new Uint32Array(icap), ni = 0;
  for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
    const c = k * nn + j * nx + i, inside = rho[c] > iso;
    for (let ax = 0; ax < 3; ax++) {
      const step = ax === 0 ? 1 : ax === 1 ? nx : nn;
      if (inside === (rho[c + step] > iso)) continue;
      const du = ax === 0 ? nx : ax === 1 ? nn : 1, dv = ax === 0 ? nn : ax === 1 ? 1 : nx;
      const a = vid[c], b = vid[c - du], cc = vid[c - du - dv], d = vid[c - dv];
      if (a < 0 || b < 0 || cc < 0 || d < 0) continue;
      if (ni + 6 > icap) { icap *= 2; const i2 = new Uint32Array(icap); i2.set(idx); idx = i2; }
      if (inside) { idx[ni++] = a; idx[ni++] = b; idx[ni++] = cc; idx[ni++] = a; idx[ni++] = cc; idx[ni++] = d; }
      else { idx[ni++] = a; idx[ni++] = cc; idx[ni++] = b; idx[ni++] = a; idx[ni++] = d; idx[ni++] = cc; }
    }
  }

  // Per-vertex normal (density gradient), colour and occlusion; positions to world space.
  const P = new Float32Array(nv * 3), Nr = new Float32Array(nv * 3), C = new Uint8Array(nv * 4);
  const D = [nx, ny, nz], M = [mx, my, mz];
  for (let v = 0; v < nv; v++) {
    const x = pos[3 * v], y = pos[3 * v + 1], z = pos[3 * v + 2];
    let gx = trilinear(rho, x + 0.5, y, z, D) - trilinear(rho, x - 0.5, y, z, D);
    let gy = trilinear(rho, x, y + 0.5, z, D) - trilinear(rho, x, y - 0.5, z, D);
    let gz = trilinear(rho, x, y, z + 0.5, D) - trilinear(rho, x, y, z - 0.5, D);
    const gl = Math.hypot(gx, gy, gz) || 1; gx /= -gl; gy /= -gl; gz /= -gl;
    Nr[3 * v] = gx; Nr[3 * v + 1] = gy; Nr[3 * v + 2] = gz;
    P[3 * v] = ox + x * h; P[3 * v + 1] = oy + y * h; P[3 * v + 2] = oz + z * h;
    const w = Math.max(1e-6, trilinear(rho, x, y, z, D));
    // colour stored as sqrt(linear) for 8-bit precision in the darks; the shader squares it
    C[4 * v] = clamp255(Math.sqrt(trilinear(cr, x, y, z, D) / w));
    C[4 * v + 1] = clamp255(Math.sqrt(trilinear(cg, x, y, z, D) / w));
    C[4 * v + 2] = clamp255(Math.sqrt(trilinear(cb, x, y, z, D) / w));
    // occlusion: how much matter surrounds a probe 1.5 A outside the surface
    const o = trilinear(occ, (x + gx * 1.5 / h) * 0.5, (y + gy * 1.5 / h) * 0.5, (z + gz * 1.5 / h) * 0.5, M);
    C[4 * v + 3] = clamp255(1 / (1 + 3.5 * o * o));
  }
  return { pos: P, nrm: Nr, col: C, idx: idx.slice(0, ni), nvert: nv, ntri: ni / 3, h, n: [nx, ny, nz] };
}

const EDGES = Int8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 0, 2, 1, 3, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7]);

function clamp255(x) { return x <= 0 ? 0 : x >= 1 ? 255 : (x * 255 + 0.5) | 0; }

function trilinear(f, x, y, z, D) {
  const nx = D[0], ny = D[1], nn = nx * ny;
  x = Math.min(Math.max(x, 0), nx - 1.001); y = Math.min(Math.max(y, 0), ny - 1.001); z = Math.min(Math.max(z, 0), D[2] - 1.001);
  const i = x | 0, j = y | 0, k = z | 0, fx = x - i, fy = y - j, fz = z - k;
  const c = k * nn + j * nx + i;
  const a = f[c] + (f[c + 1] - f[c]) * fx, b = f[c + nx] + (f[c + nx + 1] - f[c + nx]) * fx;
  const d = f[c + nn] + (f[c + nn + 1] - f[c + nn]) * fx, e = f[c + nn + nx] + (f[c + nn + nx + 1] - f[c + nn + nx]) * fx;
  const ab = a + (b - a) * fy, de = d + (e - d) * fy;
  return ab + (de - ab) * fz;
}

// In-place box blur of one line of m samples starting at `base` with stride `st`.
function boxBlur(f, tmp, base, st, m, r) {
  const inv = 1 / (2 * r + 1);
  let acc = 0;
  for (let t = -r; t <= r; t++) acc += f[base + Math.min(m - 1, Math.max(0, t)) * st];
  for (let t = 0; t < m; t++) {
    tmp[t] = acc * inv;
    acc += f[base + Math.min(m - 1, t + r + 1) * st] - f[base + Math.max(0, t - r) * st];
  }
  for (let t = 0; t < m; t++) f[base + t * st] = tmp[t];
}

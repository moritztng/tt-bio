// SC26 protein renderer. WebGL2, no dependencies.
//
//   const r = new Renderer(canvas, {scale: 'auto', ease: 0.12, final: 'cartoon'});
//   r.setTopology(topo);          // atoms, residues, per-residue pLDDT (see protocol.js)
//   r.loadReplay(frames);         // or r.push(frame) per live frame
//
// While the fold runs, every atom is a small matte point at the sampler's real coordinates for the
// step on screen (trajectory.js: real states, held, with an optional short blend between two
// consecutive ones). The camera does not move. When the last state, the scored structure, is
// reached, the points give way to a cartoon of exactly those coordinates, coloured by pLDDT, and
// only then does the camera turn, slowly. `final: 'surface'` draws a Gaussian molecular surface
// instead of the cartoon.

import { program, texture, framebuffer, buffer, FULLSCREEN_VS } from './gl.js';
import { POINTS_VS, POINTS_FS, MESH_VS, MESH_FS, BG_FS, COMPOSITE_FS } from './shaders.js';
import { perspective, lookAt, Spring } from './math.js';
import { Timeline, centroid } from './trajectory.js';
import { residueColors, atomRadii, lin, POINT, POINT_SIDE, GROUND } from './palette.js';
import { backbone } from './cartoon.js';

const FOV = 26 * Math.PI / 180;
const DEG = Math.PI / 180;

export class Renderer {
  constructor(canvas, opt = {}) {
    this.canvas = canvas;
    this.opt = { scale: 'auto', msaa: 'auto', scheme: 'plddt', final: 'cartoon', ease: 0.12, orbitDegPerSec: 5,
      fill: 0.8, dotPx: [0.6, 1.25], skip: '', maxCells: 400000, meshH: 0.55, offset: [0, 0], ...opt };
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false,
      powerPreference: 'high-performance', preserveDrawingBuffer: !!opt.preserve });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.hdr = !!gl.getExtension('EXT_color_buffer_float') || !!gl.getExtension('EXT_color_buffer_half_float');
    this.prog = {
      points: program(gl, POINTS_VS, POINTS_FS, 'points'),
      mesh: program(gl, MESH_VS, MESH_FS, 'mesh'),
      bg: program(gl, FULLSCREEN_VS, BG_FS, 'bg'),
      composite: program(gl, FULLSCREEN_VS, COMPOSITE_FS, 'composite'),
    };
    this.ground = lin(GROUND);
    this.emptyVao = gl.createVertexArray();
    this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');  // GPU frame time, where exposed
    this.timeline = new Timeline({ ease: this.opt.ease });
    this.clock = 0; this.playing = false; this.speed = 1; this.live = false;
    this.fin = new Spring(0, 3.0);   // 0 points, 1 the final representation
    this.cam = { yaw: 0, spin: new Spring(0, 0.6) };
    this.worker = null;
    this.surf = { busy: false, id: 0, ntri: 0, ms: [] };
    this.stats = { frames: 0, gpuMs: [] };
    this.resize();
  }

  // ---------------------------------------------------------------- data
  setTopology(topo) {
    const gl = this.gl;
    this.topo = topo;
    const n = topo.natom;
    this.bb = backbone(topo);
    // the last fold's buffers go now, not whenever the garbage collector gets to them
    for (const b of [this.bufA, this.bufB, this.bufC, ...(this.meshBufs ?? [])]) if (b) gl.deleteBuffer(b);
    for (const v of [this.vaoPoints, this.vaoMesh]) if (v) gl.deleteVertexArray(v);
    this.bufA = buffer(gl, gl.ARRAY_BUFFER, new Float32Array(n * 3), gl.DYNAMIC_DRAW);
    this.bufB = buffer(gl, gl.ARRAY_BUFFER, new Float32Array(n * 3), gl.DYNAMIC_DRAW);
    // while folding the backbone (N, CA, C) is light grey and every other atom, ligands included,
    // a darker one, so the chain and its helices read in the points before the cartoon
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) col.set(this.bb.inCartoon[i] && ['N', 'CA', 'C'].includes(topo.atomName[i]) ? POINT : POINT_SIDE, 3 * i);
    this.bufC = buffer(gl, gl.ARRAY_BUFFER, col);
    this.vaoPoints = gl.createVertexArray();
    gl.bindVertexArray(this.vaoPoints);
    [[this.bufA, 0, 3], [this.bufB, 1, 3], [this.bufC, 2, 3]].forEach(([b, loc, k]) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, k, gl.FLOAT, false, 0, 0);
    });
    this.meshBufs = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
    this.vaoMesh = gl.createVertexArray();
    gl.bindVertexArray(this.vaoMesh);
    const b = this.meshBufs;
    gl.bindBuffer(gl.ARRAY_BUFFER, b[0]); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[1]); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[2]); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b[3]);
    gl.bindVertexArray(null);
    this.meshCount = 0;
    this.timeline = new Timeline({ ease: this.opt.ease });
    this.shown = { a: null, b: null }; this.dot = 1;
    this.fin.x = this.fin.target = 0; this.fin.v = 0;
    // until the final structure is known, frame for a globular protein of this length:
    // Rg ~ 2.2 N^0.38 A, bounding radius ~1.6 Rg
    this.expectRg = 2.2 * Math.pow(topo.nres, 0.38);
    this.finalRadius = null;
    this.frameC = [0, 0, 0];
    this.basis = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    this.cam.yaw = 0; this.cam.spin.x = this.cam.spin.target = 0; this.cam.spin.v = 0;
  }

  setScheme(s) { this.opt.scheme = s; if (this.timeline.final) this._buildFinal(); }

  loadReplay(frames) {
    this.live = false;
    this.timeline = new Timeline({ ease: this.opt.ease });
    this.timeline.load(frames);
    this._frameFinal(this.timeline.final.coords);
    this.clock = frames[0].time;
  }

  push(frame) {
    this.live = true;
    this.timeline.push(frame);
    if (frame.final) this._frameFinal(this.timeline.final.coords);
  }

  play() { this.playing = true; }
  pause() { this.playing = false; }
  seek(t) { this.clock = this.timeline.frames[0].time + t; }
  setMode() {}   // kept for callers of the old API; the fold drives the look

  // ---------------------------------------------------------------- camera
  // Frame the FINAL structure (or, live, the size a protein of this length will fold to) and never
  // refit to the noise: noise spills past the edges and the protein arrives. The view looks down
  // the structure's shortest principal axis with its longest axis across the screen, and holds
  // still until the fold is done.
  _distanceFor(radius) {
    const aspect = this.canvas.width / this.canvas.height;
    const half = Math.min(FOV / 2, Math.atan(Math.tan(FOV / 2) * aspect));
    return radius / Math.sin(half * this.opt.fill);
  }

  get radius() { return this.finalRadius ?? 1.6 * this.expectRg + 2; }

  _frameFinal(x) {
    const c = centroid(x);
    let r = 0;
    for (let i = 0; i < x.length / 3; i++) r = Math.max(r, Math.hypot(x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]));
    this.finalRadius = r + 2;
    this.frameC = c;
    this.basis = principalBasis(x, c);
    this._buildFinal();
  }

  // ---------------------------------------------------------------- final representation
  // Built in mesh-worker.js, never on the render thread: the cartoon of a large complex took up to
  // 1.8 s there and froze the page at every fold change. The points stay on screen until the mesh
  // arrives (the fold takes seconds to land; the mesh, a fraction of one), and a mesh built for an
  // earlier fold or colour scheme is dropped.
  _buildFinal() {
    const x = this.timeline.final.coords, topo = this.topo;
    const rc = residueColors(topo, this.opt.scheme);
    if (this.opt.final === 'surface') return this._requestSurface(x, rc);
    this._mesh({ kind: 'cartoon', topo: { natom: topo.natom, element: topo.element, atomResidue: topo.atomResidue },
      bb: this.bb, x, rc });
  }

  _mesh(msg) {
    if (!this.worker) {
      this.worker = new Worker(new URL('./mesh-worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = ({ data: m }) => {
        if (m.id !== this.surf.id || m.empty) return;
        if (m.kind === 'cartoon') { this.cartoonMs = m.ms; this.ss = m.ss; }
        else { this.surf.last = { ms: m.ms, ntri: m.ntri }; }
        this._upload(m);
      };
    }
    this.worker.postMessage({ id: ++this.surf.id, ...msg });
  }

  _upload(m) {
    const gl = this.gl, b = this.meshBufs;
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[0]); gl.bufferData(gl.ARRAY_BUFFER, m.pos, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[1]); gl.bufferData(gl.ARRAY_BUFFER, m.nrm, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[2]); gl.bufferData(gl.ARRAY_BUFFER, m.col, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b[3]); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.idx, gl.STATIC_DRAW);
    this.meshCount = m.idx.length;
  }

  // Gaussian molecular surface of the protein atoms: each atom a Gaussian whose lone isosurface
  // sits at its van der Waals radius, summed and contoured at 0.5, so neighbours fuse into one
  // smooth skin (the QuickSurf method, probe radius 0). Ligands stay balls.
  _requestSurface(x, rc) {
    const topo = this.topo, k = [];
    for (let i = 0; i < topo.natom; i++) if (this.bb.inCartoon[i]) k.push(i);
    if (!k.length) return;
    const radii = atomRadii(topo), c = new Float32Array(k.length * 3), col = new Float32Array(k.length * 3), rad = new Float32Array(k.length);
    const grey = [0.6, 0.6, 0.6];
    k.forEach((a, j) => {
      const r = topo.atomResidue[a];
      rad[j] = radii[a];
      col.set(r < topo.nres ? rc.subarray(3 * r, 3 * r + 3) : grey, 3 * j);
      for (let d = 0; d < 3; d++) c[3 * j + d] = x[3 * a + d];
    });
    this._mesh({ kind: 'surface', coords: c, radii: rad, colors: col, grow: 1, iso: 0.5,
      center: this.frameC, half: this.finalRadius + 6, h: this.opt.meshH, maxCells: this.opt.maxCells });
  }

  // ---------------------------------------------------------------- targets
  resize() {
    const gl = this.gl, c = this.canvas;
    // 'auto' is full resolution with 4x MSAA at every size, so a dot or an edge at 4K is drawn at
    // 4K. On qb2's iGPU (Radeon in the Ryzen 7 9700X), headless 3840x2160, the largest gallery
    // protein (spike-ACE2, 6,761 atoms) ran 48 fps this way and 48 fps at 0.75 scale with 2x MSAA.
    const W = c.width, H = c.height;
    const s = this.opt.scale === 'auto' ? 1 : this.opt.scale;
    const msaa = this.opt.msaa === 'auto' ? 4 : this.opt.msaa;
    this.scale = s;
    const w = Math.max(1, Math.round(W * s)), h = Math.max(1, Math.round(H * s));
    if (this.rt && this.rt.w === w && this.rt.h === h && this.rt.W === W && this.rt.H === H) return;
    const fmt = this.hdr ? gl.R11F_G11F_B10F : gl.RGBA8;
    const rt = { w, h, W, H };
    rt.color = texture(gl, w, h, fmt);
    rt.resolve = framebuffer(gl, rt.color);
    const samples = Math.min(msaa, gl.getInternalformatParameter(gl.RENDERBUFFER, fmt, gl.SAMPLES)?.[0] ?? 0);
    rt.samples = samples;
    rt.msFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, rt.msFbo);
    const cb = gl.createRenderbuffer(), db = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, cb);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, fmt, w, h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, cb);
    gl.bindRenderbuffer(gl.RENDERBUFFER, db);
    gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, w, h);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, db);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.rt = rt;
  }

  // Dot size while the cloud is spread out. On-screen ink scales with the atom count, so a
  // 400-atom protein's noise reads as an empty screen where a 4,500-atom one fills it. Small
  // proteins draw their dots up to 1.4x larger while the cloud is over ~1.5x the final radius of
  // gyration, back to 1x by the time it forms. Positions are untouched; only the dot size changes.
  _dotScale(x) {
    const k = Math.min(1.4, Math.max(1, Math.sqrt(2000 / this.topo.natom)));
    if (k === 1) return 1;
    const fin = this.timeline.final;
    if (fin && this._rgFinalOf !== fin) { this._rgFinalOf = fin; this._rgFinal = gyration(fin.coords); }
    const spread = gyration(x) / ((fin ? this._rgFinal : this.expectRg) || 1);
    const t = Math.min(1, Math.max(0, (spread - 1.5) / 2.5));
    return 1 + (k - 1) * t * t * (3 - 2 * t);
  }

  // ---------------------------------------------------------------- frame
  update(dt) {
    const tl = this.timeline;
    if (!tl.frames.length) return null;
    if (this.live) this.clock = performance.now() / 1000 - Math.min(1.0, tl.delay || 0.1);
    else if (this.playing) this.clock += dt * this.speed;
    const s = tl.at(this.clock);
    const gl = this.gl;
    if (s.a !== this.shown.a) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.bufA); gl.bufferSubData(gl.ARRAY_BUFFER, 0, s.a.coords);
      this.dot = this._dotScale(s.a.coords);
    }
    if (s.b !== this.shown.b) { gl.bindBuffer(gl.ARRAY_BUFFER, this.bufB); gl.bufferSubData(gl.ARRAY_BUFFER, 0, s.b.coords); }
    this.shown = s;
    this.progress = s.progress;
    this.step = s.a.step ?? s.index;
    this.done = s.alpha === 0 && s.a === tl.final;
    this.fin.target = this.done && this.meshCount ? 1 : 0;
    if (!this.done) Object.assign(this.fin, { x: 0, v: 0 });   // a new fold never starts half-cartoon
    this.fin.step(dt);
    // the camera turns only once the structure is done
    this.cam.spin.target = this.done ? this.opt.orbitDegPerSec * DEG : 0;
    if (!this.done) Object.assign(this.cam, { yaw: 0 }), Object.assign(this.cam.spin, { x: 0, v: 0 });
    this.cam.yaw += this.cam.spin.step(dt) * dt;
    return s;
  }

  view() {
    const c = this.frameC, B = this.basis, R = this._distanceFor(this.radius), yaw = this.cam.yaw;
    // B columns are (long, mid, short): look down short, long across, mid up; turn about mid
    const dir = [0, 1, 2].map(k => B[6 + k] * Math.cos(yaw) + B[k] * Math.sin(yaw));
    const eye = [c[0] + dir[0] * R, c[1] + dir[1] * R, c[2] + dir[2] * R];
    const aspect = this.canvas.width / this.canvas.height;
    const proj = perspective(FOV, aspect, R * 0.05, R * 40);
    proj[8] = -this.opt.offset[0] * 2; proj[9] = -this.opt.offset[1] * 2;  // lens shift
    return { view: lookAt(eye, c, [B[3], B[4], B[5]]), proj, dist: R };
  }

  // Camera proof: fraction of displayed atoms inside the viewport, and the camera distance.
  onScreen() {
    const s = this.shown; if (!s?.a) return null;
    const x = interp(s, new Float32Array(s.a.coords.length)), v = this.view();
    const M = v.view, P = v.proj; let inside = 0; const n = x.length / 3;
    for (let i = 0; i < n; i++) {
      const X = x[3 * i], Y = x[3 * i + 1], Z = x[3 * i + 2];
      const vx = M[0] * X + M[4] * Y + M[8] * Z + M[12], vy = M[1] * X + M[5] * Y + M[9] * Z + M[13], vz = M[2] * X + M[6] * Y + M[10] * Z + M[14];
      if (vz >= 0) continue;
      const cx = (P[0] * vx + P[8] * vz) / -vz, cy = (P[5] * vy + P[9] * vz) / -vz;
      if (Math.abs(cx) <= 1 && Math.abs(cy) <= 1) inside++;
    }
    return { onscreen: inside / n, camDist: v.dist };
  }

  _light(p, v) {
    const gl = this.gl;
    gl.uniform3fv(p.u.uGround, this.ground);
    gl.uniform2f(p.u.uFog, v.dist - 0.6 * this.radius, v.dist + 2.5 * this.radius);
    gl.uniformMatrix4fv(p.u.uView, false, v.view);
    gl.uniformMatrix4fv(p.u.uProj, false, v.proj);
  }

  render(dt) {
    const t0 = performance.now();
    const gl = this.gl, T = this.timer;
    if (T && this._q && gl.getQueryParameter(this._q, gl.QUERY_RESULT_AVAILABLE)) {
      if (!gl.getParameter(T.GPU_DISJOINT_EXT)) this.stats.gpuMs.push(gl.getQueryParameter(this._q, gl.QUERY_RESULT) / 1e6);
      if (this.stats.gpuMs.length > 600) this.stats.gpuMs.shift();
      gl.deleteQuery(this._q); this._q = null;
    }
    const timing = T && !this._q;
    if (timing) { this._q = gl.createQuery(); gl.beginQuery(T.TIME_ELAPSED_EXT, this._q); }
    const s = this.update(dt);
    const rt = this.rt;
    gl.bindFramebuffer(gl.FRAMEBUFFER, rt.msFbo);
    gl.viewport(0, 0, rt.w, rt.h);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.prog.bg.p);
    gl.uniform3fv(this.prog.bg.u.uGround, this.ground);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (s) {
      const v = this.view(), f = this.fin.x;
      gl.enable(gl.DEPTH_TEST);
      if (f < 0.999 && !this.opt.skip.includes('points')) {   // under an opaque cartoon the points are hidden
        const p = this.prog.points;
        gl.useProgram(p.p);
        this._light(p, v);
        // dot radius in pixels of the output, one 1080p pixel = rt.H / 1080, then into the target's
        const px = rt.H / 1080 * this.scale * this.dot;
        gl.uniform1f(p.u.uAlpha, s.alpha);
        gl.uniform1f(p.u.uFold, 1 - f);
        gl.uniform1f(p.u.uMinPx, this.opt.dotPx[0] * px);
        gl.uniform1f(p.u.uMaxPx, this.opt.dotPx[1] * px);
        gl.uniform1f(p.u.uRef, v.dist - 0.5 * this.radius);
        gl.uniform1f(p.u.uNear, v.dist * 0.12);
        gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false);
        gl.bindVertexArray(this.vaoPoints);
        gl.drawArrays(gl.POINTS, 0, this.topo.natom);
        gl.disable(gl.BLEND); gl.depthMask(true);
      }
      // the final representation fades in over the points: depth prepass, then one blended layer;
      // once it is opaque, one plain pass
      if (f >= 0.999 && this.meshCount) this._drawMesh(v, 1, false);
      else if (f > 0.002 && this.meshCount) {
        gl.colorMask(false, false, false, false);
        this._drawMesh(v, 1, false);
        gl.colorMask(true, true, true, true);
        gl.depthFunc(gl.LEQUAL);
        this._drawMesh(v, f, true);
        gl.depthFunc(gl.LESS);
      }
      gl.disable(gl.DEPTH_TEST);
    }
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, rt.msFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, rt.resolve);
    gl.blitFramebuffer(0, 0, rt.w, rt.h, 0, 0, rt.w, rt.h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, rt.W, rt.H);
    const c = this.prog.composite;
    gl.useProgram(c.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, rt.color);
    gl.uniform1i(c.u.uScene, 0);
    gl.uniform1f(c.u.uSeed, (this.stats.frames % 64) * 0.618);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (timing) gl.endQuery(T.TIME_ELAPSED_EXT);
    this.stats.frames++;
    this.stats.cpuMs = performance.now() - t0;
  }

  _drawMesh(v, opacity, blended) {
    const gl = this.gl, p = this.prog.mesh;
    gl.useProgram(p.p);
    this._light(p, v);
    gl.uniform1f(p.u.uOpacity, opacity);
    if (blended) { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(false); }
    gl.bindVertexArray(this.vaoMesh);
    gl.drawElements(gl.TRIANGLES, this.meshCount, gl.UNSIGNED_INT, 0);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
  }
}

function interp(s, out) {
  const A = s.a.coords, B = s.b.coords, al = s.alpha;
  if (al === 0) { out.set(A); return out; }
  for (let i = 0; i < A.length; i++) out[i] = A[i] + (B[i] - A[i]) * al;
  return out;
}

// Principal axes of a structure, as a column basis (long, mid, short), right-handed.
function principalBasis(x, c) {
  const n = x.length / 3, C = [0, 0, 0, 0, 0, 0];
  for (let i = 0; i < n; i++) {
    const a = x[3 * i] - c[0], b = x[3 * i + 1] - c[1], d = x[3 * i + 2] - c[2];
    C[0] += a * a; C[1] += a * b; C[2] += a * d; C[3] += b * b; C[4] += b * d; C[5] += d * d;
  }
  const M = [[C[0], C[1], C[2]], [C[1], C[3], C[4]], [C[2], C[4], C[5]]];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 20; sweep++) for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
    if (Math.abs(M[p][q]) < 1e-12) continue;
    const th = (M[q][q] - M[p][p]) / (2 * M[p][q]);
    const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
    const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
    for (let k = 0; k < 3; k++) { const a = M[k][p], b = M[k][q]; M[k][p] = cs * a - sn * b; M[k][q] = sn * a + cs * b; }
    for (let k = 0; k < 3; k++) { const a = M[p][k], b = M[q][k]; M[p][k] = cs * a - sn * b; M[q][k] = sn * a + cs * b; }
    for (let k = 0; k < 3; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = cs * a - sn * b; V[k][q] = sn * a + cs * b; }
  }
  const order = [0, 1, 2].sort((i, j) => M[j][j] - M[i][i]);
  const col = (i) => [V[0][i], V[1][i], V[2][i]];
  const e1 = col(order[0]), e2 = col(order[1]);
  const e3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
  return [...e1, ...e2, ...e3];
}

function gyration(x) {
  const [cx, cy, cz] = centroid(x), n = x.length / 3;
  let s = 0;
  for (let i = 0; i < n; i++) s += (x[3 * i] - cx) ** 2 + (x[3 * i + 1] - cy) ** 2 + (x[3 * i + 2] - cz) ** 2;
  return Math.sqrt(s / n);
}

// SC26 protein renderer. WebGL2, no dependencies.
//
//   const r = new Renderer(canvas, {scale: 0.75});
//   r.setTopology(topo);          // atoms, residues (see protocol.js)
//   r.loadReplay(frames);         // or r.push(frame) per live frame
//   r.setMode('auto');            // 'auto' | 'points' | 'surface' | 'glass' | 'ribbon'
//
// Modes are weights that move on critically damped springs, so any change of mode at any moment
// is a smooth cross-fade, never a cut. In 'auto' the fold drives the look: points while it is noise,
// a surface that grows out of the points as it condenses. Atoms the model is unsure of (per-residue
// confidence under opt.confFloor, pLDDT 70 by default) never get a surface: they stay points of
// light over the finished skin, so confidence is the difference between a solid and a cloud.

import { program, texture, framebuffer, buffer, FULLSCREEN_VS } from './gl.js';
import { POINTS_VS, POINTS_FS, SURFACE_VS, SURFACE_FS, DOWN_FS, UP_FS, BG_FS, COMPOSITE_FS } from './shaders.js';
import { perspective, lookAt, Spring } from './math.js';
import { Timeline, centroid } from './trajectory.js';
import { atomColors, atomRadii, residueColors } from './palette.js';
import { ribbonIndex, ribbonIndices, buildRibbon, ribbonVertexCount } from './ribbon.js';

const FOV = 26 * Math.PI / 180;
const DEG = Math.PI / 180;

const MODES = {
  //          points surface solid ribbon
  points:   [1, 0, 0, 0],
  surface:  [0.0, 1, 1, 0],
  glass:    [0.55, 1, 0, 1],
  ribbon:   [0.25, 0, 0, 1],
};

export class Renderer {
  constructor(canvas, opt = {}) {
    this.canvas = canvas;
    this.opt = { scale: 'auto', msaa: 'auto', scheme: 'chain', orbitDegPerSec: 3, fill: 0.8, bloom: 0.35,
      exposure: 1.0, skip: '', pointRadius: 0.3, aperture: 0.05, maxCells: 400000, meshH: 0.55, offset: [0, 0], confFloor: 70, ...opt };
    const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: false,
      powerPreference: 'high-performance', preserveDrawingBuffer: !!opt.preserve });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.hdr = !!gl.getExtension('EXT_color_buffer_float') || !!gl.getExtension('EXT_color_buffer_half_float');
    gl.getExtension('OES_texture_float_linear');
    this.prog = {
      points: program(gl, POINTS_VS, POINTS_FS, 'points'),
      surface: program(gl, SURFACE_VS, SURFACE_FS, 'surface'),
      down: program(gl, FULLSCREEN_VS, DOWN_FS, 'down'),
      up: program(gl, FULLSCREEN_VS, UP_FS, 'up'),
      bg: program(gl, FULLSCREEN_VS, BG_FS, 'bg'),
      composite: program(gl, FULLSCREEN_VS, COMPOSITE_FS, 'composite'),
    };
    this.emptyVao = gl.createVertexArray();
    this.timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');  // GPU frame time, where exposed
    this.timeline = new Timeline();
    this.clock = 0; this.playing = false; this.speed = 1; this.live = false;
    this.w = { points: new Spring(1, 2.2), surface: new Spring(0, 2.2), solid: new Spring(0, 1.6),
      ribbon: new Spring(0, 2.2), grow: new Spring(0, 1.1) };
    this.mode = 'auto';
    this.cam = { yaw: new Spring(0, 0.5), radius: new Spring(30, 0.9), elev: 14 * DEG, t: 0, yawAngle: 0 };
    this.worker = new Worker(new URL('./surface-worker.js', import.meta.url));
    this.worker.onmessage = (e) => this._onMesh(e.data);
    this.surf = { busy: false, id: 0, ntri: 0, ms: [], lastKey: '' };
    this.stats = { frames: 0, gpuMs: [] };
    this.resize();
  }

  // ---------------------------------------------------------------- data
  setTopology(topo) {
    const gl = this.gl;
    this.topo = topo;
    this.radii = atomRadii(topo);
    this.colors = atomColors(topo, this.opt.scheme);
    this.resColors = residueColors(topo, this.opt.scheme);
    const n = topo.natom;
    this.bufA = buffer(gl, gl.ARRAY_BUFFER, new Float32Array(n * 3), gl.DYNAMIC_DRAW);
    this.bufB = buffer(gl, gl.ARRAY_BUFFER, new Float32Array(n * 3), gl.DYNAMIC_DRAW);
    this.bufC = buffer(gl, gl.ARRAY_BUFFER, this.colors);
    // confidence arrives as pLDDT in 0-1 or 0-100; 1 marks an atom of an unsure residue
    const conf = topo.confidence, scale = conf && Math.max(...conf) <= 1.01 ? 100 : 1;
    this.unsure = new Float32Array(n);
    if (conf) for (let a = 0; a < n; a++) this.unsure[a] = conf[topo.atomResidue[a]] * scale < this.opt.confFloor ? 1 : 0;
    this.sure = this.unsure.some(u => u) ? Uint32Array.from({ length: n }, (_, a) => a).filter(a => !this.unsure[a]) : null;
    this.bufK = buffer(gl, gl.ARRAY_BUFFER, this.unsure);
    this.vaoPoints = gl.createVertexArray();
    gl.bindVertexArray(this.vaoPoints);
    [[this.bufA, 0, 3], [this.bufB, 1, 3], [this.bufC, 2, 3], [this.bufK, 3, 1]].forEach(([b, loc, k]) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, k, gl.FLOAT, false, 0, 0);
    });
    // surface mesh buffers, refilled by the worker
    this.surfBufs = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
    this.vaoSurf = this._meshVao(this.surfBufs);
    // ribbon
    this.rib = ribbonIndex(topo);
    const rv = ribbonVertexCount(this.rib.ca.length);
    this.ribData = { pos: new Float32Array(rv * 3), nrm: new Float32Array(rv * 3), col: new Uint8Array(rv * 4) };
    this.ribBufs = [gl.createBuffer(), gl.createBuffer(), gl.createBuffer(), gl.createBuffer()];
    this.vaoRib = this._meshVao(this.ribBufs);
    const ri = ribbonIndices(this.rib.ca.length);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ribBufs[3]);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, ri, gl.STATIC_DRAW);
    this.ribCount = ri.length;
    gl.bindVertexArray(null);
    this.timeline = new Timeline();
    this.surf.ntri = 0;
    this.shown = { a: null, b: null };
    // until the final structure is known, frame for a globular protein of this length:
    // Rg ~ 2.2 N^0.38 A, bounding radius ~1.6 Rg
    this.expectRg = 2.2 * Math.pow(topo.nres, 0.38);
    this.frameC = [0, 0, 0];
    this.cam.radius.x = this.cam.radius.target = this._distanceFor(1.6 * this.expectRg + 2);
    this.basis = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  }

  setScheme(s) {
    this.opt.scheme = s;
    if (!this.topo) return;
    this.colors = atomColors(this.topo, s);
    this.resColors = residueColors(this.topo, s);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bufC); gl.bufferData(gl.ARRAY_BUFFER, this.colors, gl.STATIC_DRAW);
    this.surf.lastKey = '';
  }

  _meshVao(b) {
    const gl = this.gl, vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[0]); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[1]); gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, b[2]); gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b[3]);
    gl.bindVertexArray(null);
    return vao;
  }

  loadReplay(frames) {
    this.live = false;
    this.timeline = new Timeline();
    this.timeline.load(frames);
    this._frameFinal(this.timeline.final.coords, true);
    this.clock = frames[0].time;
  }

  push(frame) {
    this.live = true;
    this.timeline.push(frame);
    if (frame.final) this._frameFinal(this.timeline.final.coords, false);
  }

  play() { this.playing = true; }
  pause() { this.playing = false; }
  seek(t) { this.clock = this.timeline.frames[0].time + t; }

  setMode(m) {
    this.mode = m;
    const w = MODES[m];
    if (!w) return;
    this.w.points.target = w[0]; this.w.surface.target = w[1]; this.w.solid.target = w[2]; this.w.ribbon.target = w[3];
    this.w.grow.target = w[1] > 0 ? 1 : 0;
  }

  // ---------------------------------------------------------------- camera
  // The framing rule: frame the FINAL structure (or, live, the size a protein of this length will
  // fold to) and never refit to the noise. Noise spills past the edges and the protein arrives.
  _distanceFor(radius) {
    const aspect = this.canvas.width / this.canvas.height;
    const half = Math.min(FOV / 2, Math.atan(Math.tan(FOV / 2) * aspect));
    return radius / Math.sin(half * this.opt.fill);
  }

  _frameFinal(x, snap) {
    const c = centroid(x);
    let r = 0;
    for (let i = 0; i < x.length / 3; i++) r = Math.max(r, Math.hypot(x[3 * i] - c[0], x[3 * i + 1] - c[1], x[3 * i + 2] - c[2]));
    this.finalRadius = r + 2;
    this.cam.radius.target = this._distanceFor(this.finalRadius);
    this.frameC = c;
    if (snap) {
      this.cam.radius.x = this.cam.radius.target;
      this.basis = principalBasis(x, c);  // orbit about the structure's shortest axis
    }
  }

  // ---------------------------------------------------------------- surface
  _requestMesh(coords, grow) {
    if (this.surf.busy) return;
    let radii = this.radii, colors = this.colors;
    if (this.sure) {   // only the atoms the model is sure of are meshed
      const k = this.sure;
      if (!k.length) { this.surf.ntri = 0; return; }
      const c = new Float32Array(k.length * 3), col = new Float32Array(k.length * 3);
      radii = new Float32Array(k.length);
      k.forEach((a, j) => { radii[j] = this.radii[a]; for (let d = 0; d < 3; d++) { c[3 * j + d] = coords[3 * a + d]; col[3 * j + d] = this.colors[3 * a + d]; } });
      coords = c; colors = col;
    }
    const key = this.shown.a?.time + ':' + this.shown.b?.time + ':' + this.shown.alpha?.toFixed(3) + ':' + grow.toFixed(3) + ':' + this.opt.scheme;
    if (key === this.surf.lastKey) return;
    this.surf.lastKey = key;
    this.surf.busy = true;
    this.surf.reqT = performance.now();
    this.worker.postMessage({ id: ++this.surf.id, coords, radii, colors,
      grow: 0.5 + 0.5 * grow, iso: 0.5, center: this.frameC, half: (this.finalRadius ?? 1.6 * this.expectRg) + 6,
      h: this.opt.meshH, maxCells: this.opt.maxCells });
  }

  _onMesh(m) {
    const gl = this.gl;
    this.surf.busy = false;
    this.surf.ms.push(m.ms);
    if (this.surf.ms.length > 240) this.surf.ms.shift();
    this.surf.last = { ms: m.ms, nvert: m.nvert, ntri: m.ntri, h: m.h, n: m.n };
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.surfBufs[0]); gl.bufferData(gl.ARRAY_BUFFER, m.pos, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.surfBufs[1]); gl.bufferData(gl.ARRAY_BUFFER, m.nrm, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.surfBufs[2]); gl.bufferData(gl.ARRAY_BUFFER, m.col, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.surfBufs[3]); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, m.idx, gl.DYNAMIC_DRAW);
    this.surf.ntri = m.ntri;
  }

  // ---------------------------------------------------------------- targets
  resize() {
    const gl = this.gl, c = this.canvas;
    // 'auto' is what holds 60 fps on qb2's iGPU (Radeon in the Ryzen 7 9700X), measured:
    // up to 1440p full resolution with 4x MSAA; above it 0.75 scale with 2x MSAA
    // (4K: 60.0 fps, 0 of 901 frames over 20 ms; full res 2x MSAA managed 46 fps).
    const W = c.width, H = c.height, big = W * H > 4.0e6;
    const s = this.opt.scale === 'auto' ? (big ? 0.75 : 1) : this.opt.scale;
    const msaa = this.opt.msaa === 'auto' ? (big ? 2 : 4) : this.opt.msaa;
    this.scale = s;
    const w = Math.max(1, Math.round(W * s)), h = Math.max(1, Math.round(H * s));
    if (this.rt && this.rt.w === w && this.rt.h === h && this.rt.W === W && this.rt.H === H) return;
    // R11G11B10F: HDR in 32 bits. On an iGPU sharing DDR5 the frame is bandwidth-bound, and a
    // 4x-multisampled RGBA16F target at 4K alone is 265 MB per frame to clear and resolve.
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
    rt.bloom = [];
    // Above ~1440p the glow chain starts at quarter resolution: it is soft by nature, and on this
    // iGPU a half-res 4K bloom alone cost the frame its vsync (45 fps vs 59.8 without it).
    const bigBloom = w > 2600;
    let bw = bigBloom ? w >> 1 : w, bh = bigBloom ? h >> 1 : h;
    for (let i = 0; i < (bigBloom ? 5 : 6) && bw > 8 && bh > 8; i++) {
      bw = Math.max(1, bw >> 1); bh = Math.max(1, bh >> 1);
      const t = texture(gl, bw, bh, fmt);
      rt.bloom.push({ t, f: framebuffer(gl, t), w: bw, h: bh });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.rt = rt;
    if (this.topo && !this.finalRadius) this.cam.radius.x = this.cam.radius.target = this._distanceFor(1.6 * this.expectRg + 2);
    else if (this.finalRadius) this.cam.radius.x = this.cam.radius.target = this._distanceFor(this.finalRadius);
  }

  // ---------------------------------------------------------------- frame
  update(dt) {
    const tl = this.timeline;
    if (!tl.frames.length) return null;
    if (this.live) this.clock = performance.now() / 1000 - Math.min(1.0, tl.delay || 0.1);
    else if (this.playing) this.clock += dt * this.speed;
    const s = tl.at(this.clock);
    const gl = this.gl;
    if (s.a !== this.shown.a) { gl.bindBuffer(gl.ARRAY_BUFFER, this.bufA); gl.bufferSubData(gl.ARRAY_BUFFER, 0, s.a.coords); }
    if (s.b !== this.shown.b) { gl.bindBuffer(gl.ARRAY_BUFFER, this.bufB); gl.bufferSubData(gl.ARRAY_BUFFER, 0, s.b.coords); }
    this.shown = s;

    // auto mode: compactness of what is on screen drives the look
    const rg = radiusOfGyration(s.a.coords) * (1 - s.alpha) + radiusOfGyration(s.b.coords) * s.alpha;
    const target = this.finalRg ?? this.expectRg;
    this.compact = target / Math.max(rg, 1e-3);
    // Display contraction: an early noise frame can be ten times wider than the protein, and drawn
    // at true scale it is a few faint points with the rest off screen. Scale it toward its own
    // centroid so the cloud just fills the screen (1.3 Rg ~ where a Gaussian cloud thins out). The
    // shape is untouched, k reaches 1 well before the fold settles, and the final frame is exact.
    const edge = (this.finalRadius ?? 1.6 * this.expectRg) / this.opt.fill;
    const done = s.alpha === 0 && s.a === tl.final;
    this.drawK = done ? 1 : Math.min(1, edge / (1.3 * Math.max(rg, 1e-3)));
    const ca = centroid(s.a.coords), cb = centroid(s.b.coords);
    this.drawC = ca.map((x, k) => x * (1 - s.alpha) + cb[k] * s.alpha);
    this.progress = s.progress;
    if (this.mode === 'auto') {
      const g = done ? 1 : smoothstep(0.62, 0.92, this.compact);
      this.w.grow.target = g;
      this.w.surface.target = g > 0 ? 1 : 0;
      this.w.solid.target = done ? 1 : 0;
      this.w.points.target = done ? 0 : 1;
      this.w.ribbon.target = 0;
    }
    for (const k in this.w) this.w[k].step(dt);
    if (tl.final && !this.finalRg) this.finalRg = radiusOfGyration(tl.final.coords);
    // camera
    this.cam.yaw.target = this.opt.orbitDegPerSec * DEG;
    this.cam.yawAngle += this.cam.yaw.step(dt) * dt;
    this.cam.radius.step(dt);
    this.cam.t += dt;
    const needSurf = this.w.surface.x > 0.002 || this.w.surface.target > 0;
    const needRib = this.w.ribbon.x > 0.002 || this.w.ribbon.target > 0;
    if (needSurf || needRib) {
      const x = interp(s, this._scratch ??= new Float32Array(s.a.coords.length));
      if (needSurf) this._requestMesh(x.slice(), this.w.grow.x);
      if (needRib) {
        const nv = buildRibbon(x, this.rib, this.resColors, this.ribData);
        const gl2 = this.gl;
        gl2.bindVertexArray(null);
        gl2.bindBuffer(gl2.ARRAY_BUFFER, this.ribBufs[0]); gl2.bufferData(gl2.ARRAY_BUFFER, this.ribData.pos.subarray(0, nv * 3), gl2.DYNAMIC_DRAW);
        gl2.bindBuffer(gl2.ARRAY_BUFFER, this.ribBufs[1]); gl2.bufferData(gl2.ARRAY_BUFFER, this.ribData.nrm.subarray(0, nv * 3), gl2.DYNAMIC_DRAW);
        gl2.bindBuffer(gl2.ARRAY_BUFFER, this.ribBufs[2]); gl2.bufferData(gl2.ARRAY_BUFFER, this.ribData.col.subarray(0, nv * 4), gl2.DYNAMIC_DRAW);
      }
    }
    return s;
  }

  view() {
    const c = this.frameC, B = this.basis, R = this.cam.radius.x;
    const elev = this.cam.elev + 4 * DEG * Math.sin(this.cam.t * 2 * Math.PI / 53);
    const yaw = this.cam.yawAngle;
    // spherical in the structure's principal frame: B columns are (long, mid, short); short is up
    const lx = Math.cos(elev) * Math.sin(yaw), ly = Math.sin(elev), lz = Math.cos(elev) * Math.cos(yaw);
    const dir = [0, 1, 2].map(k => B[k] * lx + B[6 + k] * ly + B[3 + k] * lz);
    const eye = [c[0] + dir[0] * R, c[1] + dir[1] * R, c[2] + dir[2] * R];
    const up = [B[6], B[7], B[8]];
    const aspect = this.canvas.width / this.canvas.height;
    const proj = perspective(FOV, aspect, R * 0.05, R * 40);
    proj[8] = -this.opt.offset[0] * 2; proj[9] = -this.opt.offset[1] * 2;  // lens shift
    return { view: lookAt(eye, c, up), proj, dist: R };
  }

  // Camera proof: fraction of displayed atoms inside the viewport, and the camera distance.
  onScreen() {
    const s = this.shown; if (!s?.a) return null;
    const x = interp(s, new Float32Array(s.a.coords.length)), v = this.view();
    const k = this.drawK ?? 1, C = this.drawC ?? this.frameC;
    for (let i = 0; i < x.length; i++) x[i] = C[i % 3] + (x[i] - C[i % 3]) * k;
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

  render(dt) {
    const t0 = performance.now();
    const gl0 = this.gl, T = this.timer;
    if (T && this._q && gl0.getQueryParameter(this._q, gl0.QUERY_RESULT_AVAILABLE)) {
      if (!gl0.getParameter(T.GPU_DISJOINT_EXT)) this.stats.gpuMs.push(gl0.getQueryParameter(this._q, gl0.QUERY_RESULT) / 1e6);
      if (this.stats.gpuMs.length > 600) this.stats.gpuMs.shift();
      gl0.deleteQuery(this._q); this._q = null;
    }
    const timing = T && !this._q;
    if (timing) { this._q = gl0.createQuery(); gl0.beginQuery(T.TIME_ELAPSED_EXT, this._q); }
    const s = this.update(dt);
    const gl = this.gl, rt = this.rt;
    gl.bindFramebuffer(gl.FRAMEBUFFER, rt.msFbo);
    gl.viewport(0, 0, rt.w, rt.h);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    const bg = this.prog.bg;
    gl.useProgram(bg.p);
    gl.uniform2f(bg.u.uRes, rt.w, rt.h);
    gl.uniform2f(bg.u.uCenter, 0.5 + this.opt.offset[0], 0.5 + this.opt.offset[1]);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (s) {
      const v = this.view();
      const W = this.w;
      gl.enable(gl.DEPTH_TEST);
      // ribbon: opaque inside
      if (W.ribbon.x > 0.002 && this.ribCount) {
        this._drawMesh(v, this.vaoRib, this.ribCount, { opacity: W.ribbon.x, solid: 1, emissive: 0.05 });
      }
      // points: additive light, tested against the ribbon, not written to depth
      const keep = this.sure ? 1 - W.points.x : 0;   // unsure atoms stay lit as the others fade
      if ((W.points.x > 0.002 || keep > 0.002) && !this.opt.skip.includes('points')) {
        const p = this.prog.points;
        gl.useProgram(p.p);
        gl.depthMask(false);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE);
        gl.uniformMatrix4fv(p.u.uView, false, v.view);
        gl.uniformMatrix4fv(p.u.uProj, false, v.proj);
        gl.uniform1f(p.u.uAlpha, s.alpha);
        gl.uniform1f(p.u.uProjScale, rt.h / (2 * Math.tan(FOV / 2)));
        gl.uniform1f(p.u.uFocus, v.dist);
        gl.uniform1f(p.u.uAperture, this.opt.aperture);
        gl.uniform1f(p.u.uSlab, this.finalRadius ?? 1.6 * this.expectRg);
        gl.uniform1f(p.u.uRadius, this.opt.pointRadius);
        gl.uniform1f(p.u.uMaxPx, rt.h * 0.03);
        const settle = 0.55 + 0.45 * smoothstep(0.3, 1.0, this.compact);
        gl.uniform1f(p.u.uGain, 3.2 * W.points.x * settle * (1 - 0.55 * W.surface.x * smoothstep(0.02, 0.45, W.grow.x)));
        gl.uniform1f(p.u.uKeep, 3.2 * settle * keep);
        gl.uniform1f(p.u.uNear, v.dist * 0.12);
        gl.uniform1f(p.u.uK, this.drawK ?? 1);
        gl.uniform3fv(p.u.uC, this.drawC ?? this.frameC);
        gl.bindVertexArray(this.vaoPoints);
        gl.drawArrays(gl.POINTS, 0, this.topo.natom);
        gl.disable(gl.BLEND);
        gl.depthMask(true);
      }
      // surface: depth prepass then one blended layer, so glass shows only its front face
      if (W.surface.x > 0.002 && this.surf.ntri && !this.opt.skip.includes('surface')) {
        const op = W.surface.x * smoothstep(0.02, 0.45, W.grow.x);
        if (op > 0.002) {
          gl.colorMask(false, false, false, false);
          this._drawMesh(v, this.vaoSurf, this.surf.ntri * 3, { opacity: 1, solid: 1, emissive: 0 }, true);
          gl.colorMask(true, true, true, true);
          gl.depthFunc(gl.LEQUAL);
          gl.depthMask(false);
          this._drawMesh(v, this.vaoSurf, this.surf.ntri * 3, { opacity: op, solid: W.solid.x * smoothstep(0.6, 1, W.grow.x), emissive: 0 }, true);
          gl.depthMask(true);
          gl.depthFunc(gl.LESS);
        }
      }
      gl.disable(gl.DEPTH_TEST);
    }
    // resolve
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, rt.msFbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, rt.resolve);
    gl.blitFramebuffer(0, 0, rt.w, rt.h, 0, 0, rt.w, rt.h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    if (!this.opt.skip.includes('bloom')) this._bloom();
    // composite to screen
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, rt.W, rt.H);
    const c = this.prog.composite;
    gl.useProgram(c.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, rt.color);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, rt.bloom[0].t);
    gl.uniform1i(c.u.uScene, 0); gl.uniform1i(c.u.uBloom, 1);
    gl.uniform2f(c.u.uRes, rt.W, rt.H);
    gl.uniform1f(c.u.uBloomK, this.opt.bloom);
    gl.uniform1f(c.u.uExposure, this.opt.exposure);
    gl.uniform1f(c.u.uSeed, (this.stats.frames % 64) * 0.618);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (timing) gl.endQuery(T.TIME_ELAPSED_EXT);
    this.stats.frames++;
    this.stats.cpuMs = performance.now() - t0;
  }

  _drawMesh(v, vao, count, m, blended) {
    const gl = this.gl, p = this.prog.surface;
    gl.useProgram(p.p);
    gl.uniformMatrix4fv(p.u.uView, false, v.view);
    gl.uniformMatrix4fv(p.u.uProj, false, v.proj);
    gl.uniform1f(p.u.uOpacity, m.opacity);
    gl.uniform1f(p.u.uSolid, m.solid);
    gl.uniform1f(p.u.uEmissive, m.emissive);
    if (blended || m.opacity < 1) { gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA); }
    gl.bindVertexArray(vao);
    gl.drawElements(gl.TRIANGLES, count, gl.UNSIGNED_INT, 0);
    gl.disable(gl.BLEND);
  }

  _bloom() {
    const gl = this.gl, rt = this.rt, d = this.prog.down, u = this.prog.up;
    gl.bindVertexArray(this.emptyVao);
    gl.useProgram(d.p);
    gl.uniform1i(d.u.uSrc, 0);
    gl.activeTexture(gl.TEXTURE0);
    let src = rt.color, sw = rt.w, sh = rt.h;
    for (const L of rt.bloom) {
      gl.uniform1f(d.u.uKnee, L === rt.bloom[0] ? 0.9 : 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, L.f);
      gl.viewport(0, 0, L.w, L.h);
      gl.bindTexture(gl.TEXTURE_2D, src);
      const k = sw / L.w > 2.5 ? 2 : 1;  // a 4x first step widens the taps to cover its footprint
      gl.uniform2f(d.u.uTexel, k / sw, k / sh);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      src = L.t; sw = L.w; sh = L.h;
    }
    gl.useProgram(u.p);
    gl.uniform1i(u.u.uSrc, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = rt.bloom.length - 1; i > 0; i--) {
      const S = rt.bloom[i], D = rt.bloom[i - 1];
      gl.bindFramebuffer(gl.FRAMEBUFFER, D.f);
      gl.viewport(0, 0, D.w, D.h);
      gl.bindTexture(gl.TEXTURE_2D, S.t);
      gl.uniform2f(u.u.uTexel, 1 / S.w, 1 / S.h);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    gl.disable(gl.BLEND);
  }
}

function smoothstep(a, b, x) { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

function interp(s, out) {
  const A = s.a.coords, B = s.b.coords, al = s.alpha;
  if (al === 0) { out.set(A); return out; }
  for (let i = 0; i < A.length; i++) out[i] = A[i] + (B[i] - A[i]) * al;
  return out;
}

function radiusOfGyration(x) {
  const c = centroid(x), n = x.length / 3;
  let s = 0;
  for (let i = 0; i < n; i++) s += (x[3 * i] - c[0]) ** 2 + (x[3 * i + 1] - c[1]) ** 2 + (x[3 * i + 2] - c[2]) ** 2;
  return Math.sqrt(s / n);
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

// The engine's stream (demo/sc26/PROTOCOL.md) turned into finished folds and chip state.
//
// A fold is collected from fold_start, its frames and fold_done, and handed to onFold only once it
// is complete, so the stage never starts something it cannot finish. Live and replayed folds take
// the same path; `source` says which.

import { decodeCoords } from '../render/src/protocol.js';

const THREE = {
  A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE', G: 'GLY', H: 'HIS', I: 'ILE', K: 'LYS', L: 'LEU',
  M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG', S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
};

export class Stream {
  constructor(url, { onFold, onChips, onStage, onReply, onOpen, as } = {}) {
    this.url = url;
    this.cb = { onFold, onChips, onStage, onReply, onOpen };
    this.as = as;            // ?as=<model>: rehearse a model's screens on another model's stream
    this.open = {};          // id -> fold being collected
    this.chips = [];         // last status.chips
    this.jobs = {};          // job id -> {name, kind, model, chip, n, stage, step, total, t0, tChip, tAt}
                             // tChip: the chip's seconds on this job at its last event, tAt: when that arrived
    this.connected = false;
    this.lastMessage = 0;
    this._connect();
  }

  _connect() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => { this.connected = true; this.cb.onOpen?.(); };
    ws.onmessage = (e) => { this.lastMessage = performance.now(); this._on(JSON.parse(e.data)); };
    ws.onclose = () => { this.connected = false; setTimeout(() => this._connect(), 1500); };
    ws.onerror = () => ws.close();
  }

  send(msg) {
    if (this.ws?.readyState !== WebSocket.OPEN) return false;
    this.ws.send(JSON.stringify(msg));
    return true;
  }

  // the chip's clock for this job: the last time it reported, and when each stage began
  _tick(j, m, now) {
    if (typeof m.t !== 'number') return;
    j.tChip = m.t; j.tAt = now;
    (j.at ??= {})[j.stage] ??= m.t;
  }

  _on(m) {
    const now = performance.now() / 1000;
    if (this.as) { if (m.model) m.model = this.as; if (m.models) m.models = [this.as]; }
    switch (m.type) {
      case 'hello': case 'status':
        this.chips = m.chips ?? [];
        if (m.models) this.models = m.models;
        // A fold that started before this page connected: the engine says what it is and when the chip
        // took it, on the engine's own clock, so the elapsed time needs no clock shared with this browser.
        for (const c of this.chips) {
          const d = c.doing;
          if (d && !this.jobs[d.id] && m.t_wall) this.jobs[d.id] = { name: d.name ?? null, kind: d.kind, model: this.models?.[0] ?? null,
            chip: c.chip, n: d.n_res, stage: null, step: 0, total: 1, t0: now, tChip: Math.max(0, m.t_wall - d.t_wall), tAt: now };
        }
        this.cb.onChips?.(this.chips);
        break;
      case 'chip': {
        const c = this.chips.find(c => c.chip === m.chip);
        if (c) Object.assign(c, m, { warming: m.state === 'warming' && m.name ? { name: m.name, n_res: m.n_res, stage: m.stage } : null });
        this.cb.onChips?.(this.chips);
        break;
      }
      case 'fold_start':
        this.open[m.id] = { start: m, frames: [] };
        // no stage until the chip names one: the fold's first stage is the model's, not ours to guess
        this.jobs[m.id] = { name: m.name ?? null, kind: m.kind, model: m.model ?? null, chip: m.chip, n: m.n_res,
          stage: null, step: 0, total: 1, t0: now, tChip: m.t ?? 0, tAt: now };
        // the server marks a chip busy without a 'chip' message (only 'ready' after each fold), so a
        // fold_start on a chip is what says it is busy and with which job
        { const c = this.chips.find(c => c.chip === m.chip);
          if (c && m.chip != null) { c.state = 'busy'; c.job = m.id; this.cb.onChips?.(this.chips); } }
        this.cb.onStage?.(m.id, this.jobs[m.id], m);
        break;
      case 'stage': {
        const j = this.jobs[m.id];
        if (j) { Object.assign(j, { stage: m.stage, step: m.step ?? 0, total: m.total ?? 1 }); this._tick(j, m, now); this.cb.onStage?.(m.id, j, m); }
        break;
      }
      case 'frame': {
        const f = this.open[m.id];
        if (f) f.frames.push(m);
        const j = this.jobs[m.id];
        if (j) { Object.assign(j, { stage: 'diffusion', step: m.step + 1, total: m.of }); this._tick(j, m, now); j.tFirst ??= m.t; this.cb.onStage?.(m.id, j, m); }
        break;
      }
      case 'fold_done': {
        const f = this.open[m.id];
        delete this.open[m.id];
        const j = this.jobs[m.id];
        if (j) { j.stage = 'done'; j.seconds = m.seconds; j.tDone = now; this.cb.onStage?.(m.id, j, m); }
        // the chip's last finished fold, measured; the status every 2 s says the same, this is sooner
        { const c = this.chips.find(c => c.chip === m.chip);
          if (c && m.chip != null && m.source === 'live') c.last_fold = { name: m.name ?? null, n_res: m.n_res, seconds: m.seconds }; }
        if (f && f.frames.length) this.cb.onFold?.(assemble(f.start, f.frames, m));
        break;
      }
      case 'fold_error':
        delete this.open[m.id];
        if (this.jobs[m.id]) { this.jobs[m.id].stage = 'dropped'; this.cb.onStage?.(m.id, this.jobs[m.id], m); }
        break;
      case 'queued': case 'rejected':
        this.cb.onReply?.(m);
        break;
    }
    // forget jobs nobody will ask about again
    for (const id in this.jobs) if (now - this.jobs[id].t0 > 600) delete this.jobs[id];
  }
}

// A recording (one protocol message per line, PROTOCOL.md "Replay") as one finished fold, marked
// as a recording the way the server marks the replays it plays.
export async function loadRecording(url) {
  const lines = (await (await fetch(url)).text()).split('\n').filter(Boolean).map(l => JSON.parse(l));
  const start = lines.find(m => m.type === 'fold_start'), done = lines.find(m => m.type === 'fold_done');
  const frames = lines.filter(m => m.type === 'frame');
  if (!start || !done || !frames.length) return null;
  const f = assemble({ ...start, kind: 'replay', source: 'replay', chip: null }, frames, done);
  f.received = -1;
  return f;
}

// One finished fold, in the renderer's terms plus everything the words on screen need.
function assemble(start, frames, done) {
  frames.sort((a, b) => a.step - b.step);
  const seq = start.sequence.replace(/:/g, '');   // a complex's chains arrive joined by ':'; atoms number residues across them
  const resName = [...seq].map(c => THREE[c] ?? 'UNK');
  const plddt = done.plddt ?? null;
  const topo = {
    natom: start.atoms.name.length, nres: seq.length,
    atomName: start.atoms.name,
    element: start.atoms.element.map(e => e.toUpperCase()),
    atomResidue: Int32Array.from(start.atoms.residue),
    resName, chain: new Array(seq.length).fill('A'),
    confidence: plddt ? Float32Array.from(plddt) : null,
  };
  // The last frame is the scored structure; fold_done.xyz is the same numbers.
  const coords = frames.map(f => decodeCoords(f.xyz));
  const x0 = frames.map(f => f.x0 ? decodeCoords(f.x0) : null);   // the step's denoised estimate, for alignment
  const steps = frames.map(f => f.step);
  const tReal = frames.map(f => f.t);
  const diffusion = done.stages?.diffusion ?? (tReal[tReal.length - 1] - tReal[0]);
  const mean = plddt ? plddt.reduce((a, b) => a + b, 0) / plddt.length : null;
  return {
    id: start.id, kind: start.kind, source: start.source ?? done.source ?? 'live', model: start.model,
    chip: start.chip ?? null, recorded: start.recorded ?? null,
    name: start.name ?? null, story: start.story ?? null, sequence: seq,
    nres: start.n_res ?? seq.replace(/:/g, '').length, chains: start.chains?.length ?? 1,
    seconds: done.seconds, diffusionSeconds: diffusion, aiclk: done.aiclk_mhz?.median ?? null,
    plddtMean: mean, topo, coords, x0, steps, of: frames[frames.length - 1].of ?? frames.length - 1,
    tReal, received: performance.now(),
  };
}

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
  constructor(url, { onFold, onChips, onStage, onReply, onOpen } = {}) {
    this.url = url;
    this.cb = { onFold, onChips, onStage, onReply, onOpen };
    this.open = {};          // id -> fold being collected
    this.chips = [];         // last status.chips
    this.jobs = {};          // job id -> {name, stage, step, total, t0}
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

  _on(m) {
    const now = performance.now() / 1000;
    switch (m.type) {
      case 'hello': case 'status':
        this.chips = m.chips ?? [];
        this.cb.onChips?.(this.chips);
        break;
      case 'chip': {
        const c = this.chips.find(c => c.chip === m.chip);
        if (c) Object.assign(c, m);
        this.cb.onChips?.(this.chips);
        break;
      }
      case 'fold_start':
        this.open[m.id] = { start: m, frames: [] };
        this.jobs[m.id] = { name: m.name ?? null, kind: m.kind, chip: m.chip, stage: 'lm', step: 0, total: 1, t0: now };
        this.cb.onStage?.(m.id, this.jobs[m.id], m);
        break;
      case 'stage': {
        const j = this.jobs[m.id];
        if (j) { Object.assign(j, { stage: m.stage, step: m.step ?? 0, total: m.total ?? 1 }); this.cb.onStage?.(m.id, j, m); }
        break;
      }
      case 'frame': {
        const f = this.open[m.id];
        if (f) f.frames.push(m);
        const j = this.jobs[m.id];
        if (j) { Object.assign(j, { stage: 'diffusion', step: m.step + 1, total: m.of }); this.cb.onStage?.(m.id, j, m); }
        break;
      }
      case 'fold_done': {
        const f = this.open[m.id];
        delete this.open[m.id];
        const j = this.jobs[m.id];
        if (j) { j.stage = 'done'; j.seconds = m.seconds; this.cb.onStage?.(m.id, j, m); }
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
  const seq = start.sequence;
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
  const tReal = frames.map(f => f.t);
  const diffusion = done.stages?.diffusion ?? (tReal[tReal.length - 1] - tReal[0]);
  const mean = plddt ? plddt.reduce((a, b) => a + b, 0) / plddt.length : null;
  return {
    id: start.id, kind: start.kind, source: start.source ?? done.source ?? 'live', model: start.model,
    chip: start.chip ?? null, recorded: start.recorded ?? null,
    name: start.name ?? null, story: start.story ?? null, sequence: seq, nres: seq.length,
    seconds: done.seconds, diffusionSeconds: diffusion, aiclk: done.aiclk_mhz?.median ?? null,
    plddtMean: mean, topo, coords, tReal, received: performance.now(),
  };
}

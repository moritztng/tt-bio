// The engine's stream (demo/sc26/PROTOCOL.md) turned into finished folds and chip state.
//
// The stream says what every chip is doing and when a fold finishes; it carries no coordinates. A
// finished fold the page wants is pulled from GET /fold/<id> and handed to onFold complete, so the
// stage never starts something it cannot finish. Live and replayed folds take the same path;
// `source` says which.
//
// The page pulls one fold at a time, and only what would change the stage: a protein it has not
// got, a live fold of one it only has as a recording, or a live fold newer than REFRESH_S. A newer
// fold of a protein replaces an older one still waiting, so a slow link skips folds instead of
// falling behind. Over a slow link it asks for every k-th sampler state, k chosen from the
// throughput it measured on its last pulls so a fold arrives within BUDGET_S; the states it shows
// are still real ones, with their real step numbers. A socket silent for SILENT_MS (the engine
// sends a status every 2 s) is replaced by a new one, without a reload.

const SILENT_MS = 6000;   // the engine sends a status every 2 s
const REFRESH_S = 120;    // a live fold of a protein the page already holds live is pulled again after this
const BUDGET_S = 6;       // a pull should take about this long; over a slower link, fewer states
const MAX_EVERY = 8;      // and never fewer than every 8th state (26 of 201 for a 200-step fold)

const THREE = {
  A: 'ALA', C: 'CYS', D: 'ASP', E: 'GLU', F: 'PHE', G: 'GLY', H: 'HIS', I: 'ILE', K: 'LYS', L: 'LEU',
  M: 'MET', N: 'ASN', P: 'PRO', Q: 'GLN', R: 'ARG', S: 'SER', T: 'THR', V: 'VAL', W: 'TRP', Y: 'TYR',
};

export class Stream {
  constructor(url, { onFold, onChips, onStage, onReply, onOpen, as } = {}) {
    this.url = url;
    this.cb = { onFold, onChips, onStage, onReply, onOpen };
    this.as = as;            // ?as=<model>: rehearse a model's screens on another model's stream
    this.base = new URL(url.replace(/^ws/, 'http')).origin;
    this.wanted = new Map(); // protein -> the newest finished fold of it the page may pull
    this.have = new Map();   // protein -> {source, recorded, at} of the copy the page holds
    this.rate = 1e6;         // bytes/s the last pulls achieved; a first guess until one is measured
    this.fetching = null;
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
    const opened = performance.now();
    ws.onopen = () => { this.connected = true; this.lastMessage = performance.now(); this.openedAt = this.lastMessage / 1000; this.cb.onOpen?.(); };
    ws.onmessage = (e) => { this.lastMessage = performance.now(); this._on(JSON.parse(e.data)); };
    ws.onclose = () => { if (this.ws === ws) { this.connected = false; setTimeout(() => this._connect(), 1500); } };
    ws.onerror = () => ws.close();
    // A link that drops silently can leave a socket open for minutes with nothing arriving: give up
    // on it after SILENT_MS and open another. The stage keeps showing what it has meanwhile.
    clearInterval(this.watch);
    this.watch = setInterval(() => {
      if (this.ws !== ws) return clearInterval(this.watch);
      const since = performance.now() - (ws.readyState === WebSocket.OPEN ? this.lastMessage : opened);
      if (since > SILENT_MS) { this.ws = null; this.connected = false; ws.close(); this._connect(); }
    }, 1000);
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
      case 'hello': case 'status': {
        // a status built just before a fold landed can arrive just after its fold_done: keep the newer last fold
        const had = new Map(this.chips.map(c => [c.chip, c.last_fold]));
        this.chips = m.chips ?? [];
        for (const c of this.chips) {
          const h = had.get(c.chip);
          if (h?.t_wall && !(c.last_fold?.t_wall >= h.t_wall)) c.last_fold = h;
        }
        if (m.models) this.models = m.models;
        if (m.type === 'hello') for (const f of m.folds ?? []) this._offer(f, m.t_wall);
        // A fold that started before this page connected: the engine says what it is and when the chip
        // took it, on the engine's own clock, so the elapsed time needs no clock shared with this browser.
        for (const c of this.chips) {
          const d = c.doing;
          if (d && !this.jobs[d.id] && m.t_wall) this.jobs[d.id] = { name: d.name ?? null, kind: d.kind, model: this.models?.[0] ?? null,
            chip: c.chip, n: d.n_res, stage: null, step: 0, total: 1, t0: now, tChip: Math.max(0, m.t_wall - d.t_wall), tAt: now };
        }
        this.cb.onChips?.(this.chips);
        break;
      }
      case 'chip': {
        const c = this.chips.find(c => c.chip === m.chip);
        if (c) Object.assign(c, m, { warming: m.state === 'warming' && m.name ? { name: m.name, n_res: m.n_res, stage: m.stage } : null });
        this.cb.onChips?.(this.chips);
        break;
      }
      case 'fold_start':
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
        const j = this.jobs[m.id];
        if (j) { Object.assign(j, { stage: 'diffusion', step: m.step + 1, total: m.of }); this._tick(j, m, now); j.tFirst ??= m.t; this.cb.onStage?.(m.id, j, m); }
        break;
      }
      case 'fold_done': {
        const j = this.jobs[m.id];
        if (j) { j.stage = 'done'; j.seconds = m.seconds; j.tDone = now; this.cb.onStage?.(m.id, j, m); }
        // the chip's last finished fold, measured; the status every 2 s says the same, this is sooner
        { const c = this.chips.find(c => c.chip === m.chip);
          if (c && m.chip != null && m.source === 'live') c.last_fold = { name: m.name ?? null, n_res: m.n_res, seconds: m.seconds, t_wall: m.t_wall }; }
        this._offer(m, m.t_wall);
        break;
      }
      case 'fold_error':
        if (this.jobs[m.id]) { this.jobs[m.id].stage = 'dropped'; this.cb.onStage?.(m.id, this.jobs[m.id], m); }
        break;
      case 'queued': case 'rejected':
        this.cb.onReply?.(m);
        break;
    }
    // forget jobs nobody will ask about again
    for (const id in this.jobs) if (now - this.jobs[id].t0 > 600) delete this.jobs[id];
  }

  // A finished fold the page could pull. `tWall` is the engine's clock when the summary was sent,
  // so `received` is when the chip finished it on this page's clock (the words "N min ago").
  _offer(m, tWall) {
    const k = m.name ?? m.sequence, had = this.have.get(k);
    m.received = performance.now() - 1000 * Math.max(0, (tWall ?? m.t_wall) - m.t_wall);
    if (m.kind !== 'visitor' && had && (m.source !== 'live'
      ? had.source === 'live' || had.recorded === m.recorded
      : had.source === 'live' && m.received - had.at < 1000 * REFRESH_S)) return;
    this.wanted.set(k, m);
    this._pump();
  }

  // Pull the most useful wanted fold, one at a time: a visitor's first, then a protein the page has
  // not got, then a live fold of one it holds only as a recording, then a refresh; newest first.
  async _pump() {
    if (this.fetching || !this.wanted.size) return;
    const rank = (m) => {
      const had = this.have.get(m.name ?? m.sequence);
      return m.kind === 'visitor' ? 0 : !had ? 1 : had.source !== 'live' ? 2 : 3;
    };
    const m = [...this.wanted.values()].sort((a, b) => rank(a) - rank(b) || b.received - a.received)[0];
    const k = m.name ?? m.sequence;
    this.wanted.delete(k);
    const full = m.n_atoms * (6 * (m.n_frames - 1) + 12);
    const every = Math.min(MAX_EVERY, Math.max(1, Math.ceil(full / (this.rate * BUDGET_S))));
    const ctl = new AbortController();
    const expect = full / every / this.rate;
    const timer = setTimeout(() => ctl.abort(), 1000 * Math.max(15, 4 * expect));
    this.fetching = m;
    try {
      const t0 = performance.now();
      const res = await fetch(`${this.base}/fold/${encodeURIComponent(m.id)}?every=${every}`, { signal: ctl.signal });
      if (res.ok) {
        const t1 = performance.now(), buf = await res.arrayBuffer(), dt = (performance.now() - t1) / 1000;
        if (buf.byteLength > 64e3) this.rate = 0.5 * this.rate + 0.5 * buf.byteLength / Math.max(0.05, dt);
        this.pulled = { bytes: buf.byteLength, every, seconds: (performance.now() - t0) / 1000, rate: this.rate };
        const f = await fromBody(buf, m);
        this.have.set(k, { source: f.source, recorded: f.recorded, at: m.received });
        this.cb.onFold?.(f);
      }
    } catch (e) {
      // the link dropped or crawled: assume half the speed, and try this protein again shortly
      this.rate /= 2;
      if (!this.wanted.has(k)) this.wanted.set(k, m);
    } finally {
      clearTimeout(timer);
      this.fetching = null;
      setTimeout(() => this._pump(), this.wanted.size ? 200 : 0);
    }
  }
}

// A recording (one protocol message per line, PROTOCOL.md "Replay") as one finished fold, marked
// as a recording the way the server marks the replays it plays.
export async function loadRecording(url) {
  const lines = (await (await fetch(url)).text()).split('\n').filter(Boolean).map(l => JSON.parse(l));
  const start = lines.find(m => m.type === 'fold_start'), done = lines.find(m => m.type === 'fold_done');
  if (!start || !done?.frames) return null;
  const fr = done.frames, b64 = (s) => Uint8Array.from(atob(s), c => c.charCodeAt(0)).buffer;
  const meta = { ...start, ...done, kind: 'replay', source: 'replay', chip: null, every: 1,
    step: fr.step, t: fr.t, origin: fr.origin, scale: fr.scale };
  const f = assemble(meta, states(meta, new Int16Array(b64(fr.q16)), new Float32Array(b64(done.xyz))));
  f.received = -1;
  return f;
}

// GET /fold/<id> (PROTOCOL.md): a u32 length, the metadata as gzipped JSON padded to 4 bytes, the
// final structure as float32, then the packed states as int16.
async function fromBody(buf, summary) {
  const ml = new DataView(buf).getUint32(0, true);
  const json = new Blob([new Uint8Array(buf, 4, ml)]).stream().pipeThrough(new DecompressionStream('gzip'));
  const meta = JSON.parse(await new Response(json).text());
  const n = meta.n_atoms, at = 4 + ml + (-(4 + ml) & 3);
  const f = assemble({ ...meta, model: summary.model ?? meta.model },
    states(meta, new Int16Array(buf, at + 12 * n), new Float32Array(buf.slice(at, at + 12 * n))));
  f.received = summary.received;
  return f;
}

// The packed states back to float32 Angstrom: state i is q * scale[i] + origin[i]. The last state is
// the final structure, exact.
function states(meta, q, final) {
  const n3 = final.length;
  return [...meta.scale.map((s, i) => {
    const o = meta.origin[i], x = new Float32Array(n3);
    for (let j = 0; j < n3; j += 3) {
      x[j] = q[i * n3 + j] * s + o[0]; x[j + 1] = q[i * n3 + j + 1] * s + o[1]; x[j + 2] = q[i * n3 + j + 2] * s + o[2];
    }
    return x;
  }), final];
}

// One finished fold, in the renderer's terms plus everything the words on screen need. Its states
// are already superposed onto the final structure (`aligned`), by the chip worker.
function assemble(m, coords) {
  const seq = m.sequence.replace(/:/g, '');   // a complex's chains arrive joined by ':'; atoms number residues across them
  const resName = [...seq].map(c => THREE[c] ?? 'UNK');
  const plddt = m.plddt ?? null;
  const topo = {
    natom: m.atoms.name.length, nres: seq.length,
    atomName: m.atoms.name,
    element: m.atoms.element.map(e => e.toUpperCase()),
    atomResidue: Int32Array.from(m.atoms.residue),
    resName, chain: new Array(seq.length).fill('A'),
    confidence: plddt ? Float32Array.from(plddt) : null,
  };
  const steps = m.step, tReal = m.t;
  const diffusion = m.stages?.diffusion ?? (tReal[tReal.length - 1] - tReal[0]);
  const mean = plddt ? plddt.reduce((a, b) => a + b, 0) / plddt.length : null;
  return {
    id: m.id, kind: m.kind, source: m.source ?? 'live', model: m.model,
    chip: m.chip ?? null, recorded: m.recorded ?? null,
    name: m.name ?? null, story: m.story ?? null, sequence: seq,
    nres: m.n_res ?? seq.length, chains: m.chains?.length ?? 1,
    seconds: m.seconds, diffusionSeconds: diffusion, aiclk: m.aiclk_mhz?.median ?? null,
    plddtMean: mean, topo, coords, x0: null, aligned: true, steps, of: steps[steps.length - 1] + 1,
    every: m.every ?? 1, tReal, received: performance.now(),
  };
}

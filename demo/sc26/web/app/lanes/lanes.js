// The hardware screens of the SC26 demo: four chip lanes, the measured comparison, and the
// dataflow screen. One module, no dependencies, no network: the app mounts a view and feeds it
// `chips` messages from its stream; index.html does the same from /telemetry for development.
//
//   const view = mount(element, "lanes" | "compare" | "depth", facts);
//   view.update(chipsMessage);

const NS = "http://www.w3.org/2000/svg";
const GRID = [11, 10];          // the Tensix grid each chip of this box gives a program
const BOARDS = [[0, 1], [2, 3]]; // qb2's p300c boards; a board reset takes both of its chips

const h = (tag, attrs = {}, ...kids) => {
  const svg = ["svg", "rect", "line", "polyline", "polygon", "text", "g", "path", "animate", "animateMotion",
    "circle", "defs", "marker"].includes(tag);
  const el = svg ? document.createElementNS(NS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) el.append(kid);
  return el;
};
const fmt = (v, d = 0) => (v == null ? "—" : Number(v).toFixed(d));
const clamp01 = (x) => Math.max(0, Math.min(1, x));

function frame(root, kicker, title, foot) {
  root.replaceChildren();
  const live = h("div", { class: "hw-live" }, h("i"), h("span", { text: "waiting for telemetry" }));
  const wrap = h("div", { class: "hw" },
    h("div", { class: "hw-head" },
      h("div", {}, h("div", { class: "hw-kicker", text: kicker }), h("h1", { class: "hw-title", html: title })),
      live),
  );
  root.append(wrap);
  return { wrap, live, foot: (html) => wrap.append(h("div", { class: "hw-foot", html })) };
}

function setLive(live, msg) {
  const recorded = msg.source !== "live";
  const word = live.parentElement.querySelector(".hw-title b.live");
  if (word) word.hidden = recorded;
  live.classList.toggle("recorded", recorded);
  live.lastChild.textContent = recorded
    ? `${msg.source} snapshot, not live`
    : `live · sampled ${fmt(msg.rate_hz)}× a second`;
}

// ---------------------------------------------------------------------------------------- lanes
function glyph() {
  const [cols, rows] = GRID, s = 6, g = 2.4;
  const svg = h("svg", { class: "glyph", viewBox: `0 0 ${cols * (s + g)} ${rows * (s + g)}`, width: "100%" });
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++)
      svg.append(h("rect", { x: x * (s + g), y: y * (s + g), width: s, height: s, rx: 1.2,
        style: `animation-delay:${(x * 0.17 + y * 0.03).toFixed(2)}s` }));
  return svg;
}

// Power on a fixed 0..max scale, so the four lanes compare at a glance and an idle chip is a low line.
function spark(points, max) {
  if (points.filter((p) => p != null).length < 2) return "";
  const n = points.length - 1;
  const xy = points.map((p, i) => (p == null ? null : [(i / n) * 100, 31 - clamp01(p / max) * 30]))
    .filter(Boolean).map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`);
  return xy.join(" ");
}

function lane(card) {
  const q = {};
  const cell = (cls, ...kids) => h("div", { class: cls }, ...kids);
  const el = h("div", { class: "lane", "data-state": "resetting" },
    cell("chip", q.glyph = glyph()),
    cell("id", q.name = h("div", { class: "name" }), q.state = h("div", { class: "state" })),
    cell("gauge", q.pw = h("div", { class: "big" }),
      q.sp = h("svg", { class: "spark", viewBox: "0 0 100 32", preserveAspectRatio: "none" },
        q.area = h("polygon"), q.line = h("polyline")),
      h("div", { class: "label", text: "Power, last minute" })),
    cell("gauge", q.tmp = h("div", { class: "big" }), h("div", { class: "label", text: "ASIC" })),
    cell("what", q.what = h("div", { class: "name" }), q.detail = h("div", { class: "detail" })),
    cell("today", q.n = h("div", { class: "big" }), h("div", { class: "label", text: "folds today" })),
  );
  return { el, q };
}

const STATE_WORD = { folding: "folding", idle: "ready", busy: "busy", resetting: "no reading", warming: "warming up",
  resting: "resting" };

function paintLane({ el, q }, c) {
  const live = c.state !== "resetting" && c.state !== "resting";
  el.dataset.state = c.state;
  q.state.textContent = c.folding ? "folding" : STATE_WORD[c.state] ?? c.state;
  q.pw.innerHTML = live ? `${fmt(c.power_w)}<small>W</small>` : "—";
  q.tmp.innerHTML = live ? `${fmt(c.temp_c)}<small>°C</small>` : "—";
  const pts = spark(c.power_60s || [], c.power_max_w ?? 125);
  q.line.setAttribute("points", pts);
  q.area.setAttribute("points", pts ? `0,32 ${pts} 100,32` : "");
  const glow = live ? 0.14 + 0.8 * clamp01(((c.power_w ?? 0) - 20) / ((c.power_max_w ?? 125) - 20)) : 0.1;
  for (const r of q.glyph.children) r.style.fillOpacity = glow.toFixed(3);

  const f = c.folding, last = c.last_fold;
  const lastLine = last ? `last: ${last.residues ?? "?"} residues in ${fmt(last.seconds, 1)} s` : "";
  if (f) {
    // no running seconds here: the one clock on screen is the stage's (demo/sc26/README.md)
    q.what.textContent = f.name ?? "A protein";
    q.detail.textContent = `${f.residues ?? "?"} residues`;
  } else if (c.state === "resetting") {
    // telemetry only knows the chip gave no reading (a reset, a hang, or a lent board look the same)
    q.what.textContent = "No reading";
    q.detail.textContent = "clock and heartbeat not answering";
  } else if (c.state === "resting") {
    // the engine rests a chip that keeps hanging; it rejoins by itself at back_at, as the stage says
    const min = Math.max(1, Math.ceil(((c.back_at ?? 0) - Date.now() / 1000) / 60));
    q.what.textContent = "Resting after a hang";
    q.detail.textContent = `back in ${min} min, rejoins by itself`;
  } else if (c.state === "warming") {
    q.what.textContent = "Warming up";
    q.detail.textContent = "compiling for the booth's proteins";
  } else if (c.state === "busy") {
    q.what.textContent = "Working outside the demo";
    q.detail.textContent = lastLine || "clock is up, no demo fold on this chip";
  } else {
    q.what.textContent = "Ready";
    q.detail.textContent = lastLine || "waiting for a sequence";
  }
  q.n.textContent = fmt(c.folds_today);
}

function lanes(root) {
  const f = frame(root, "QuietBox 2 · Tenstorrent Blackhole", "Four chips<b class=\"live\">, live</b>");
  const box = h("div", { class: "box" });
  const ls = {};
  for (const [i, pair] of BOARDS.entries()) {
    const b = h("div", { class: "board" }, h("div", { class: "board-tag", text: `p300c board ${"AB"[i]}` }));
    const col = h("div");
    for (const card of pair) { ls[card] = lane(card); col.append(ls[card].el); }
    b.append(col);
    box.append(b);
  }
  f.wrap.append(box);
  const title = f.wrap.querySelector(".hw-title");
  const foot = h("div", { class: "hw-foot" });
  f.wrap.append(foot);
  return {
    update(msg) {
      setLive(f.live, msg);
      for (const c of msg.chips) if (ls[c.card]) paintLane(ls[c.card], c);
      // a chip taken out of the demo is left off, and the title counts the chips that fold
      const off = new Set(msg.chips.filter((c) => c.state === "out_of_service").map((c) => c.card));
      // and the rest are named by their place, 1, 2, 3, as the stage's rows are (app.js seats)
      let seat = 0;
      for (const [card, l] of Object.entries(ls)) {
        l.el.style.display = off.has(+card) ? "none" : "";
        if (!off.has(+card)) l.q.name.textContent = `Chip ${++seat}`;
      }
      const n = ["No", "One", "Two", "Three", "Four"][Object.keys(ls).length - off.size];
      title.firstChild.textContent = `${n} chips`;
      foot.innerHTML = "Read from the Tenstorrent kernel driver's own counters, no tool polling the chips. " +
        (msg.sample_ms != null ? `One reading of all ${n.toLowerCase()} chips takes ${fmt(msg.sample_ms, 1)} ms.` : "");
    },
  };
}

// -------------------------------------------------------------------------------------- compare
const GPU = [["h200", "H200"], ["b200", "B200"], ["a100", "A100"]];

function compare(root, facts) {
  const f = frame(root, "The honest comparison", "One chip against <b>one GPU</b>");
  const lo = Math.log10(2), hi = Math.log10(100);
  const x = (s) => `${(100 * (Math.log10(s) - lo)) / (hi - lo)}%`;

  const dots = h("div", { class: "dots" });
  for (const r of facts.rows) {
    const all = [r.tt, ...GPU.map(([k]) => r[k])];
    const track = h("div", { class: "track" }, h("div", { class: "rule" }),
      h("div", { class: "link", style: `left:${x(Math.min(...all))};right:calc(100% - ${x(Math.max(...all))})` }));
    for (const [k] of GPU) track.append(h("div", { class: `mk ${k}`, style: `left:${x(r[k])}` }));
    track.append(h("div", { class: "mk tt", style: `left:${x(r.tt)}` }));
    dots.append(h("div", { class: "dot-row" },
      h("div", { class: "m", html: `${r.model}<small>${r.group}</small>` }), track,
      h("div", { class: "v", html: `${fmt(r.tt, 1)}<small> s</small>` })));
  }
  const ticks = h("div", { class: "ticks" });
  for (const t of [2, 5, 10, 20, 50, 100]) ticks.append(h("span", { style: `left:${x(t)}`, text: `${t} s` }));
  dots.append(h("div", { class: "axis" }, h("div"), ticks, h("div")));

  const legend = h("div", { class: "legend" },
    h("span", {}, h("i", { class: "mk tt" }), "one Blackhole chip"),
    ...GPU.map(([k, n]) => h("span", {}, h("i", { class: `mk ${k}` }), n)));

  const slower = facts.rows.filter((r) => r.tt > r.h200).length;
  const [gal, dgx] = facts.servers;
  // Server throughput per dollar, derived the way the benchmarks page derives it: one chip's
  // seconds times the chip count, over list price. Fold rows only, as the page states its range.
  const perDollar = facts.rows.filter((r) => r.group === "fold").map((r) =>
    (gal.accelerators / r.tt / gal.price_usd) / (dgx.accelerators / r.b200 / dgx.price_usd));
  const bc = facts.bindcraft2;
  const now = h("div", { class: "say small" });
  const side = h("div", {},
    h("div", { class: "say", html: `Per chip, a Blackhole is <b>slower than an H200 on ${slower} of these ${facts.rows.length} models.</b>` }),
    h("div", { class: "say", html: `The case is the box. A ${gal.name} puts <b>${gal.accelerators} chips in $${gal.price_usd.toLocaleString("en-US")}</b>, ` +
      `a ${dgx.name} ${dgx.accelerators} GPUs in $${dgx.price_usd.toLocaleString("en-US")}. Per dollar of list price, that is ` +
      `<b>${fmt(Math.min(...perDollar), 1)}× to ${fmt(Math.max(...perDollar), 1)}× the folds.</b>` }),
    h("div", { class: "say small", text: `${bc.model}, one gradient round at ${bc.size}: ${bc.tt} s on ${bc.board} at a ` +
      `${bc.aiclk.replace(", sampled during the rounds", "")}, against ${bc.h200} s on an H200 (${bc.source.split(", ")[1]}).` }),
    h("div", { class: "nowbox" }, h("div", { class: "hw-kicker", text: "On this box, today" }), now));

  f.wrap.append(h("div", { class: "cmp" }, h("div", {}, legend, dots), side));
  f.foot(`Seconds per prediction at 512 residues, warm, one at a time, lower is better, log scale. From ` +
    `${facts.source}, updated ${facts.updated}. ${facts.board.split(". That")[0]}. ` +
    `GPU rows run each model's own upstream code. Per-dollar figures assume ${gal.accelerators} chips do ${gal.accelerators}× the work of one, ` +
    `as the benchmarks page does, and use list prices. The published cells do not record the chip's clock.`);
  return {
    update(msg) {
      setLive(f.live, msg);
      const seats = msg.chips.filter((c) => c.state !== "out_of_service").map((c) => c.card).sort((a, b) => a - b);
      const done = msg.chips.filter((c) => c.last_fold && seats.includes(c.card));
      now.textContent = done.length
        ? done.map((c) => `Chip ${seats.indexOf(c.card) + 1}: ${c.last_fold.model ?? "fold"}, ${c.last_fold.residues} residues in ` +
            `${fmt(c.last_fold.seconds, 1)} s` +
            `, ${c.folds_today} today`).join(". ") + "."
        : "No demo fold has finished on this box yet today. Each one appears here with its time.";
    },
  };
}

// ---------------------------------------------------------------------------------------- depth
function depth(root) {
  const f = frame(root, "Inside one Blackhole chip", "Dataflow, <b>not a cache hierarchy</b>");
  const W = 1600, H = 640, [cols, rows] = GRID, cs = 38, gap = 12, gx = 330, gy = 60;
  const svg = h("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "xMidYMid meet" });

  svg.append(h("defs", {}, (() => {
    const m = h("marker", { id: "hw-arrow", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto" });
    m.append(h("path", { d: "M0 0 L8 4 L0 8 z", fill: "rgba(236,238,244,0.5)" }));
    return m;
  })()));

  // Off-chip memory on the left
  svg.append(h("rect", { class: "mem", x: 40, y: gy, width: 170, height: 490, rx: 10 }));
  svg.append(h("text", { class: "t-ink", x: 60, y: gy + 40, "font-size": 18, "font-weight": 500, text: "GDDR6" }));
  svg.append(h("text", { x: 60, y: gy + 66, "font-size": 15, text: "32 GB" }));
  svg.append(h("text", { x: 60, y: gy + 88, "font-size": 15, text: "512 GB/s" }));
  svg.append(h("text", { x: 40, y: gy - 22, "font-size": 15, text: "off-chip memory" }));

  // Network-on-chip and the Tensix grid
  const gw = cols * (cs + gap) - gap, gh = rows * (cs + gap) - gap;
  for (let y = 0; y < rows; y++) svg.append(h("line", { class: "noc", x1: gx - 30, x2: gx + gw + 10, y1: gy + y * (cs + gap) + cs / 2, y2: gy + y * (cs + gap) + cs / 2 }));
  for (let x = 0; x < cols; x++) svg.append(h("line", { class: "noc", y1: gy - 10, y2: gy + gh + 10, x1: gx + x * (cs + gap) + cs / 2, x2: gx + x * (cs + gap) + cs / 2 }));
  for (const yy of [0.2, 0.5, 0.8]) svg.append(h("line", { class: "hair", x1: 210, x2: gx - 30, y1: gy + gh * yy, y2: gy + gh * yy }));
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++)
      svg.append(h("rect", { class: "core", x: gx + x * (cs + gap), y: gy + y * (cs + gap), width: cs, height: cs, rx: 5 }));
  svg.append(h("text", { x: gx, y: gy - 22, "font-size": 15, text: "110 Tensix cores, two networks-on-chip" }));

  // Tiles in flight: 32x32 blocks moving core to core along the networks
  const lanesY = [1, 4, 7, 9], lanesX = [2, 5, 8];
  lanesY.forEach((r, i) => {
    const y = gy + r * (cs + gap) + cs / 2 - 4;
    const t = h("rect", { class: "tile", width: 11, height: 11, rx: 2, y: y - 1.5 });
    t.append(h("animate", { attributeName: "x", from: 210, to: gx + gw, dur: `${5 + i}s`, begin: `${-i * 1.3}s`, repeatCount: "indefinite" }));
    svg.append(t);
  });
  lanesX.forEach((c, i) => {
    const x = gx + c * (cs + gap) + cs / 2 - 4;
    const t = h("rect", { class: "tile", width: 11, height: 11, rx: 2, x: x - 1.5 });
    t.append(h("animate", { attributeName: "y", from: gy + gh, to: gy - 10, dur: `${6 + i}s`, begin: `${-i * 2.1}s`, repeatCount: "indefinite" }));
    svg.append(t);
  });

  // One core, opened up
  const hc = [8, 3], hx = gx + hc[0] * (cs + gap), hy = gy + hc[1] * (cs + gap);
  svg.append(h("rect", { class: "core", x: hx, y: hy, width: cs, height: cs, rx: 5, style: "stroke:var(--accent);stroke-width:2" }));
  const zx = gx + gw + 70, zy = gy, zw = W - zx - 30, zh = gh;
  svg.append(h("path", { class: "acc-line", d: `M${hx + cs} ${hy} L${zx} ${zy}`, "stroke-opacity": 0.4 }));
  svg.append(h("path", { class: "acc-line", d: `M${hx + cs} ${hy + cs} L${zx} ${zy + zh}`, "stroke-opacity": 0.4 }));
  svg.append(h("rect", { x: zx, y: zy, width: zw, height: zh, rx: 14, style: "fill:rgb(var(--live-rgb) / 0.04);stroke:rgb(var(--live-rgb) / 0.5)" }));
  svg.append(h("text", { x: zx, y: zy - 22, "font-size": 15, text: "one core" }));

  // The core as the pipeline a tile walks through: in, unpack, compute, pack, out, with SRAM between.
  const pad = 26, ux = zx + pad, uw = zw - 2 * pad;
  const unit = (x, y, w, hh, label, sub, cls = "unit") => {
    svg.append(h("rect", { class: cls, x, y, width: w, height: hh, rx: 7 }));
    svg.append(h("text", { class: cls.includes("hot") ? "t-acc" : "t-ink", x: x + 14, y: y + 26, "font-size": 15, "font-weight": 500, text: label }));
    if (sub) svg.append(h("text", { x: x + 14, y: y + 46, "font-size": 12.5, text: sub }));
  };
  const sw = uw, ey = zy + pad, sty = ey + 62 + 70, sy = sty + 60 + 56;
  unit(ux, sy, sw, zy + zh - pad - sy, "SRAM, 1.5 MiB", "scratchpad the program fills, not a cache", "mem");
  const steps = [["Data in", "RISC-V"], ["Unpack", "RISC-V"], ["Math", "RISC-V"], ["Pack", "RISC-V"], ["Data out", "RISC-V"]];
  const sg = 12, stw = (uw - 4 * sg) / 5;
  steps.forEach(([l, sub], i) => unit(ux + i * (stw + sg), sty, stw, 60, l, sub, i === 2 ? "unit hot" : "unit"));
  svg.append(h("text", { x: ux, y: sty - 12, "font-size": 12.5, text: "five small RISC-V cores, one program each" }));
  const ew = (uw - sg) / 2;
  unit(ux, ey, ew, 62, "Matrix engine", "multiplies 32 × 32 tiles");
  unit(ux + ew + sg, ey, ew, 62, "Vector engine", "elementwise, exp, softmax");
  const mx = ux + 2 * (stw + sg) + stw / 2;
  svg.append(h("path", { class: "arrow", d: `M${mx} ${sty} L${mx} ${ey + 66}` }));
  for (let i = 0; i < 4; i++) {
    const x0 = ux + i * (stw + sg) + stw, y0 = sty + 30;
    svg.append(h("path", { class: "arrow", d: `M${x0 + 1} ${y0} L${x0 + sg - 1} ${y0}` }));
  }
  // in and out go through SRAM: data in writes it, unpack reads it, pack writes it, data out reads it
  for (const i of [0, 1, 3, 4]) {
    const x = ux + i * (stw + sg) + stw / 2;
    svg.append(h("path", { class: "arrow", d: i === 0 || i === 3 ? `M${x} ${sty + 60} L${x} ${sy - 4}` : `M${x} ${sy} L${x} ${sty + 64}`,
      "stroke-dasharray": "3 4" }));
  }
  // one tile walking the pipeline
  const walk = `M${ux - 20} ${sty + 30} L${ux + stw / 2} ${sty + 30} L${ux + stw / 2} ${sy + 30} L${ux + 1.5 * stw + sg} ${sy + 30} ` +
    `L${ux + 1.5 * stw + sg} ${sty + 30} L${mx} ${sty + 30} L${mx} ${ey + 30} L${mx} ${sty + 30} L${ux + 3.5 * stw + 3 * sg} ${sty + 30} ` +
    `L${ux + 3.5 * stw + 3 * sg} ${sy + 30} L${ux + 4.5 * stw + 4 * sg} ${sy + 30} L${ux + 4.5 * stw + 4 * sg} ${sty + 30} L${ux + uw + 20} ${sty + 30}`;
  const tile = h("rect", { class: "tile", x: -6, y: -6, width: 12, height: 12, rx: 2 });
  tile.append(h("animateMotion", { path: walk, dur: "7s", repeatCount: "indefinite", calcMode: "linear" }));
  svg.append(tile);

  f.wrap.append(h("div", { class: "depth" }, svg, h("div", { class: "facts", html:
    "<div><b>110</b>Tensix cores on each chip of this box, an 11 × 10 grid.</div>" +
    "<div><b>165 MiB</b>of SRAM, 1.5 MiB beside every core. Software places every tile.</div>" +
    "<div><b>64 MiB</b>the pair tensor of a 512-residue fold in bfloat16: 32,768 tiles of 32 × 32.</div>" +
    "<div><b>5</b>small RISC-V cores in every Tensix core. Two move tiles over the network, three drive the engines.</div>" })));
  return { update: (msg) => setLive(f.live, msg) };
}

export function mount(root, view, facts) {
  return { lanes, compare, depth }[view](root, facts);
}

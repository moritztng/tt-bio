/* Runs both pages in site/ against site/data/perf-512aa.json under a small DOM stub, then
 * asserts every row in the data reached them. Neither page has a build step or a browser test,
 * so a data block the renderer forgets, or a variable left behind by an edit, used to fail only
 * on load. Checks are derived from the data, not hardcoded, so a new model or a new category is
 * covered the moment it lands.
 *
 * Exit 0 = every row drew. Run from the repo root: node site/benchmarks/render_check.js */
const fs = require("fs");

const raw = fs.readFileSync("site/data/perf-512aa.json", "utf8");
const D = JSON.parse(raw);

function mkEl(tag) {
  const el = {
    tagName: tag, children: [], _text: "", _html: "", style: {}, dataset: {},
    classList: { add() {}, remove() {}, contains() { return false; } },
    setAttribute() {}, getAttribute() { return null; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener() {}, querySelector() { return mkEl("div"); }, querySelectorAll() { return []; },
    getBoundingClientRect() { return { width: 900, height: 300 }; },
    clientWidth: 900, insertAdjacentHTML() {}, remove() {},
  };
  Object.defineProperty(el, "textContent",
    { get() { return this._text; }, set(v) { this._text = String(v); } });
  Object.defineProperty(el, "innerHTML",
    { get() { return this._html; }, set(v) { this._html = String(v); } });
  return el;
}

/* The page does fetch(...).then(json).then(render).catch(report); a thenable that resolves
 * synchronously runs the same path and surfaces a render throw as a throw. */
function fetchStub() {
  return {
    _v: { json: () => JSON.parse(raw) },
    then(f) { this._v = f(this._v); return this; },
    catch() { return this; },
  };
}

/* Both pages in site/ read this file and render it with their own script, so the runner is
 * shared and each page gets its own store of drawn nodes. */
function runPage(file) {
  const html = fs.readFileSync(file, "utf8");
  const script = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];
  const markupIds = new Set();
  for (const m of html.matchAll(/id="([^"]+)"/g)) markupIds.add(m[1]);
  const store = new Map();
  const unknownIds = new Set();
  const document = {
    createElement: mkEl,
    createElementNS: (ns, t) => mkEl(t),
    createTextNode: (t) => ({ nodeType: 3, textContent: String(t), children: [] }),
    addEventListener() {},
    getElementById(id) {
      if (!store.has(id)) {
        if (!markupIds.has(id)) unknownIds.add(id);
        store.set(id, mkEl("div"));
      }
      return store.get(id);
    },
    querySelector() { return mkEl("div"); }, querySelectorAll() { return []; }, body: mkEl("body"),
  };
  const window = { addEventListener() {}, innerWidth: 1200, devicePixelRatio: 1 };
  /* The landing page reveals sections on scroll; observe-and-never-fire is the right stub, the
   * elements it watches are already in the markup. */
  const IntersectionObserver = class { observe() {} unobserve() {} disconnect() {} };
  try {
    new Function("document", "window", "fetch", "setTimeout", "clearTimeout", "console",
                 "IntersectionObserver", script)(
      document, window, fetchStub, (f) => f, () => {}, console, IntersectionObserver);
  } catch (e) {
    console.error(file + " threw while rendering: " + (e && e.stack || e));
    process.exit(1);
  }
  if (unknownIds.size) {
    console.error(file + " reads ids that are not in the markup: " + [...unknownIds].join(", "));
    process.exit(1);
  }
  return { html, script, store };
}

const page = runPage("site/benchmarks/index.html");
const script = page.script, store = page.store;

function deepText(el) {
  if (!el) return "";
  let out = String(el._text || "") + String(el._html || "");
  for (const c of el.children || []) out += " " + deepText(c);
  return out;
}
function drawn(id) { return deepText(store.get(id)); }

const failures = [];
/* Every drawn number is a number. A cell that is blocked on one platform can still be another
 * cell's denominator -- throughput per dollar is indexed to one server -- and a row measured on
 * Tenstorrent with no measurement on that server used to draw "NaNx" in the column that WAS
 * measured. Every check below asserts that a row drew and that a blocked cell is labelled; none
 * of them look at whether a drawn figure is finite, so four nodes went out wrong and green. */
for (const [id, el] of store) {
  const t = deepText(el);
  const bad = t.match(/NaN|Infinity|undefined/);
  if (bad) {
    failures.push("#" + id + " drew " + bad[0] + ": " +
                  t.replace(/\s+/g, " ").slice(Math.max(0, bad.index - 80), bad.index + 80));
  }
}
function want(where, needle, why) {
  if (!drawn(where).includes(needle)) failures.push(why + " (missing from #" + where + ")");
}

/* Categories the page draws beside the prediction rows, read off the page itself so this
 * check cannot drift from CATEGORIES. */
const catKeys = [...script.matchAll(/key: "([a-z0-9-]+)", label:/g)].map((m) => m[1]);
if (!catKeys.length) { console.error("no CATEGORIES found in the page"); process.exit(1); }

const catModels = catKeys.flatMap((k) => ((D[k] && D[k].models) || []).filter((m) => !m.hidden));
const predModels = D.models.filter((m) => !m.hidden);

/* Prediction rows: the measured table, the per-accelerator chart, and both derived tables. */
for (const m of predModels) {
  want("t-measured", m.name, m.name + " is a prediction row");
  want("c3-svg", m.name, m.name + " should be in the prediction-time chart");
  for (const t of ["t-derived", "t-perdollar-capex", "t-perdollar"]) {
    want(t, m.name, m.name + " should be in the derived tables");
  }
}
/* Category rows: their own table and chart, and the same derived tables and server charts. */
for (const m of catModels) {
  want("t-design", m.name, m.name + " is a category row");
  want("c5-svg", m.name, m.name + " should be in the seconds chart");
  for (const t of ["t-derived", "t-perdollar-capex", "t-perdollar"]) {
    want(t, m.name, m.name + " should be in the derived tables");
  }
  for (const c of ["c1-svg", "c1b-svg", "c2-svg"]) {
    want(c, m.name, m.name + " should be in the server charts");
  }
}
/* Every measured number reaches a table, and every unmeasured cell says so rather than
 * silently drawing nothing. */
for (const m of predModels.concat(catModels)) {
  for (const [key, c] of Object.entries(m.cells)) {
    const s = c.s_per_fold === undefined ? c.s_per_design : c.s_per_fold;
    const table = predModels.includes(m) ? "t-measured" : "t-design";
    if (c.status === "measured") {
      /* fmtSec picks its own precision per magnitude, so match any number in the table
       * that rounds to this cell rather than one spelling of it. */
      const nums = (drawn(table).match(/[0-9]+\.[0-9]+/g) || []).map(Number);
      if (!nums.some((v) => Math.abs(v - s) < 0.011)) {
        failures.push(m.name + "/" + key + " measured " + s + " s and no cell in #" +
                      table + " rounds to it");
      }
    } else {
      const label = c.status === "not measured" ? "not measured" : "does not run";
      want(table, label, m.name + "/" + key + " should be labelled " + label);
    }
  }
}
/* Host and device halves, wherever both sides measured them. */
for (const m of predModels.concat(catModels)) {
  const t = m.cells.p150a, g = m.cells.h200;
  if (t && g && t.split && g.split) {
    want("t-hostsplit", m.name, m.name + " measured both halves and belongs in the split table");
  }
}
/* Each category states its own scope and methods; inheriting the page's would be a wrong claim. */
for (const k of catKeys) {
  if (!D[k]) continue;
  want("c5-cond", D[k].cond.slice(0, 40), k + "'s conditions");
  want("c5-note", D[k].note.slice(0, 40), k + "'s note");
  const dl = (store.get("methods-dl") || mkEl("dl")).children.map((c) => deepText(c)).join(" ");
  if (!dl.includes(D[k].methods.slice(0, 40))) failures.push(k + "'s methods paragraph is missing");
  const scope = (store.get("scope-dl") || mkEl("dl")).children.map((c) => deepText(c)).join(" ");
  for (const m of (D[k].models || []).filter((x) => !x.hidden)) {
    if (!scope.includes(m.target)) failures.push(k + "/" + m.id + "'s target is missing from the scope list");
  }
}


/* ---------- embeddings ----------
 * The embed rows are deliberately not a CATEGORIES entry, so the category loops above cannot see
 * them and a refactor could drop all six without a single check going red. These three blocks are
 * the reason that cannot happen quietly: the rows drew, they did NOT reach a chart or table they
 * must stay out of, and every per-cell note reached the visible notes list. */
const embedModels = ((D.embed && D.embed.models) || []).filter((m) => !m.hidden);

for (const m of embedModels) {
  want("t-embed", m.name, m.name + " is an embedding row and belongs in #t-embed");
  want("c6-svg", m.name, m.name + " should be in the sequences-per-second chart");
  for (const [key, c] of Object.entries(m.cells)) {
    if (c.status === "measured") {
      /* The page derives seq/s as batch / s_per_batch and nothing else may; assert the derivation
       * landed rather than that some number is present. */
      const seq = c.batch / c.s_per_batch;
      const shown = Math.round(seq).toLocaleString("en-US");
      if (!drawn("t-embed").includes(shown)) {
        failures.push(m.name + "/" + key + " derives " + shown +
                      " seq/s and #t-embed does not show it");
      }
      const perSeq = (c.s_per_batch / c.batch).toFixed(4);
      if (!drawn("t-embed").includes(perSeq)) {
        failures.push(m.name + "/" + key + " derives " + perSeq +
                      " s/seq and #t-embed does not show it");
      }
    } else {
      const label = c.status === "not measured" ? "not measured" : "does not run";
      want("t-embed", label, m.name + "/" + key + " should be labelled " + label);
    }
  }
}

/* An embedding row may not enter any cost, per-server or folding surface. The cost index on this
 * page is built on the DGX H200 at 512 aa folds; an embedding forward has no server price or power
 * figure it could honestly carry, so "card-only" has to be enforced and not just intended. Group
 * labels and cell text only: a row name appearing inside a hover title is provenance prose, not a
 * drawn series. */
function drawnLabels(id) {
  const el = store.get(id);
  if (!el) return "";
  return deepText(el).replace(/<title>[\s\S]*?<\/title>/g, "");
}
const forbidden = ["c1-svg", "c1b-svg", "c2-svg", "c3-svg", "c5-svg",
                   "t-derived", "t-perdollar-capex", "t-perdollar", "t-measured", "t-design"];
for (const m of embedModels) {
  for (const id of forbidden) {
    if (drawnLabels(id).includes(m.name)) {
      failures.push(m.name + " is card-only and must not be drawn in #" + id +
                    " (adding embed to CATEGORIES would do exactly this)");
    }
  }
}

/* Every note in the data reached the visible list. The six H200-against-B200 cell notes are the
 * ones a reader most needs, and a hover-only note is a note most readers never see. */
for (const m of D.models.concat(catModels, embedModels)) {
  if (m.note) want("rownotes", m.note.slice(0, 40), m.name + "'s row note");
  for (const [key, c] of Object.entries(m.cells)) {
    if (c.note) {
      want("rownotes", c.note.slice(0, 40), m.name + "/" + key + "'s cell note");
    }
  }
}

/* A tripwire, and the only hardcoded numbers in this file. Every other check is derived from the
 * data, which means a row DELETED from the data is invisible to all of them: the page renders, the
 * check counts what is left and exits 0. That is the "renders but quietly omits a series" failure
 * this file exists to catch, so the published section sizes are pinned here. Changing a row count
 * is a deliberate act; update this table in the same commit and say why. */
/* Three rows are held in perf/page_rows_pending.json until every processor column is measured
 * (OpenBind-0, PXDesign, Nesso-1). They come back to these counts as they are restored. */
const EXPECT_ROWS = { models: 7, design: 2, affinity: 0, embed: 6 };
for (const [key, n] of Object.entries(EXPECT_ROWS)) {
  const got = key === "models"
    ? predModels.length
    : ((D[key] && D[key].models) || []).filter((m) => !m.hidden).length;
  if (got !== n) {
    failures.push("section '" + key + "' publishes " + got + " rows, expected " + n +
                  ". If that is intended, update EXPECT_ROWS in this file in the same commit.");
  }
}

/* Both main-flow caveats sit above every chart, outside every disclosure. */
{
  const scope = (store.get("scope-dl") || mkEl("dl")).children.map((c) => deepText(c)).join(" ");
  if (D.scope.gpu_generations && !scope.includes(D.scope.gpu_generations.slice(0, 40))) {
    failures.push("the H200-against-B200 caveat did not reach #scope-dl");
  }
  if (D.embed && D.embed.scope && !scope.includes(D.embed.scope.slice(0, 40))) {
    failures.push("the embedding scope line did not reach #scope-dl");
  }
}

/* ---------- the landing page ----------
 * site/index.html reads the same file and draws the same rows with its own copy of the three
 * formulas. It shipped a hand-written copy of the derived values and drew 6 of the 9 rows the
 * data carried, on the front page, with nothing red. So it is checked here too: every folding,
 * design and affinity row reaches the bars, no embedding row does, and the hero's per-dollar
 * range matches an independent recomputation rather than being a number somebody typed. */
const land = runPage("site/index.html");
const bars = deepText(land.store.get("bars"));
for (const m of predModels.concat(catModels)) {
  if (!bars.includes(m.name)) {
    failures.push(m.name + " is a published row and the landing page does not draw it");
  }
}
for (const m of embedModels) {
  if (bars.includes(m.name)) {
    failures.push(m.name + " is card-only and must not be drawn on the landing page, whose " +
                  "chart is per server");
  }
}
{
  const P = Object.fromEntries(D.platforms.map((p) => [p.id, p]));
  const perk = (s, p) => 3600.0 / s * p.accelerators * (p.scaling_efficiency ?? 1.0) /
                         p.price_usd * 1000;
  const ratios = predModels.concat(catModels).map((m) => {
    const t = m.cells.p150a, g = m.cells.b200;
    if (!t || !g || t.status !== "measured" || g.status !== "measured") return null;
    const secs = (c) => c.s_per_fold ?? c.s_per_design;
    return { name: m.name, r: perk(secs(t), P.galaxy_bh) / perk(secs(g), P.dgx_b200),
             pending: !!m.parity_pending };
  }).filter(Boolean).sort((a, b) => a.r - b.r);
  const hi = ratios[ratios.length - 1];
  const hero = deepText(land.store.get("perkrange"));
  const want = ratios[0].r.toFixed(1) + "\u00d7 to " + hi.r.toFixed(1) + "\u00d7";
  if (!hero.includes(want)) {
    failures.push("the landing hero's per-dollar range should read " + want +
                  " over " + ratios.length + " rows, and #perkrange reads \'" + hero + "\'");
  }
  if (hi.pending && !hero.includes(hi.name)) {
    failures.push(hi.name + " sets the top of the hero range and still owes a reference-parity " +
                  "run, so the hero has to name it");
  }
}

if (failures.length) {
  console.error(failures.length + " row(s) did not reach the page:");
  for (const f of failures) console.error("  " + f);
  process.exit(1);
}
console.log("render_check: " + (predModels.length + catModels.length + embedModels.length) +
            " rows drew, " + catKeys.length + " categories, " + embedModels.length +
            " embedding rows card-only, no missing ids, landing page in step");

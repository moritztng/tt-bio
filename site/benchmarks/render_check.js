/* Runs site/benchmarks/index.html's own script against site/data/perf-512aa.json under a small
 * DOM stub, then asserts every row in the data reached the page. The page has no build step and
 * no browser test, so a data block the renderer forgets, or a variable left behind by an edit,
 * used to fail only on load. Checks are derived from the data, not hardcoded, so a new model or
 * a new category is covered the moment it lands.
 *
 * Exit 0 = every row drew. Run from the repo root: node site/benchmarks/render_check.js */
const fs = require("fs");

const html = fs.readFileSync("site/benchmarks/index.html", "utf8");
const script = html.match(/<script[^>]*>([\s\S]*?)<\/script>/)[1];
const raw = fs.readFileSync("site/data/perf-512aa.json", "utf8");
const D = JSON.parse(raw);

const markupIds = new Set();
for (const m of html.matchAll(/id="([^"]+)"/g)) markupIds.add(m[1]);

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

/* The page does fetch(...).then(json).then(render).catch(report); a thenable that resolves
 * synchronously runs the same path and surfaces a render throw as a throw. */
function fetchStub() {
  return {
    _v: { json: () => JSON.parse(raw) },
    then(f) { this._v = f(this._v); return this; },
    catch() { return this; },
  };
}

try {
  new Function("document", "window", "fetch", "setTimeout", "clearTimeout", "console", script)(
    document, window, fetchStub, (f) => f, () => {}, console);
} catch (e) {
  console.error("the page threw while rendering: " + (e && e.stack || e));
  process.exit(1);
}
if (unknownIds.size) {
  console.error("script reads ids that are not in the markup: " + [...unknownIds].join(", "));
  process.exit(1);
}

function deepText(el) {
  if (!el) return "";
  let out = String(el._text || "") + String(el._html || "");
  for (const c of el.children || []) out += " " + deepText(c);
  return out;
}
function drawn(id) { return deepText(store.get(id)); }

const failures = [];
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

if (failures.length) {
  console.error(failures.length + " row(s) did not reach the page:");
  for (const f of failures) console.error("  " + f);
  process.exit(1);
}
console.log("render_check: " + (predModels.length + catModels.length) + " rows drew, " +
            catKeys.length + " categories, no missing ids");

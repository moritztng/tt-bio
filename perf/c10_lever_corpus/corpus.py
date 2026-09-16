"""Verify, price and render the C10 lever ledger. CPU only, no device, no model import.

Every row in ``ledger_src.json`` is a claim about something this project already measured. This
module's whole job is to refuse a claim it cannot ground:

* the evidence file must exist and must contain the row's quote verbatim (whitespace-normalised);
* the quoted ratio must appear inside that quote, so a row cannot carry a number its own source
  does not say;
* a clock is either an integer MHz that appears in a clock-evidence quote, or the literal string
  ``unrecorded``. There is no third option and no default;
* ``status`` is cross-checked against the flag defaults parsed out of ``tt_bio/`` rather than taken
  from prose, because a merged lever defaulting off is not a landed win;
* a row flagged ``contested`` carries two disagreeing readings and is refused into UNCOUNTED rather
  than arbitrated here.

A refused row never reaches the ledger. It reaches ``refusals.json`` with the reason.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_STATE_ROOT = Path(os.environ.get("C10_STATE_ROOT", Path.home() / ".coworker" / "state"))

# The planning model this campaign was launched against. fold_s = FIXED_S + WORK_MHZ_S / AICLK_MHz.
# It interpolates two endpoints of six folds, one of them co-tenanted; its units are MHz*s, which
# are Mcycles only under the model. It is not a hardware cycle counter.
FIXED_S = 2.901
WORK_MHZ_S = 15355.0
TARGET_WORK_MHZ_S = 9583.65  # 10.0 s at a hypothetical constant 1350 MHz with the intercept fixed
BURST_MHZ = 1350
BASE_MHZ = 800

RATIO_RE = re.compile(r"\d\.\d{2,5}x")
FLAG_RE = re.compile(r'env_flag\(\s*"(TT_BIO_[A-Z_0-9]+)"\s*,\s*(True|False|[A-Z_][A-Z_0-9]*)\s*\)')
FLAG_STR_RE = re.compile(
    r'"(TT_BIO_[A-Z_0-9]+)"\s*,\s*"1"\s+if\s+([A-Z_][A-Z_0-9]*)\s+else\s+"0"')
CONST_RE = re.compile(r"^([A-Z_][A-Z_0-9]*)\s*=\s*(True|False)\s*$", re.M)

VALID_STATUS = {"shipped-on", "shipped-default-off", "unmerged", "declined", "refuted",
                "superseded"}
VALID_SCOPE = {"fold", "sampler-wall", "step", "stage", "block", "op", "throughput"}


def norm(s: str) -> str:
    """Collapse whitespace so a quote survives reflowing, without letting it match anything else."""
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- source of truth: code

def flag_defaults(repo: Path = REPO) -> dict[str, bool]:
    """Parse shipped flag defaults out of ``tt_bio/``. This is the status oracle, not prose."""
    consts: dict[str, bool] = {}
    texts: dict[Path, str] = {}
    for p in sorted((repo / "tt_bio").rglob("*.py")):
        t = p.read_text(errors="replace")
        texts[p] = t
        for name, val in CONST_RE.findall(t):
            consts[name] = val == "True"
    out: dict[str, bool] = {}
    for t in texts.values():
        for flag, default in FLAG_RE.findall(t):
            if default in ("True", "False"):
                out[flag] = default == "True"
            elif default in consts:
                out[flag] = consts[default]
        for flag, const in FLAG_STR_RE.findall(t):
            if const in consts:
                out[flag] = consts[const]
    return out


# --------------------------------------------------------------------------- evidence resolution

def resolve(path: str, repo: Path, state_root: Path) -> Path:
    if path.startswith("state:"):
        return state_root / path[len("state:"):]
    if path.startswith("repo:"):
        return repo / path[len("repo:"):]
    raise ValueError(f"evidence path must start with 'state:' or 'repo:', got {path!r}")


def check_quote(ev: dict, repo: Path, state_root: Path) -> str | None:
    """Return a refusal reason, or None when the quote is grounded in the named file."""
    try:
        p = resolve(ev["path"], repo, state_root)
    except ValueError as e:
        return str(e)
    if not p.is_file():
        return f"evidence file missing: {ev['path']}"
    if norm(ev["quote"]) not in norm(p.read_text(errors="replace")):
        return f"quote not found verbatim in {ev['path']}: {norm(ev['quote'])[:70]!r}"
    return None


# --------------------------------------------------------------------------- pricing

def implied_mcycles(ratio: float, clock_mhz: float, scope: str) -> float | None:
    """Mcycles a fold-scope ratio implies under the planning model, at a KNOWN clock.

    r = (F + W/f) / (F + (W - dW)/f)  =>  dW = (W + F*f) * (1 - 1/r)

    Only defined for a whole-fold ratio on the 512 aa cell and only when the clock is known. A step,
    block or op ratio does not have a fold denominator, so it gets None and stays in seconds.
    """
    if scope != "fold" or ratio <= 0:
        return None
    return (WORK_MHZ_S + FIXED_S * clock_mhz) * (1.0 - 1.0 / ratio)


def clock_band(ratio: float, scope: str) -> dict | None:
    """What an unrecorded-clock fold ratio would imply at each end of this card's clock range.

    This is a BRACKET over an unknown, not a measurement, and the ledger labels it as one.
    """
    if scope != "fold" or ratio <= 0:
        return None
    lo = implied_mcycles(ratio, BASE_MHZ, scope)
    hi = implied_mcycles(ratio, BURST_MHZ, scope)
    return {"mcycles_if_800mhz": round(lo, 1), "mcycles_if_1350mhz": round(hi, 1),
            "spread_pct": round(100.0 * (hi / lo - 1.0), 2)}


def fold_seconds_at(clock_mhz: float) -> float:
    return FIXED_S + WORK_MHZ_S / clock_mhz


# --------------------------------------------------------------------------- the ledger

def build(src: list[dict], repo: Path = REPO, state_root: Path = DEFAULT_STATE_ROOT):
    """Return (ledger, refusals). Refusals are named, never silently dropped."""
    defaults = flag_defaults(repo)
    ledger, refusals = [], []
    seen: set[str] = set()

    for row in src:
        name = row.get("name")
        why: list[str] = []
        if not name:
            refusals.append({"name": "<unnamed>", "reasons": ["row has no name"]})
            continue
        if name in seen:
            why.append("duplicate lever name")
        seen.add(name)

        for field in ("op_class", "deletes", "status", "evidence", "clock"):
            if field not in row:
                why.append(f"missing required field {field!r}")
        if why:
            refusals.append({"name": name, "reasons": why})
            continue

        if row["status"] not in VALID_STATUS:
            why.append(f"status {row['status']!r} not one of {sorted(VALID_STATUS)}")

        if not row["evidence"]:
            why.append("no evidence entries")
        for ev in row["evidence"]:
            r = check_quote(ev, repo, state_root)
            if r:
                why.append(r)

        # --- the ratio must be spoken by its own source
        ratio = row.get("ratio")
        scope = row.get("ratio_scope")
        if row.get("contested"):
            why.append("CONTESTED: two disagreeing readings on record; this row is not priced here")
        elif ratio is None:
            if not row.get("unpriced_reason"):
                why.append("no ratio and no unpriced_reason")
        else:
            if scope not in VALID_SCOPE:
                why.append(f"ratio_scope {scope!r} not one of {sorted(VALID_SCOPE)}")
            txt = " ".join(norm(e["quote"]) for e in row["evidence"])
            if row["ratio_text"] not in txt:
                why.append(f"ratio {row['ratio_text']!r} does not appear in the row's own quotes")
            if not RATIO_RE.fullmatch(row["ratio_text"]) and "x" in row["ratio_text"]:
                why.append(f"ratio_text {row['ratio_text']!r} is not a bare N.NNNNx figure")
            if abs(float(row["ratio_text"].rstrip("x")) - float(ratio)) > 5e-6:
                why.append("ratio field disagrees with ratio_text")

        # --- the clock is attested or it is the literal string 'unrecorded'
        clock = row["clock"]
        if clock == "unrecorded":
            pass
        elif isinstance(clock, int):
            ce = row.get("clock_evidence")
            if not ce:
                why.append(f"clock {clock} MHz declared with no clock_evidence")
            else:
                r = check_quote(ce, repo, state_root)
                if r:
                    why.append(f"clock_evidence: {r}")
                elif str(clock) not in norm(ce["quote"]):
                    why.append(f"clock_evidence quote does not contain {clock}")
        else:
            why.append(f"clock must be an int MHz or the literal 'unrecorded', got {clock!r}")

        # --- status against the tree, not against prose
        flag = row.get("flag")
        if flag:
            live = defaults.get(flag)
            expect = {True: "shipped-on", False: "shipped-default-off"}.get(live)
            if live is None and row["status"] in ("shipped-on", "shipped-default-off"):
                why.append(f"{flag} is not in tt_bio/ but status says {row['status']}")
            if live is not None and expect != row["status"]:
                why.append(f"{flag} defaults {live} in tt_bio/ but status says {row['status']}")
        elif row["status"] in ("shipped-on", "shipped-default-off"):
            # No env_flag to read, so the row must quote the line of tt_bio/ that sets the default.
            se = row.get("status_evidence")
            if not se:
                why.append(f"status {row['status']} with no flag and no status_evidence from tt_bio/")
            elif not se["path"].startswith("repo:tt_bio/"):
                why.append("status_evidence must cite a file under repo:tt_bio/")
            else:
                r = check_quote(se, repo, state_root)
                if r:
                    why.append(f"status_evidence: {r}")

        # An unshipped lever must prove it is unshipped: either its symbol is absent from tt_bio/,
        # or the un-levered form it would replace is still present there.
        if row["status"] in ("unmerged", "declined", "refuted"):
            sym, pres = row.get("absent_symbol"), row.get("present_symbol")
            if not sym and not pres:
                why.append(f"status {row['status']} needs absent_symbol or present_symbol "
                           "so the claim is checked against tt_bio/ rather than against prose")
            src = [f.read_text(errors="replace") for f in (repo / "tt_bio").rglob("*.py")]
            if sym and any(sym in s for s in src):
                why.append(f"absent_symbol {sym!r} IS present in tt_bio/")
            if pres and not any(pres in s for s in src):
                why.append(f"present_symbol {pres!r} is NOT in tt_bio/")

        if why:
            refusals.append({"name": name, "reasons": why})
            continue

        out = dict(row)
        out["tree_default"] = defaults.get(flag) if flag else None
        if ratio is not None:
            if isinstance(clock, int):
                mc = implied_mcycles(float(ratio), clock, scope)
                out["implied_mcycles"] = round(mc, 1) if mc is not None else None
                out["clock_band"] = None
            else:
                out["implied_mcycles"] = None
                out["clock_band"] = clock_band(float(ratio), scope)
        ledger.append(out)

    return ledger, refusals


# --------------------------------------------------------------------------- rendering

def render(ledger: list[dict], refusals: list[dict]) -> str:
    L = []
    L.append("# C10 lever ledger\n")
    L.append("Generated by `perf/c10_lever_corpus/corpus.py`. Do not hand-edit; edit "
             "`ledger_src.json` and regenerate.\n")
    L.append(f"{len(ledger)} priced rows, {len(refusals)} refused.\n")
    L.append("\n## Ledger\n")
    L.append("| lever | op class | deletes | ratio | scope | cell | clock | fold s | accuracy | "
             "status | evidence |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(ledger, key=lambda r: (-(r.get("ratio") or 0), r["name"])):
        clock = f"{r['clock']} MHz" if isinstance(r["clock"], int) else "**unrecorded**"
        ratio = r["ratio_text"] if r.get("ratio") else "-"
        acc = r.get("accuracy_A")
        acc = "bit-exact" if acc == 0 else (f"{acc} A" if acc is not None else "-")
        L.append("| `{n}` | {c} | {d} | {r} | {s} | {cell} | {clk} | {fs} | {a} | {st} | {e} |".format(
            n=r["name"], c=r["op_class"], d=r["deletes"], r=ratio, s=r.get("ratio_scope") or "-",
            cell=r.get("cell", "-"), clk=clock, fs=r.get("fold_seconds") or "-", a=acc,
            st=r["status"], e=r["evidence"][0]["path"]))
    L.append("\n## Refused / uncounted\n")
    for r in sorted(refusals, key=lambda r: r["name"]):
        L.append(f"- **{r['name']}** — " + "; ".join(r["reasons"]))
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT))
    ap.add_argument("--src", default=str(HERE / "ledger_src.json"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--require-no-refusals", action="store_true",
                    help="exit 2 if any row was refused (for a gate, not for this ledger)")
    a = ap.parse_args(argv)

    src = json.loads(Path(a.src).read_text())
    ledger, refusals = build(src, REPO, Path(a.state_root))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    (out / "refusals.json").write_text(json.dumps(refusals, indent=2, sort_keys=True) + "\n")
    (out / "LEDGER.md").write_text(render(ledger, refusals))
    print(f"{len(ledger)} priced, {len(refusals)} refused -> {out}")
    clocked = [r for r in ledger if isinstance(r["clock"], int)]
    print(f"rows carrying a recorded clock: {len(clocked)} of {len(ledger)}")
    if a.require_no_refusals and refusals:
        for r in refusals:
            print("REFUSED", r["name"], r["reasons"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

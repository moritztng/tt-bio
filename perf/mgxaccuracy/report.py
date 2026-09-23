#!/usr/bin/env python3
"""Designability by size, with the floor a size effect has to beat.

The row this serves opened on scRMSD 11.480 A at a 1536-residue target against 4.054 A at
512, n=1 each. Both numbers are real and the comparison is not, for two reasons this report
keeps in front of the reader:

  * **one design is not a distribution.** BoltzGen draws a fresh binder each time. At n=8 the
    same 512 target spans 3.30-13.78 A, so two single draws can differ by 2.8x with nothing
    between them but noise.
  * **the two rungs are not the same target.** A size ladder cut from one structure walks the
    binding problem along with the size: 512 is that structure's first 512 residues, 1536 is
    a two-chain 1536-mer. So the honest floor is the spread ACROSS TARGETS AT ONE SIZE, and a
    size effect is only a size effect if it clears that.

So the report prints three things in order: per-target distributions, the fixed-size
across-target spread, and only then the size comparison, labelled with which of the two it
beats.

    python3 perf/mgxaccuracy/report.py perf/mgxaccuracy/results/*.jsonl ~/mgxacc-work/*.jsonl
"""
import argparse
import json
import pathlib
import statistics as st
import sys

STRICT_A = 2.0
PERMISSIVE_A = 4.0

# The two instruments the ceiling row used, each with the bars that mean something for it.
# clash fraction has no published bar: `tt_bio/size_limits.py` records per-model BANDS, which
# are model-specific, so this prints the distribution and leaves the judging to the reader.
METRICS = {
    "scrmsd": {"label": "scRMSD (A) — isolated refold, Kabsch CA-RMSD",
               "fmt": "{:>9.2f}", "bars": (("<=2A", STRICT_A), ("<=4A", PERMISSIVE_A))},
    "clash_frac": {"label": "clash fraction — heavy-atom clashes < 2.0 A / atoms",
                   "fmt": "{:>9.5f}", "bars": ()},
}


def _geom_rows(r: dict) -> list[float] | None:
    """Per-design clash fraction from a `perf/mgxscale/batchqa.py --geometry` record.

    The same question as scRMSD, asked by the other instrument the ceiling row used, so it
    gets the same treatment: a distribution, not a draw."""
    if "geometry" not in r:
        return None
    out = []
    for g in r["geometry"]:
        c = g.get("checks", g)
        cl = c.get("clashes") or {}
        n, k = c.get("n_atoms"), cl.get("n")
        if k is not None and n:
            out.append(k / n)
    return out or None


def _from_out_dir(name: str) -> tuple[str, int | None, int]:
    """(model, size, offset) from a batchqa out_dir like `out_boltzgen_1536_d8_s400_tag`.

    batchqa records the directory, not the axes; job.py's tag is the only place they are
    written down, so this reads them back rather than asking the caller to retype them."""
    parts = name.removeprefix("out_").split("_")
    model = parts[0] if parts else "?"
    size = off = None
    for i, x in enumerate(parts[1:], 1):
        if size is None and x.isdigit():
            size = int(x)
        elif x.startswith("o") and x[1:].isdigit():
            off = int(x[1:])
    return model, size, off or 0


def load(paths) -> list[dict]:
    """One record per (model, size, crop offset) cell, from either shape of row.

    `job.py` writes a run row with the scRMSD under `dsg`; `prior_runs.jsonl` records
    already-published numbers at the top level. Both are read, and a row with no scRMSD at
    all is skipped rather than counted as a zero."""
    out = []
    for p in paths:
        p = pathlib.Path(p).expanduser()
        if not p.is_file():
            print(f"[report] missing {p}", file=sys.stderr)
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            geom = _geom_rows(r)
            if geom is not None:
                model, size, off = _from_out_dir(r.get("out_dir", ""))
                out.append({"model": model, "size": size, "offset": off, "scrmsd": geom,
                            "side": "device", "card": None, "aiclk": None, "load": None,
                            "src": p.name})
                continue
            d = r.get("dsg") or r
            sc = d.get("scrmsd")
            if not sc or d.get("error"):
                # Say so. A row that carries a summary but no per-design values (the committed
                # 7ROA reference is one) cannot join a distribution, and dropping it in silence
                # is how a reader comes to believe it was counted.
                why = d.get("error") or "no per-design scRMSD"
                print(f"[report] skipped {p.name} {r.get('model','?')} "
                      f"{r.get('target_res','?')} res: {why}", file=sys.stderr)
                continue
            out.append({
                "model": r.get("model", "?"),
                "size": r.get("target_res"),
                "offset": r.get("crop_offset", 0) or 0,
                "scrmsd": [float(v) for v in sc],
                "side": r.get("side", "device"),
                "card": r.get("card"),
                "aiclk": (r.get("aiclk") or {}).get("median") if isinstance(r.get("aiclk"), dict)
                         else r.get("aiclk_median_during"),
                "load": (r.get("load") or {}).get("median") if isinstance(r.get("load"), dict)
                        else r.get("load_median"),
                "src": p.name,
            })
    return out


def cell(vals: list[float], bars) -> dict:
    s = sorted(vals)
    return {"n": len(s), "min": s[0], "median": st.median(s), "max": s[-1],
            "bars": [sum(v <= t for v in s) / len(s) for _, t in bars]}


def mannwhitney(a: list[float], b: list[float]) -> tuple[float, str]:
    """U and an exact-ish verdict for two small samples, with no scipy on this host.

    Reported for the POOLED per-design values, which is the optimistic reading: designs from
    one target are not independent of that target, so a pooled p understates how much of the
    difference is one target being harder. That is why the target-level comparison is printed
    beside it and is the one the verdict leans on."""
    ranks = {v: i for i, v in enumerate(sorted(a + b), 1)}
    ra = sum(ranks[v] for v in a)
    na, nb = len(a), len(b)
    u_a = ra - na * (na + 1) / 2
    u = min(u_a, na * nb - u_a)
    mu = na * nb / 2
    sd = (na * nb * (na + nb + 1) / 12) ** 0.5
    if sd == 0:
        return u, "degenerate"
    z = abs(u - mu) / sd
    return u, f"z={z:.2f}" + ("  (|z|>1.96: separated)" if z > 1.96 else
                              "  (|z|<=1.96: NOT separated)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--model", default="boltzgen")
    ap.add_argument("--metric", default="scrmsd", choices=sorted(METRICS),
                    help="which metric the input files carry; sets the bars and the format")
    args = ap.parse_args()

    rows = [r for r in load(args.jsonl) if r["model"] == args.model]
    if not rows:
        print("no scored rows")
        return 1

    # Merge cells that are the same (size, offset, side) measured more than once.
    cells: dict[tuple, list[float]] = {}
    meta: dict[tuple, list[str]] = {}
    for r in rows:
        k = (r["side"], r["size"], r["offset"])
        cells.setdefault(k, []).extend(r["scrmsd"])
        meta.setdefault(k, []).append(
            f"{r['src']}" + (f" card{r['card']}" if r["card"] is not None else "")
            + (f" AICLK {r['aiclk']}" if r["aiclk"] else "")
            + (f" load {r['load']}" if r["load"] else ""))

    spec = METRICS[args.metric]
    fmt, bars = spec["fmt"], spec["bars"]
    print(f"\n{'='*100}\nby target — {args.model}   {spec['label']}\n{'='*100}")
    head = (f"{'side':<10}{'size':>6}{'offset':>8}{'n':>4}"
            + "".join(f"{h:>9}" for h in ("min", "median", "max"))
            + "".join(f"{name:>7}" for name, _ in bars))
    print(head + "   provenance")
    for k in sorted(cells):
        side, size, off = k
        c = cell(cells[k], bars)
        line = (f"{side:<10}{size:>6}{off:>8}{c['n']:>4}"
                + fmt.format(c["min"]) + fmt.format(c["median"]) + fmt.format(c["max"])
                + "".join(f"{v*100:>6.0f}%" for v in c["bars"]))
        print(f"{line}   {meta[k][0]}")
        for extra in meta[k][1:]:
            print(f"{'':<{len(line)}}   {extra}")

    # The floor: how far apart are two targets of the SAME size?
    print(f"\n{'-'*94}\nfixed-size spread across targets — the floor a size effect must beat"
          f"\n{'-'*94}")
    floors = {}
    for side, size in sorted({(k[0], k[1]) for k in cells}):
        meds = [st.median(cells[k]) for k in cells if k[0] == side and k[1] == size]
        if len(meds) < 2:
            print(f"{side:<10}{size:>6}  only {len(meds)} target(s) — no floor yet")
            continue
        floors[(side, size)] = (min(meds), max(meds))
        print(f"{side:<10}{size:>6}  {len(meds)} targets, medians "
              f"{' / '.join(fmt.format(m).strip() for m in sorted(meds))}"
              f"   -> spread {fmt.format(max(meds) - min(meds)).strip()}")

    # Only now the size comparison.
    print(f"\n{'-'*94}\n512 vs 1536\n{'-'*94}")
    for side in sorted({k[0] for k in cells}):
        a = [v for k in cells if k[0] == side and k[1] == 512 for v in cells[k]]
        b = [v for k in cells if k[0] == side and k[1] == 1536 for v in cells[k]]
        if not a or not b:
            print(f"{side}: have {len(a)} designs at 512 and {len(b)} at 1536 — incomplete")
            continue
        ma, mb = st.median(a), st.median(b)
        u, verdict = mannwhitney(a, b)
        print(f"{side}: pooled median {fmt.format(ma).strip()} at 512 (n={len(a)}) vs "
              f"{fmt.format(mb).strip()} at 1536 (n={len(b)});  U={u:.0f}  {verdict}")
        fl = floors.get((side, 512))
        if fl:
            inside = fl[0] <= mb <= fl[1]
            print(f"     the 1536 pooled median is "
                  f"{'INSIDE' if inside else 'OUTSIDE'} the 512 across-target band "
                  f"{fmt.format(fl[0]).strip()}-{fmt.format(fl[1]).strip()}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

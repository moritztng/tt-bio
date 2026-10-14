"""BOARD.md rows (state/bmk/BOARD.md format) for one session: every parsed arm with its warm median, n / range, SM clock during
the arm, versions (GPU driver, torch, the stock package, kit commit) and the command that reproduces it.

    python3 perf/kitcmp/board.py perf/kitcmp/results/<label> --gpu a100 --row bmk-a100 [--when 2026-10-14]
"""
import argparse, csv, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parse import arms  # noqa: E402

MODE = {"off": "kit-off", "exact": "kit-exact", "fast": "kit-fast", "off-det": "kit-off-det", "exact-det": "kit-exact-det"}
# kit -> the stock package its pip freeze names
PKG = {"boltz2": "boltz", "protenix_v2": "protenix", "opendde": "opendde", "openfold3": "openfold3", "openfold3_ob0": "openfold3",
       "esmfold2": "esm", "esmc": "esm", "rosettafold3": "rc-foundry", "rfdiffusion3": "rc-foundry", "pxdesign": "pxdesign",
       "stock-boltzgen": "boltzgen", "stock-saprot": "transformers", "stock-nesso1": "nesso"}


def freeze(path):
    v = {}
    if path.exists():
        for l in path.read_text(errors="replace").splitlines():
            m = re.match(r"([A-Za-z0-9_.\-]+)\s*(?:==|@)\s*(\S+)", l)
            if m:  # a wheel installed from a file names its version in the filename (boltz-2.2.1-py3-none-any.whl)
                w = re.search(r"/[A-Za-z0-9_.]+-(\d[^-/]*)-[^/]*\.whl$", m.group(2))
                v[m.group(1).lower().replace("_", "-")] = w.group(1) if w else m.group(2)
    return v


def driver(path):
    if path.exists():
        rows = list(csv.reader(path.open()))
        if len(rows) > 1:
            return rows[1][1].strip()
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--row", required=True)
    ap.add_argument("--when", default="")
    a = ap.parse_args()
    res = a.session / "results"
    kit = (a.session / "kit-commit").read_text().strip()[:9] if (a.session / "kit-commit").exists() else "?"
    for r in arms(res):
        arm = r.get("arm") or ""
        src = res / arm if arm else None
        if r["mode"].startswith("stock"):
            key = {"boltzgen": "stock-boltzgen", "nesso1": "stock-nesso1"}.get(r["cell"], "stock-saprot")
            d = res / key
            fz, drv, ver = freeze(d / "pip-freeze.txt"), driver(d / "gpu.csv"), ""
            mode, cmd = r["mode"], f"MODELS={key} perf/kitcmp/session.sh --gpu {a.gpu} <label> <cap> (box_stock.sh {key[6:]})"
        else:
            k = re.sub(r"-(full_nomsa|fast|no-.*)$", "", arm) if arm else "esmc"
            d = res / f"setup-{k}"
            fz, drv = freeze(d / "pip-freeze.txt"), driver(d / "gpu.csv")
            ver, mode = f"kit {kit}, ", MODE[r["mode"]]
            key = k
            cmd = (f"MODELS={k}:{r['mode']} perf/kitcmp/session.sh --gpu {a.gpu} <label> <cap> "
                   f"(box_arms.sh {k}: run.sh --config <gpu> --mode {r['mode'].replace('-det', ' --det 1')})")
        pkg = PKG.get(key, "")
        ver += f"torch {fz.get('torch', '?')}, {pkg} {fz.get(pkg, '?')}, driver {drv}"
        if r["median"] is None:
            continue
        rng = f"n={r['n']} {min(r['warm']):.4g}-{max(r['warm']):.4g}"
        clk = f"SM {r['sm_mhz']:.0f} MHz med (min {r['sm_min']:.0f}), {r['watts']:.0f} W" if r["sm_mhz"] else "SM not sampled under load"
        print(f"| {r['cell']} | {a.gpu} | {mode} | {r['median']:.4g} | {rng} | {clk} | {ver} | {a.when} | {cmd} | {a.row} |")


if __name__ == "__main__":
    main()

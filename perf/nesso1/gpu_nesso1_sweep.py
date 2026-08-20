"""Drive the Nesso-1 reference sweep on a rented GPU: one subprocess per cell, sequentially.

    python perf/nesso1/gpu_nesso1_sweep.py --legs ladder,ligands --reps 3

One process per cell rather than one process for all of them, because the cells differ in shape
and a shared process would let one rung's allocator state and autotune cache leak into the next.
Sequential rather than parallel, because every cell asserts it was alone on the card and a
co-tenant -- even our own -- voids the absolute seconds.

Each cell is run twice, kernels ON and kernels OFF. The checkpoint ships `use_kernels: true`, so
ON is the shipped default and OFF is what `--no_kernels` gives you. Both are recorded because
cuEquivariance is a CUDA-only kernel library with no Tenstorrent equivalent: a reference measured
with it on is a different denominator from one measured without, and which one the port is scored
against is a decision that has to be made in the open.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).parent
RUN = HERE / "gpu_nesso1_run.py"


def cells(legs: set[str]) -> list[dict]:
    out = []
    if "ladder" in legs:
        for d in sorted((HERE / "inputs" / "ladder").iterdir()):
            aa = int(d.name[2:])
            for kern in (True, False):
                out.append({"leg": "ladder", "dir": d, "aa": aa, "ligand_heavy": 22,
                            "kernels": kern,
                            "label": "ladder_aa%d_%s" % (aa, "cueq" if kern else "torch")})
    if "ligands" in legs:
        for d in sorted((HERE / "inputs" / "ligands").iterdir()):
            out.append({"leg": "ligands", "dir": d, "aa": 256, "ligand_tag": d.name,
                        "kernels": True,
                        "label": "lig_%s_cueq" % d.name})
    if "norefine" in legs:
        # The size ladder with the two-stage pocket crop OFF. `refine_protein_inference` ships ON
        # with a 256-token budget, so recycles 1..N run on a cropped complex no matter how long the
        # protein is -- the shipped default flattens the length axis by construction. Measuring the
        # OFF arm is the only way to say how much of the flatness is the crop and how much is the
        # model.
        for d in sorted((HERE / "inputs" / "ladder").iterdir()):
            aa = int(d.name[2:])
            if aa < 256:
                continue
            out.append({"leg": "norefine", "dir": d, "aa": aa, "ligand_heavy": 22,
                        "kernels": True, "refine": "off",
                        "label": "norefine_aa%d_cueq" % aa})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="ladder,ligands")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--python", default="/work/v_nesso/bin/python")
    ap.add_argument("--results", default="/work/results")
    ap.add_argument("--out-root", default="/work/out")
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args()

    legs = {x.strip() for x in args.legs.split(",") if x.strip()}
    results = pathlib.Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    todo = cells(legs)
    print("%d cells: %s" % (len(todo), ", ".join(c["label"] for c in todo)), flush=True)

    for i, c in enumerate(todo):
        rep = results / ("%s.json" % c["label"])
        if rep.exists():
            try:
                if json.loads(rep.read_text()).get("ok"):
                    print("[%2d/%d] %-28s cached" % (i + 1, len(todo), c["label"]), flush=True)
                    continue
            except Exception:                                 # noqa: BLE001
                pass
        cmd = [args.python, str(RUN), "--inputs", str(c["dir"]),
               "--out-dir", "%s/%s" % (args.out_root, c["label"]),
               "--report", str(rep), "--reps", str(args.reps), "--label", c["label"],
               "--refine", c.get("refine", "on")]
        if not c["kernels"]:
            cmd.append("--no-kernels")
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        ok = False
        why = "no report"
        if rep.exists():
            d = json.loads(rep.read_text())
            ok, why = d.get("ok"), (d.get("why") or "")[:120]
            reps_s = d.get("rep_s") or []
            warm = sorted(reps_s[1:])[len(reps_s[1:]) // 2] if len(reps_s) > 1 else None
            fwd = None
            ph = d.get("phases") or {}
            if len(reps_s) > 1:
                warm_fwd = [ph[str(r)].get("forward") for r in range(1, len(reps_s))
                            if str(r) in ph]
                warm_fwd = [x for x in warm_fwd if x is not None]
                if warm_fwd:
                    fwd = sorted(warm_fwd)[len(warm_fwd) // 2]
            print("[%2d/%d] %-28s ok=%s  rep=%s  forward=%s  (%.0fs)  %s"
                  % (i + 1, len(todo), c["label"], ok, warm, fwd, time.time() - t0, why),
                  flush=True)
        else:
            print("[%2d/%d] %-28s NO REPORT rc=%d\n%s"
                  % (i + 1, len(todo), c["label"], p.returncode, p.stderr[-1500:]), flush=True)
        if not ok:
            print("  stderr tail:\n%s" % p.stderr[-1200:], flush=True)


if __name__ == "__main__":
    sys.exit(main())

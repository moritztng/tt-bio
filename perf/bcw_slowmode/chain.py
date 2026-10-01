#!/usr/bin/env python3
"""The slow-mode legs on ONE chip of a Wormhole Galaxy, under one lease.

    chain.py --chip 30 --params ~/bwx/af2_params --out OUT g:128:4:8:lean r:hHSA:150:f:lean

Two leg kinds, each its own process, each written to the ledger as it lands:

    g:N:EXTRA:EVO:MODE   `perf/bcx_afgrad/afgrad.py stack --ckpt --memory MODE`: the whole
                         stack's dL/dlogits against a FLOAT64 reference, with the finite
                         difference and permutation controls afgrad already runs. This is how
                         a memory mode is graded, and it is graded against float64 rather
                         than against the fast mode: two approximations agreeing proves
                         nothing.
    r:TARGET:BINDER:f|t:MODE   `perf/bgx_size/rung.py --memory MODE`, footprint or timed: a
                         real BindCraft 2 campaign at a real token axis, which is where a
                         mode either reaches a size or refuses.

The lease, the held device node and the clock sampler are `perf/b2p_wh/ladder.py`'s, imported
rather than repeated: the Galaxy's agent takes a chip back the moment its holder lets go, and
its per-chip reset waits on an open fd rather than on the lease.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf" / "b2p_wh"))

import ladder as L                                                      # noqa: E402

MODES = ("fast", "lean", "offload")


def parse(spec: str):
    kind, rest = spec.split(":", 1)
    if kind == "g":
        n, extra, evo, mode = rest.split(":")
        assert mode in MODES, spec
        return {"kind": "g", "tag": f"g{n}_e{extra}_v{evo}_{mode}", "n": int(n),
                "extra": int(extra), "evo": int(evo), "mode": mode}
    if kind == "r":
        target, binder, how, mode = rest.split(":")
        assert how in ("f", "t") and mode in MODES, spec
        return {"kind": "r", "tag": f"{target}_{binder}_{how}_{mode}", "target": target,
                "binder": int(binder), "how": how, "mode": mode}
    raise SystemExit(f"{spec}: leg is g:N:EXTRA:EVO:MODE or r:TARGET:BINDER:f|t:MODE")


def command(leg, args, out: pathlib.Path):
    if leg["kind"] == "g":
        return [sys.executable, "-u", str(ROOT / "perf/bcx_afgrad/afgrad.py"), "stack",
                "--params", args.grade_params or args.params, "--n", str(leg["n"]), "--extra", str(leg["extra"]),
                "--evo", str(leg["evo"]), "--ckpt", "--memory", leg["mode"],
                "--eps", "3e-2,1e-2,3e-3"]
    cmd = [sys.executable, "-u", str(ROOT / "perf/bgx_size/rung.py"),
           "--target", leg["target"], "--binder", str(leg["binder"]),
           "--trajectories", "1", "--max-trajectories", "1",
           "--params", args.params, "--out", str(out),
           "--memory", leg["mode"],
           "--rounds", str(args.fp_rounds if leg["how"] == "f" else args.rounds)]
    if leg["how"] == "f":
        cmd.append("--footprint")
    return cmd


def harvest(leg, rung: pathlib.Path) -> dict:
    """The numbers a leg's own artifact carries, so the ledger alone answers the question."""
    got: dict = {}
    if leg["kind"] == "r":
        try:
            r = json.loads((rung / "rung.json").read_text())
            got = {k: r.get(k) for k in
                   ("evoformer_axis", "complex_residues", "resident_peak_gb", "free_at_peak_gb",
                    "largest_free_block_at_peak_mb", "device_total_gb", "rounds_done", "error",
                    "memory_used", "aiclk_run", "per_round_seconds", "round_seconds_median")}
        except (OSError, ValueError) as exc:
            got = {"note": f"no rung.json: {exc!r}"}
        return got
    stem = (f"stack_n{leg['n']}_e{leg['extra']}_v{leg['evo']}_ckpt"
            f"{'' if leg['mode'] == 'fast' else '_' + leg['mode']}.json")
    try:
        b = json.loads((ROOT / "perf/bcx_afgrad" / stem).read_text())
        got = {"device_vs_f64": b.get("device_vs_f64"),
               "torch_bf16_vs_f64": b.get("torch_bf16_vs_f64"),
               "permuted_device_grad_vs_f64": b.get("permuted_device_grad_vs_f64"),
               "fd_best": b.get("fd_best"), "zero_seed_max_abs": b.get("zero_seed_max_abs"),
               "device_repeat_max_abs_diff": b.get("device_repeat_max_abs_diff"),
               "grad_norm": (b.get("device") or {}).get("grad_norm"),
               "timing": (b.get("device") or {}).get("timing"),
               "artifact": str(ROOT / "perf/bcx_afgrad" / stem)}
    except (OSError, ValueError) as exc:
        got = {"note": f"no {stem}: {exc!r}"}
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)
    ap.add_argument("--params", required=True, help="AF2 weights directory, for the rungs")
    ap.add_argument("--grade-params", dest="grade_params", default="",
                    help="the single npz afgrad grades on; defaults to --params")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--footprint-rounds", dest="fp_rounds", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("legs", nargs="+", metavar="g:N:EXTRA:EVO:MODE | r:TARGET:BINDER:f|t:MODE")
    args = ap.parse_args()
    plan = [parse(s) for s in args.legs]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TT_VISIBLE_DEVICES"] = str(args.chip)
    from tt_bio.device_lease import DeviceLease

    lease = DeviceLease(card=args.chip, timeout=600).acquire()
    node = L.device_node(args.chip)
    held = os.open(node, os.O_RDWR)
    say = lambda m: print(f"{time.strftime('%FT%TZ', time.gmtime())} {m}", flush=True)  # noqa: E731
    say(f"holding {lease.path} and {node}; {len(plan)} legs")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: (lease.release(), sys.exit(128 + s)))

    stop = threading.Event()
    threading.Thread(target=L.sample, args=(out, stop), daemon=True).start()
    env = dict(os.environ, TT_BIO_LEASE_DIR=str(out / "leases"),
               TT_BIO_LEASE_CARDS=str(args.chip))
    ledger = out / "chain.jsonl"
    try:
        for leg in plan:
            rung = out / leg["tag"]
            rung.mkdir(exist_ok=True)
            say(f"=== {leg['tag']} load {os.getloadavg()[0]:.2f}")
            t0 = time.time()
            with open(rung / "leg.log", "w") as log:
                try:
                    rc = subprocess.call(command(leg, args, rung), stdout=log,
                                         stderr=subprocess.STDOUT, env=env, cwd=str(ROOT),
                                         timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            row = {**leg, "rc": rc, "seconds": round(time.time() - t0, 1),
                   "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                   **harvest(leg, rung)}
            with open(ledger, "a", buffering=1) as f:
                f.write(json.dumps(row) + "\n")
            say(f"    {leg['tag']} rc={rc} {row['seconds']:.0f}s "
                f"{json.dumps({k: row.get(k) for k in ('evoformer_axis', 'resident_peak_gb', 'device_vs_f64', 'error')})[:300]}")
    finally:
        stop.set()
        os.close(held)
        lease.release()
        say("chain done, lease released")
    return 0


if __name__ == "__main__":
    sys.exit(main())

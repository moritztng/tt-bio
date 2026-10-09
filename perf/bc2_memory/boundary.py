"""Issue 18's measurement, on our card: device DRAM at every trajectory boundary of a campaign.

The reporter wrapped `bindcraft.campaign.run_trajectory` and read the allocator just before and
just after each trajectory. This does the same, and at every `before` boundary, when no fold is in
flight, it also charges every live DRAM buffer to an owner (`census.py`). Per trajectory it counts
trunk loads and device calls, because the work a trajectory does is set by its rounds and its
model draws, not by its index, and a cost that grows has to be read against both.

    TT_VISIBLE_DEVICES=1 PYTHONPATH=$PWD:$PWD/.bci/bc2 timeout 3h nice -n 10 python \\
        perf/bc2_memory/boundary.py --params ~/bcx_e2e/af2_params --arm fixed \\
        --trajectories 6 --out perf/bc2_memory/out/fixed.json --project /tmp/bc2mem_fixed

`--arm unfixed` makes `pair_mm.forget` a no-op, which is the code the reporter ran. `--steps`
cuts the stage rounds (BindCraft 2's defaults are 50+25+45+5+15); `--steps default` keeps them.
Every stage gate is set to 0 so that every trajectory runs its whole schedule and does the same
work. The target is BindCraft 2's own hPDL1 with the binder length chosen to land on `--tokens`.
`--binder-lengths A,B` instead draws a length per trajectory from [A, B] as a campaign does, and
`--gates stock` keeps BindCraft 2's own stage gates. Each row also carries the device's program
cache size and the allocator's live and free block counts: a compiled program keeps its kernel
binaries in a DRAM buffer of its own, which no Python object holds and the census cannot see.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

GB = 1e9
GIB = 2 ** 30


def _settings(args):
    import bindcraft
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.settings import parse_setting_overrides, read_settings

    from tt_bio import bindcraft2

    path = args.settings or str(pathlib.Path(bindcraft.__file__).resolve().parents[1]
                                / "examples/pdl1.json")
    common = [f"campaign_seed={args.seed}", f"max_trajectories={args.trajectories}",
              "trajectory_only=true", f"project_folder={args.project}"]
    if args.gates == "zero":
        common += ["min_plddt_screen=0", "min_plddt_refine=0", "min_iptm_anneal=0",
                   "min_plddt_anneal=0", "min_iptm_harden=0", "min_plddt_harden=0",
                   "min_plddt_mutate=0", "min_iptm_mutate=0"]
    if args.steps != "default":
        common += [f"{k}_steps={v}" for k, v in (kv.split("=") for kv in args.steps.split(","))]
    if args.binder_lengths:
        lo, hi = (int(v) for v in args.binder_lengths.split(","))
        span = [cleaned_campaign_settings(read_settings(path, parse_setting_overrides(
            common + [f"binder_lengths=[{n},{n}]"]))) for n in (lo, hi)]
        s = cleaned_campaign_settings(read_settings(path, parse_setting_overrides(
            common + [f"binder_lengths=[{lo},{hi}]"])))
        return f"{lo}-{hi}", s, [bindcraft2.design_tokens(t) for t in span]
    best = None
    for length in range(40, 400):
        s = cleaned_campaign_settings(read_settings(path, parse_setting_overrides(
            common + [f"binder_lengths=[{length},{length}]"])))
        n = bindcraft2.design_tokens(s)
        if n == args.tokens:
            best = (length, s)
        elif n > args.tokens:
            break
    if best is None:
        raise SystemExit(f"no binder length puts {path} on {args.tokens} tokens")
    return (*best, [args.tokens])


def _cycles(garbage, ttnn, paths=4):
    """What the cyclic collector freed: a type histogram, and the shortest reference loop
    through a few of the objects that hold device memory, named attribute by attribute."""
    ids = {id(o): o for o in garbage}
    hist = collections.Counter(f"{type(o).__module__}.{type(o).__qualname__}" for o in garbage)

    def edge(a, b):
        if isinstance(a, dict):
            for k, v in a.items():
                if v is b:
                    return f"[{k!r}]"[:60]
        for k in getattr(type(a), "__slots__", ()) or ():
            if getattr(a, k, None) is b:
                return f".{k}"
        if getattr(a, "__dict__", None) is b:
            return ".__dict__"
        if isinstance(a, (list, tuple)):
            return "[i]"
        if type(a).__name__ == "cell":
            return ".cell_contents"
        if type(a).__name__ == "function":
            if a.__closure__ is b:
                return f"<{a.__qualname__}>.__closure__{a.__code__.co_freevars}"
            return f"<{a.__qualname__}>->"
        return "->"

    def loop(start):
        prev, frontier = {id(start): None}, [start]
        while frontier:
            nxt = []
            for o in frontier:
                for r in gc.get_referents(o):
                    if r is start:
                        chain, cur = [start], o
                        while cur is not start:
                            chain.append(cur)
                            cur = prev[id(cur)]
                        chain.append(start)
                        chain = chain[::-1][:-1] + [start]
                        return " ".join(f"{type(a).__qualname__}{edge(a, b)}" for a, b in
                                        zip(chain, chain[1:]))
                    if id(r) in ids and id(r) not in prev:
                        prev[id(r)] = o
                        nxt.append(r)
            frontier = nxt
        return None

    loops = collections.Counter()
    wanted = [o for o in garbage if type(o).__name__ in ("Tensor", "ReluTransition", "_Node")]
    for o in wanted[:40] + garbage[:20]:
        text = loop(o)
        if text:
            loops[text] += 1
    tt = sum(1 for o in garbage if isinstance(o, ttnn.Tensor))
    return {"objects": len(garbage), "ttnn_tensors": tt, "types": hist.most_common(30),
            "loops": loops.most_common(paths)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--params", required=True, help="AlphaFold 2 params directory")
    ap.add_argument("--settings", help="BindCraft 2 settings JSON (default: examples/pdl1.json)")
    ap.add_argument("--tokens", type=int, default=352)
    ap.add_argument("--trajectories", type=int, default=6)
    ap.add_argument("--steps", default="screen=4,refine=2,anneal=4,harden=1,mutate=2")
    ap.add_argument("--binder-lengths", help="A,B: draw each trajectory's binder length from [A, B]")
    ap.add_argument("--gates", choices=("zero", "stock"), default="zero")
    ap.add_argument("--resident", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=("fixed", "unfixed"), default="fixed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--dry", action="store_true", help="resolve the settings, open no card")
    args = ap.parse_args()
    if args.dry:
        from tt_bio import bindcraft2
        binder, settings, tokens = _settings(args)
        print(f"binder {binder} aa -> {tokens} tokens")
        print(bindcraft2.stage_gates(settings))
        return 0

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()

    from bindcraft import campaign

    import census
    from perf import clocksample
    from tt_bio import bindcraft2, pair_mm

    binder, settings, tokens = _settings(args)
    forget = pair_mm.forget
    if args.arm == "unfixed":
        pair_mm.forget = lambda: None

    loads = [0]
    trunk_init = bindcraft2._Trunk.__init__

    def counted(self, *a, **k):
        loads[0] += 1
        trunk_init(self, *a, **k)

    bindcraft2._Trunk.__init__ = counted

    out = {"arm": args.arm, "tokens": tokens, "binder": binder, "gates": args.gates, "resident": args.resident,
           "steps": args.steps, "trajectories_asked": args.trajectories, "seed": args.seed,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "rows": []}
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(out, indent=1))

    with clocksample.during(period=5.0) as clock, \
            bindcraft2.campaign_predictor(checkpoints=args.params, resident=args.resident) as build:
        import ttnn

        from tt_bio.tenstorrent import get_device
        device = get_device()
        evo, extra = build.evoformer, build.extra_msa

        def roots():
            r = [(f"trunk {n}", t) for n, t in list(build.pool._trunks.items())]
            r += [("evoformer tapes", evo._tapes), ("evoformer masks",
                                                    (evo._mask_dev, evo._pair_mask_dev))]
            if extra is not None:
                r += [("extra-MSA tapes", extra._tapes), ("extra-MSA masks", extra._pair_mask_dev)]
            return r + census.module_roots()

        def programs():
            return int(device.num_program_cache_entries())

        def block_table():
            try:
                return [(int(b["size"]), b["allocated"] == "yes")
                        for b in ttnn.get_memory_view(device, ttnn.BufferType.DRAM).block_table]
            except TypeError:  # some ttnn builds cannot convert the table; read tt-metal's own dump
                ttnn.dump_device_memory_state(device, "bc2_boundary_")
                lines = (pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH", ".")) / "generated" / "reports"
                         / "bc2_boundary_detailed_memory_usage.csv").read_text().splitlines()
                rows = lines[lines.index("Block table:", lines.index(",DRAM")) + 2:]
                table = []
                for line in rows:  # Block Address Size PrevID NextID Allocated
                    f = line.split()
                    if len(f) != 6 or not f[0].isdigit():
                        break
                    table.append((int(f[2]), f[5] == "yes"))
                return table

        def blocks():
            table = block_table()
            live = [n for n, used in table if used]
            small = [n for n in live if n < 2 ** 20]   # per-bank sizes; a tensor here is >= 1 MiB/bank
            return {"blocks_live": len(live), "blocks_free": len(table) - len(live),
                    "small_live": len(small), "small_bytes_per_bank": sum(small)}

        def probe(phase, traj, **extra_fields):
            d = census.dram(device)
            row = {"phase": phase, "traj": traj, "t": time.time(), **extra_fields,
                   "held": d["held"], "free": d["free"], "largest_per_bank": d["largest_per_bank"],
                   "card": d["total"], "loads": loads[0], "wt_entries": len(pair_mm._WT),
                   "trunks_alive": sum(isinstance(o, bindcraft2._Trunk) for o in gc.get_objects()),
                   "trunks_pooled": len(build.pool._trunks), "programs": programs(), **blocks()}
            if phase != "after":
                row["census"] = census.census(roots())
            out["rows"].append(row)
            save()
            print(f"[bc2_memory] {phase:>6} traj {traj}: held {d['held'] / GIB:.3f} GiB, "
                  f"free {d['free'] / GIB:.3f} GiB, largest {d['largest_per_bank'] / 2**20:.1f} "
                  f"MB/bank, loads {loads[0]}, _WT {len(pair_mm._WT)}", flush=True)

        real = campaign.run_trajectory
        count = [0]

        def run_trajectory(*a, **k):
            count[0] += 1
            n = count[0]
            probe("before", n)
            t0, l0, c0 = time.time(), loads[0], dict(evo.calls)
            try:
                return real(*a, **k)
            finally:
                probe("after", n, seconds=round(time.time() - t0, 1), traj_loads=loads[0] - l0,
                      calls={k: evo.calls[k] - c0[k] for k in c0})

        campaign.run_trajectory = run_trajectory
        probe("open", 0)
        try:
            bindcraft2.run_campaign(settings, args.project, trajectories_per_card=1,
                                    af2_weights=args.params)
            out["ended"] = "completed"
        except BaseException as e:  # noqa: BLE001 -- the refusal IS the result in the unfixed arm
            out["ended"] = f"{type(e).__name__}: {str(e)[:600]}"
        finally:
            campaign.run_trajectory = real
        probe("end", count[0])
        held = census.dram(device)["held"]
        gc.set_debug(gc.DEBUG_SAVEALL)   # keep the collected cycles to name what forms them
        t0 = time.time()
        gc.collect()
        gc_seconds = time.time() - t0
        gc.set_debug(0)
        out["cycles"] = _cycles(gc.garbage, ttnn)
        gc.garbage.clear()
        gc.collect()
        t0 = time.time()
        gc.collect()
        out["gc_seconds"] = {"collecting": round(gc_seconds, 3), "idle": round(time.time() - t0, 3),
                             "objects": len(gc.get_objects())}
        after_gc = census.dram(device)["held"]
        forget()
        after_forget = census.dram(device)["held"]
        n_programs, b0 = programs(), blocks()
        device.clear_program_cache()
        after_programs = census.dram(device)["held"]
        out["release"] = {"gc_collect": held - after_gc, "pair_mm_forget": after_gc - after_forget,
                          "program_cache": after_forget - after_programs, "programs": n_programs,
                          "blocks_before": b0, "blocks_after": blocks()}
        out["clock"] = clock.summary()
        out["clock_line"] = clock.line(0)   # tt-smi numbers a pinned chip 0, whatever its node
        save()
        print(f"[bc2_memory] end: gc.collect freed {(held - after_gc) / GB:.3f} GB, "
              f"pair_mm.forget then freed {(after_gc - after_forget) / GB:.3f} GB, clearing "
              f"{n_programs} cached programs freed {(after_forget - after_programs) / GB:.3f} GB; "
              f"{out['ended']}; {out['clock_line']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

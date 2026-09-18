#!/usr/bin/env python3
"""Is the per-forward program-cache clear at `tt_bio/boltz2.py:5731` load-bearing?

`8859d7e1` added it "to avoid stale program-cache state across variable-shape sweeps", so the
seconds it costs are not free until that claim is tested. The risk case is not one fold, it is the
SECOND fold in the same process on a DIFFERENT-sized target, which is exactly what the platform's
worker loop does: one `_WorkerState`, one device, many targets. This folds that sequence.

Two independent tests, because a digest match alone cannot distinguish "the cache was correctly
keyed" from "the second target happened to rebuild anyway":

1. DIGESTS. The same size sequence is folded twice, once with the production clear and once with
   `TT_BIO_BOLTZ2_KEEP_PROGRAM_CACHE=1`, and every CIF sha256 and plDDT must match position for
   position. A cache change must not move a digest at all, so bit-exactness is the cheap check
   here and it is taken.

2. FIRING, not grepping. `ttnn` exposes `set_program_cache_misses_allowed(False)`, which makes a
   cache MISS throw instead of silently building. With it off:
     - re-folding the SAME target must not throw. That proves every program the fold needs is
       already resident, i.e. the cache genuinely serves a repeat and the clear is deleting
       something real.
     - folding a DIFFERENT-sized target MUST throw. That proves a new shape takes a NEW cache key
       rather than silently reusing the previous target's programs, which is the staleness
       `8859d7e1` was defending against. If it does NOT throw, the clear is load-bearing and this
       row stops.
   Reading the bucketing code cannot answer either question; only firing can.
"""
from __future__ import annotations
import argparse, json, os, shutil, socket, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "perf/c12_host_decomp"), str(ROOT / "scripts/gpu_vs_tt"),
                str(ROOT / "perf/other512")]
from evidence import digest, own_nodes, snapshot, validate_snapshot, write_json
from decomp import cif_sha, KEEP_FLAG, install_flag_witness, keep_cache_flag, FLAG_QUERIES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--holder", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="512,298,512,256",
                    help="fold sequence in ONE process, in order")
    ap.add_argument("--keep", type=int, choices=[0, 1], required=True,
                    help="1 suppresses the per-forward program-cache clear for every fold")
    ap.add_argument("--misses-test", type=int, default=1)
    # The size used for the "a new shape must MISS" leg. It has to be a size NOT in
    # --sizes: picking one that the sequence already folded makes the leg vacuous, because
    # its programs are already resident and of course nothing misses. That is what the
    # first run of this harness did, and it is also the POSITIVE control for the
    # instrument -- without a leg that throws, "no throw" on the same-size refold does not
    # distinguish a cache hit from a flag that does nothing.
    ap.add_argument("--miss-other-size", type=int, default=None)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    r = dict(pid=os.getpid(), host=socket.gethostname(), node=a.node, sizes=sizes, keep=a.keep,
             rows=[], misses_test=None, errors=[], completed=False,
             started_utc_ns=time.time_ns())
    save = lambda: write_json(out / "result.json", r)
    T = None
    try:
        if socket.gethostname() != "tt-quietbox2":
            raise RuntimeError("wrong host")
        r["before"] = snapshot(a.node)
        validate_snapshot(r["before"], a.node)
        for k, v in (("TT_VISIBLE_DEVICES", str(a.node)), ("TT_BIO_LEASE_CARDS", str(a.node)),
                     ("TT_BIO_LEASE_HOLDER", a.holder)):
            if os.environ.get(k) != v:
                raise RuntimeError(f"wrong {k}: {os.environ.get(k)!r} != {v!r}")
        # Same refusal as decomp.py: the flag is set per fold IN PROCESS, never inherited from
        # the environment, so an env-var arm cannot be mistaken for a measured one.
        bad = {k: v for k, v in os.environ.items()
               if k.startswith("TT_BIO_") and k not in
               ("TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER")}
        if bad:
            raise RuntimeError(f"non-default environment: {bad}")

        import torch, ttnn
        import tt_bio.tenstorrent as T
        import tt_bio.boltz2 as boltz2
        import tt_baseline as B
        from fold_ab_multi import patch_boltz2_cfg
        install_flag_witness(boltz2)
        r["ttnn_version"] = __import__("importlib.metadata", fromlist=["x"]).version("ttnn")
        r["source_commit"] = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
        B.RECYCLING_STEPS, B.SAMPLING_STEPS, B.DIFFUSION_SAMPLES, B.SEED = 3, 200, 1, 0
        patch_boltz2_cfg()
        B._card_info = lambda: {"node": a.node}

        fix = lambda n: ROOT / f"perf/size512/fixtures/cdk2x2_{n}"
        first = sizes[0]
        _fold, meta, state = B.build_fold(
            "boltz2", out / "msa", fix(first).with_suffix(".yaml"),
            fix(first).with_suffix(".a3m"), instrument=False, hoist=False, fast=False,
            trace=False, recycling_steps=3)
        dev = T.get_device()
        if own_nodes() != [f"/dev/tenstorrent/{a.node}"]:
            raise RuntimeError(f"wrong opened device: {own_nodes()}")
        # Every size in the sequence needs its alignment cached up front, or fold N would run an
        # MSA search inside the sequence and the comparison would not be of folds alone.
        for n in sorted(set(sizes)):
            if n != first:
                B.seed_msa_cache(fix(n).with_suffix(".yaml"), fix(n).with_suffix(".a3m"),
                                 out / "msa")
        job_cfg = dict(meta["job_cfg"])
        base_struct = Path(meta["struct_dir"])

        def fold(size, tag, keep):
            """One fold of `size` at position `tag`, with the clear kept or suppressed."""
            sd = out / "cifs" / tag
            sd.mkdir(parents=True)
            cfg = dict(job_cfg, struct_dir=str(sd))
            for p in base_struct.glob("*"):
                p.unlink()
            FLAG_QUERIES.clear()
            before_entries = int(dev.num_program_cache_entries())
            t0 = time.monotonic_ns()
            with keep_cache_flag(bool(keep)):
                metrics, _best, _feats = state.predict_one(fix(size).with_suffix(".yaml"), cfg)
            ttnn.synchronize_device(dev)
            t1 = time.monotonic_ns()
            for p in base_struct.glob("*"):
                if p.is_file():
                    shutil.copyfile(p, sd / p.name)
            row = dict(tag=tag, size=size, keep=bool(keep), elapsed_s=(t1 - t0) / 1e9,
                       plddt=metrics.get("plddt"), cif=cif_sha(sd),
                       entries_before=before_entries,
                       entries_after=int(dev.num_program_cache_entries()),
                       flag_queries=len(FLAG_QUERIES),
                       flag_all_match=bool(FLAG_QUERIES) and
                       all(q == bool(keep) for q in FLAG_QUERIES))
            if not row["flag_all_match"]:
                raise RuntimeError(f"flag witness failed at {tag}: {FLAG_QUERIES[:5]}")
            if len(row["cif"]) != 1:
                raise RuntimeError(f"expected one CIF at {tag}, got {list(row['cif'])}")
            r["rows"].append(row)
            save()
            print(json.dumps({k: row[k] for k in
                              ("tag", "size", "keep", "elapsed_s", "plddt", "entries_before",
                               "entries_after")} | {"cif": list(row["cif"].values())[0][:16]}),
                  flush=True)
            return row

        # a discarded cold fold of the first size, so no arm pays process warmup
        fold(first, "cold", a.keep)
        r["rows"][-1]["discarded"] = "cold fold"
        for i, n in enumerate(sizes):
            fold(n, f"s{i}_{n}", a.keep)

        if a.misses_test:
            m = {"note": "set_program_cache_misses_allowed(False), then fold"}
            same = sizes[-1]
            other = a.miss_other_size
            if other is None or other in sizes:
                raise RuntimeError(
                    "--miss-other-size must be given and must NOT appear in --sizes, or the "
                    f"leg is vacuous: other={other} sizes={sizes}")
            # 1. a repeat of the size just folded must NOT miss
            dev.set_program_cache_misses_allowed(False)
            try:
                fold(same, f"missforbid_same_{same}", 1)
                m["same_size_refold_threw"] = False
            except Exception as e:
                m["same_size_refold_threw"] = True
                m["same_size_error"] = repr(e)[:400]
            finally:
                dev.set_program_cache_misses_allowed(True)
            # 2. a DIFFERENT size must miss, which is what proves shape-keyed entries
            dev.set_program_cache_misses_allowed(False)
            try:
                fold(other, f"missforbid_other_{other}", 1)
                m["other_size_threw"] = False
            except Exception as e:
                m["other_size_threw"] = True
                m["other_size_error"] = repr(e)[:400]
            finally:
                dev.set_program_cache_misses_allowed(True)
            m["other_size"] = other
            m["verdict"] = (
                "shape-keyed: a repeat hits and a NEVER-FOLDED size misses, so the instrument "
                "throws when it should and the repeat genuinely hit"
                if m.get("same_size_refold_threw") is False and m.get("other_size_threw") is True
                else "INCONCLUSIVE or the clear is load-bearing -- read the two flags")
            r["misses_test"] = m
            save()
            print(json.dumps(m), flush=True)
        r["completed"] = True
    except BaseException as e:
        r["errors"].append(repr(e))
        traceback.print_exc()
    finally:
        if T is not None:
            try:
                T.cleanup()
            except BaseException as e:
                r["errors"].append("cleanup: " + repr(e))
        r["finished_utc_ns"] = time.time_ns()
        save()
    return 0 if r["completed"] else 2


if __name__ == "__main__":
    sys.exit(main())

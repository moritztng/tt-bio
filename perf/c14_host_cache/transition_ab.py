#!/usr/bin/env python3
"""Price the program-cache clear at the SHAPE TRANSITION, which is what the worker loop does.

Every fold-level A/B this flag has ever had -- c12-host-decomp's and this row's first queued one --
folds ONE size repeatedly. That measures the same-size repeat, where a `bare` fold rebuilds ~243
programs and c12 read 0.0524 s off a single rep. It is not the production case. The platform's
worker loop holds one `_WorkerState` and folds a queue of differently-sized targets, so every fold
is a shape TRANSITION, and this row's correctness runs turned up an asymmetry that no repeat A/B
can see: under the production clear a 298 aa fold read 14.03 s against a 9.68 s reference while the
512 aa folds in the same arm read normally. Host load does not do that -- it would hit the LONGEST
fold hardest, not the shortest -- so it is measured here rather than argued.

Design, and why it is blocks rather than a plain interleave. Interleaving arms fold by fold is
right for one size and WRONG for two: the keep arm would inherit whatever the clear arm last left
in the cache, which is only that fold's own shape, so the keep arm's next fold would miss and
rebuild and the arm would measure the thing it is supposed to suppress. So each arm runs as a
contiguous BLOCK of reps, block order is a palindrome (keep, clear, clear, keep) so linear drift
cancels, and each keep block DISCARDS its rep 0 as the cache warm-up it needs to reach its own
steady state. A clear block needs no warm-up: by construction every one of its folds rebuilds.

One rep is one 512 aa fold followed by one 298 aa fold, so the 512 column reproduces the repeat
case the earlier A/Bs measured and the 298 column is the new one. Same clock discipline as
c12-host-decomp: ARC FORCE_AICLK before every fold, ~1 kHz aiclk and board power sampled DURING
it, and a fold whose clock did not hold is rejected rather than averaged.
"""
from __future__ import annotations
import argparse, io, json, os, re, shutil, signal, socket, statistics as st
import subprocess, sys, time, traceback, contextlib, gzip
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C12 = ROOT / "perf/c12_host_decomp"
sys.path[:0] = [str(ROOT), str(C12), str(ROOT / "scripts/gpu_vs_tt"),
                str(ROOT / "perf/b2x_host_residual"), str(ROOT / "perf/other512")]
from evidence import (coverage, digest, load_accounting, own_nodes, snapshot,
                      validate_snapshot, write_json)
from force_aiclk import FORCE_AICLK, smc
from decomp import cif_sha, holder_coverage, keep_cache_flag, install_flag_witness, FLAG_QUERIES, lines

RETRY_RE = re.compile(r"retrying narrower|circular-buffer clash|L1 overflow", re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True)
    ap.add_argument("--holder", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="512,298", help="one rep folds these, in order")
    ap.add_argument("--blocks", default="keep,clear,clear,keep")
    ap.add_argument("--reps", type=int, default=3, help="reps per block")
    ap.add_argument("--clock", type=int, default=1350)
    ap.add_argument("--settle-s", type=float, default=0.6)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    blocks = [x for x in a.blocks.split(",") if x]
    if any(b not in ("keep", "clear") for b in blocks):
        raise SystemExit("blocks must be keep|clear")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    r = dict(pid=os.getpid(), host=socket.gethostname(), node=a.node, sizes=sizes, blocks=blocks,
             reps=a.reps, clock=a.clock, rows=[], errors=[], completed=False,
             timer=("synchronize_device; monotonic_ns; state.predict_one incl. CIF write; "
                    "synchronize_device; monotonic_ns"),
             warmup_rule="rep 0 of every keep block is discarded: it is the block's cache warm-up",
             started_utc_ns=time.time_ns())
    save = lambda: write_json(out / "result.json", r)
    T = fd = sampler = mlog = None
    try:
        if socket.gethostname() != "tt-quietbox2":
            raise RuntimeError("wrong host")
        r["before"] = snapshot(a.node)
        validate_snapshot(r["before"], a.node)
        r["load_accounting"] = load_accounting(a.node)
        for k, v in (("TT_VISIBLE_DEVICES", str(a.node)), ("TT_BIO_LEASE_CARDS", str(a.node)),
                     ("TT_BIO_LEASE_HOLDER", a.holder)):
            if os.environ.get(k) != v:
                raise RuntimeError(f"wrong {k}: {os.environ.get(k)!r} != {v!r}")
        bad = {k: v for k, v in os.environ.items()
               if k.startswith("TT_BIO_") and k not in ("TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER")}
        if bad:
            raise RuntimeError(f"non-default environment: {bad}")

        import torch, ttnn
        import tt_bio.tenstorrent as T
        import tt_bio.boltz2 as boltz2
        import tt_baseline as B
        from fold_ab_multi import patch_boltz2_cfg
        install_flag_witness(boltz2)
        import importlib.metadata as IM
        r["ttnn_version"] = IM.version("ttnn")
        r["source_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        r["dirty"] = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short"], text=True)
        B.RECYCLING_STEPS, B.SAMPLING_STEPS, B.DIFFUSION_SAMPLES, B.SEED = 3, 200, 1, 0
        patch_boltz2_cfg()
        B._card_info = lambda: {"node": a.node}
        fix = lambda n: ROOT / f"perf/size512/fixtures/cdk2x2_{n}"
        _f, meta, state = B.build_fold("boltz2", out / "msa", fix(sizes[0]).with_suffix(".yaml"),
                                       fix(sizes[0]).with_suffix(".a3m"), instrument=False,
                                       hoist=False, fast=False, trace=False, recycling_steps=3)
        dev = T.get_device()
        if own_nodes() != [f"/dev/tenstorrent/{a.node}"]:
            raise RuntimeError(f"wrong opened device: {own_nodes()}")
        for n in sorted(set(sizes[1:])):
            B.seed_msa_cache(fix(n).with_suffix(".yaml"), fix(n).with_suffix(".a3m"), out / "msa")
        r["model_meta"] = {k: meta[k] for k in ("hardware", "grid", "card_type", "recycling_steps",
                                                "timed_region") if k in meta}
        job_cfg = dict(meta["job_cfg"])
        base = Path(meta["struct_dir"])

        fd = os.open(f"/dev/tenstorrent/{a.node}", os.O_RDWR | os.O_APPEND)
        mlog = (out / "sampler.log").open("w")
        sampler = subprocess.Popen(
            [sys.executable, str(C12 / "evidence.py"), str(out / "clock.jsonl"),
             str(os.getpid()), str(a.node)],
            stdin=subprocess.PIPE, stdout=mlog, stderr=subprocess.STDOUT)
        time.sleep(0.3)

        def fold(size, arm, block, rep, tag, counted):
            before = snapshot(a.node)
            validate_snapshot(before, a.node, True)
            force = list(smc(fd, FORCE_AICLK, a.clock))
            if force[0] != 0:
                raise RuntimeError(f"FORCE_AICLK({a.clock}) failed: {force}")
            time.sleep(a.settle_s)
            sd = out / "cifs" / tag
            sd.mkdir(parents=True)
            for p in base.glob("*"):
                p.unlink()
            FLAG_QUERIES.clear()
            keep = arm == "keep"
            eb = int(dev.num_program_cache_entries())
            # tt-bio prints its trimul L1 retries; count them per fold, because "the cleared arm
            # takes more retries" is a competing mechanism for any delta and it is free to test.
            cap = io.StringIO()
            ttnn.synchronize_device(dev)
            t0 = time.monotonic_ns()
            with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
                with keep_cache_flag(keep):
                    metrics, _b, _f2 = state.predict_one(fix(size).with_suffix(".yaml"),
                                                         dict(job_cfg, struct_dir=str(sd)))
            ttnn.synchronize_device(dev)
            t1 = time.monotonic_ns()
            text = cap.getvalue()
            for p in base.glob("*"):
                if p.is_file():
                    shutil.copyfile(p, sd / p.name)
            row = dict(tag=tag, arm=arm, block=block, rep=rep, size=size, counted=counted,
                       start_monotonic_ns=t0, end_monotonic_ns=t1, elapsed_s=(t1 - t0) / 1e9,
                       plddt=metrics.get("plddt"), cif=cif_sha(sd),
                       entries_before=eb, entries_after=int(dev.num_program_cache_entries()),
                       retry_lines=len(RETRY_RE.findall(text)),
                       flag_queries=len(FLAG_QUERIES),
                       flag_ok=bool(FLAG_QUERIES) and all(q == keep for q in FLAG_QUERIES),
                       valid=False)
            if not row["flag_ok"]:
                raise RuntimeError(f"flag witness failed at {tag}: {FLAG_QUERIES[:5]}")
            if len(row["cif"]) != 1:
                raise RuntimeError(f"expected one CIF at {tag}, got {list(row['cif'])}")
            time.sleep(0.15)
            row["clock"] = coverage(lines(out / "clock.jsonl"), row, a.clock)
            row["holders"] = holder_coverage(lines(out / "holders.jsonl"), row, os.getpid(), a.node)
            if not row["clock"]["pass"]:
                raise RuntimeError(f"clock artifact or thin coverage: {row['clock']}")
            if not row["holders"]["passed"]:
                raise RuntimeError(f"co-tenancy or thin holder coverage: {row['holders']}")
            row["valid"] = True
            r["rows"].append(row)
            save()
            print(json.dumps({"tag": tag, "arm": arm, "size": size, "counted": counted,
                              "s": round(row["elapsed_s"], 4), "MHz": row["clock"]["max_MHz"],
                              "W": row["clock"]["max_W"], "entries": [eb, row["entries_after"]],
                              "retries": row["retry_lines"],
                              "cif": list(row["cif"].values())[0][:16],
                              "plddt": row["plddt"]}), flush=True)
            del metrics
            return row

        # one discarded cold fold, no arm pays process warmup
        fold(sizes[0], "clear", -1, -1, "cold", False)
        for bi, arm in enumerate(blocks):
            for rep in range(a.reps):
                # rep 0 of a keep block is its cache warm-up: it inherits the previous (clear)
                # block's cache, which holds only that block's last shape, so its folds would
                # rebuild and it would not be measuring the keep arm at all.
                counted = not (arm == "keep" and rep == 0)
                for size in sizes:
                    tag = f"b{bi}_{arm}_r{rep}_{size}"
                    try:
                        fold(size, arm, bi, rep, tag, counted)
                    except Exception as e:
                        r["errors"].append(f"{tag}: {e!r}")
                        r["rows"].append(dict(tag=tag, arm=arm, block=bi, rep=rep, size=size,
                                              counted=counted, valid=False, error=repr(e)))
                        save()
                        traceback.print_exc()
        r["completed"] = True
    except BaseException as e:
        r["errors"].append(repr(e))
        traceback.print_exc()
    finally:
        if fd is not None:
            try:
                r["release_response"] = list(smc(fd, FORCE_AICLK, 0))
            except BaseException as e:
                r["errors"].append("release: " + repr(e))
            finally:
                os.close(fd)
        if sampler is not None:
            try:
                sampler.communicate(b"stop\n", timeout=15)
            except subprocess.TimeoutExpired:
                sampler.terminate(); sampler.wait(timeout=5)
            r["sampler_returncode"] = sampler.returncode
        if mlog:
            mlog.close()
        if T is not None:
            try:
                T.cleanup()
            except BaseException as e:
                r["errors"].append("cleanup: " + repr(e))
        for name in ("clock.jsonl", "holders.jsonl"):
            p = out / name
            if p.exists():
                (out / (name + ".gz")).write_bytes(gzip.compress(p.read_bytes(), mtime=0))
                p.unlink()
        r["finished_utc_ns"] = time.time_ns()
        save()
    return 0 if r["completed"] else 2


if __name__ == "__main__":
    sys.exit(main())

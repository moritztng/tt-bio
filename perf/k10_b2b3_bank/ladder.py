#!/usr/bin/env python3
"""Size ladder for B2 + B3: 298, 512, 768 and 1024 aa fold in both arms and write one digest.

The standing defect class here is a kernel change tuned at 512 aa that OOMs or silently degrades
larger. B2 raises the gated move's writer window from 32 tiles to Ctg*32 and B3 the reader's p/g
buffers from 4 tiles to 32, so both spend L1 per core, and that is what this rung tests. The L1 is
a function of the CHANNEL width, not of N -- `_gated_ctg` picks Ctg from `Ct` alone -- so the
prediction is that the ladder is flat. A prediction is not a measurement.

Pass/fail only. No ratio may be quoted from this file: the arms do not share a wall clock and the
box has co-tenants; the timed number is the op A/B under benchlock.

Per (size, arm): the fold completes, the CIF digest, the gated move's call and reject counters, the
device DRAM high-water, and the (Ctg, CB bytes) the kernel actually built at that size.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, socket, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)

FIX = REPO / "perf" / "size512" / "fixtures"
OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--sizes", default="298,512,768,1024")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    args = ap.parse_args()
    OUT_PATH = args.out
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    import tt_bio.reblock_permute as RB
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    assert "TT_BIO_TRIMUL_GP_BANK_SPLIT" not in os.environ, \
        "the arms are set in-process; an env pin would make both arms the same arm"

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": [g.x, g.y], "tt_bio_file": _TB.__file__,
        "steps": args.steps, "recycles": args.recycles,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        # What a generic_op's circular buffers have to fit into on this part. `GATE_CB_BUDGET` is
        # the fixed ceiling `_gated_ctg` prices against; this is the real one, recorded so the two
        # can be compared instead of trusted.
        "l1_unreserved": int(ttnn.get_max_worker_l1_unreserved_size()),
        "gate_cb_budget": RB.GATE_CB_BUDGET,
    }
    dump()

    work = Path(tempfile.mkdtemp(prefix="k10-b2b3-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    sizes = [int(v) for v in args.sizes.split(",")]
    for n in sizes:
        LEV._seed_msa(FIX / f"cdk2x2_{n}.yaml", (FIX / f"cdk2x2_{n}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("roof-trimul-b2b3", cfg)

    def kernel_cfg() -> dict:
        """What `_build_gated` actually built, read off the descriptor cache, not predicted."""
        out = {}
        for k, e in RB._CACHE_GATED.items():
            out[f"Nt{k[1]}_Cw{k[2]}_C{k[3]}_split{int(k[-2])}{int(k[-1])}"] = {
                "cb_bytes": [int(c.total_size) for c in e["cbs"]]}
        return out

    def fold(n: int, arm: str) -> dict:
        TT.set_trimul_gp_bank_split(arm == "on")
        RB._CACHE_GATED.clear()
        RB.STATS_GATED[0] = RB.STATS_GATED[1] = 0
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        rec = {"aa": n, "arm": arm, "roles": list(TT.gp_roles())}
        t0 = time.perf_counter()
        try:
            metrics, _b, _f = state.predict_one(FIX / f"cdk2x2_{n}.yaml", cfg)
        except Exception as e:                                                  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"[:400]
            rec["verdict"] = "OOM" if ("Out of Memory" in msg or "allocate" in msg) else "FAIL"
            rec["error"] = msg
            rec["fold_s"] = round(time.perf_counter() - t0, 3)
            return rec
        rec["fold_s"] = round(time.perf_counter() - t0, 3)
        cifs = sorted(struct_dir.glob("*.cif"))
        if not cifs:
            rec["verdict"] = "FAIL"; rec["error"] = "no CIF written"; return rec
        dst = args.cifdir / f"{n}_{arm}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy(cifs[0], dst / cifs[0].name)
        rec["verdict"] = "PASS"
        rec["cif_sha256"] = hashlib.sha256(cifs[0].read_bytes()).hexdigest()
        rec["residues"] = sum(1 for ln in cifs[0].read_text().splitlines()
                              if ln.startswith("ATOM") and " CA " in ln)
        rec["plddt"] = metrics.get("complex_plddt", metrics.get("plddt"))
        rec["gated_calls"] = list(RB.STATS_GATED)
        rec["kernel_cfg"] = kernel_cfg()
        # End-of-fold samples, not a high-water: the question this rung answers is whether the
        # fold completes at all, and the CB set B2 and B3 ask for is priced against
        # `l1_unreserved` below, which is the budget a generic_op's circular buffers must fit.
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        rec["dram_used_bytes"] = int(
            (mv.total_bytes_per_bank - mv.total_bytes_free_per_bank) * mv.num_banks)
        l1 = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
        rec["l1_largest_contig_free_per_bank"] = int(l1.largest_contiguous_bytes_free_per_bank)
        rec["loadavg1"] = round(os.getloadavg()[0], 2)
        return rec

    rungs = []
    for n in sizes:
        for arm in ("off", "on"):
            r = fold(n, arm); rungs.append(r)
            print(f"  {n:5d} aa {arm:3s} {r['verdict']:4s} {r.get('fold_s', 0):8.2f}s "
                  f"gated={r.get('gated_calls')} cif {str(r.get('cif_sha256'))[:16]} "
                  f"{r.get('error', '')}", flush=True)
            OUT["rungs"] = rungs; dump()

    by = {}
    for n in sizes:
        rs = [r for r in rungs if r["aa"] == n]
        shas = sorted({r.get("cif_sha256") for r in rs})
        by[n] = {"verdicts": [r["verdict"] for r in rs],
                 "one_digest": len(shas) == 1 and shas[0] is not None,
                 "cif_sha256": shas,
                 "gated_calls": [r.get("gated_calls") for r in rs]}
    OUT["by_size"] = by
    OUT["all_pass"] = all(v["verdicts"] == ["PASS", "PASS"] and v["one_digest"]
                          for v in by.values())
    dump()
    print(json.dumps({"by_size": by, "all_pass": OUT["all_pass"]}, indent=1), flush=True)
    return 0 if OUT["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

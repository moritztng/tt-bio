#!/usr/bin/env python3
"""L1 of `protenix-v2-beat-dgx-h200` §4: price the UNCALIBRATED `ttnn.linear` sites, then sweep them.

Two phases, ONE process, so the sweep replays each site's real `compute_kernel_config` object and
its real dtypes instead of a serialised guess.

  phase 1  census   one 512 aa fold with `ttnn.linear` wrapped. Every call records its site, its
                    operand shapes, its dtypes, its output buffer type and whether the caller
                    already passed a `program_config` or a `core_grid`. The wall is synchronised,
                    so a per-site second is a SHARE of the fold, never a fold gain
                    (`tt-bio-isolated-op-timing-oversync-inflates-cost` prices these ~2x high).

  phase 2  sweep    for the uncalibrated sites only, off-fold, at the shapes the fold actually
                    issued: candidate program configs and core grids against the unconfigured
                    call in the same process, `torch.equal` against its output.

Registered kill gate, written before the run (state/protenix-v2-beat-dgx-h200.md §4 L1):
GO to a fold A/B only if the summed per-call saving over bit-exact winners is >= 0.35 s/fold at
the fold's own call counts. Below that this is a NO-GO with a number.
"""
import argparse, json, statistics, sys, time, traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

SITES = {}
STATE = {"dev": None, "time": True}


def _sh(t):
    return [int(d) for d in t.shape]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--top", type=int, default=12, help="how many uncalibrated sites to sweep")
    ap.add_argument("--min-share", type=float, default=0.05, help="skip sites below this many s")
    ap.add_argument("--maxcfg", type=int, default=20, help="candidates per site (compile-bound)")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--only-unc", action="store_true")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    ORIG = ttnn.linear

    def wrapped(*args, **kw):
        x = args[0] if args else kw.get("input_tensor_a")
        w = args[1] if len(args) > 1 else kw.get("input_tensor_b")
        st = traceback.extract_stack(limit=3)[-2]
        site = f"{Path(st.filename).name}:{st.lineno}"
        key = (f"{site}|in={'x'.join(map(str, _sh(x)))}|w={'x'.join(map(str, _sh(w)))}"
               f"|pc={int(kw.get('program_config') is not None)}"
               f"|cg={int(kw.get('core_grid') is not None)}")
        rec = SITES.get(key)
        if rec is None:
            rec = SITES[key] = {
                "site": site, "in": _sh(x), "w": _sh(w), "n": 0, "s": 0.0,
                "has_pc": kw.get("program_config") is not None,
                "has_cg": kw.get("core_grid") is not None,
                "bias": None if kw.get("bias") is None else _sh(kw["bias"]),
                "x_dtype": str(x.dtype).split(".")[-1], "w_dtype": str(w.dtype).split(".")[-1],
                "dtype": str(kw.get("dtype")).split(".")[-1] if kw.get("dtype") else None,
                "activation": kw.get("activation"),
                "_ckc": kw.get("compute_kernel_config"),
                "_cg": kw.get("core_grid"),
                "_pc": kw.get("program_config"),
                "_memcfg": kw.get("memory_config"),
                "_bias_dtype": None if kw.get("bias") is None else kw["bias"].dtype,
            }
        rec["n"] += 1
        if STATE["time"] and STATE["dev"] is not None:
            ttnn.synchronize_device(STATE["dev"])
            t0 = time.perf_counter()
            out = ORIG(*args, **kw)
            ttnn.synchronize_device(STATE["dev"])
            rec["s"] += time.perf_counter() - t0
        else:
            out = ORIG(*args, **kw)
        rec["out_buf"] = str(out.memory_config().buffer_type).split(".")[-1]
        rec["out_dtype"] = str(out.dtype).split(".")[-1]
        return out

    ttnn.linear = wrapped

    import importlib.metadata as im
    res = {"ttnn": im.version("ttnn"), "host": "qb2", "card": 1, "size": a.size,
           "grid": list(T.COMPUTE_GRID_MAIN),
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "note": "census seconds are synchronised SHARES, not fold gains"}

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold("protenix-v2", ROOT / f".msa_pxlin_{a.size}", tgt, a3m)
    STATE["dev"] = T.get_device()
    fold_s, m = one_fold()
    ttnn.linear = ORIG
    res["fold_s"] = round(fold_s, 3)
    res["n_tokens"] = m.get("n_tokens")
    res["plddt"] = m.get("plddt")

    ordered = sorted(SITES.items(), key=lambda kv: -kv[1]["s"])
    res["sites"] = [{k: v for k, v in r.items() if not k.startswith("_")} | {"key": key}
                    for key, r in ordered]
    for row in res["sites"]:
        row["s"] = round(row["s"], 4)
        row["ms_per_call"] = round(1000 * row["s"] / max(1, row["n"]), 4)
    unc = [(key, r) for key, r in ordered if not r["has_pc"] and not r["has_cg"]]
    res["uncalibrated_total_s"] = round(sum(r["s"] for _, r in unc), 4)
    res["uncalibrated_n_sites"] = len(unc)
    res["calibrated_total_s"] = round(sum(r["s"] for _, r in ordered if r["has_pc"] or r["has_cg"]), 4)
    print(f"\nfold {fold_s:.2f}s tokens={res['n_tokens']} plddt={res['plddt']}", flush=True)
    print(f"{len(ordered)} linear entries, uncalibrated {len(unc)} "
          f"sharing {res['uncalibrated_total_s']:.3f} s (synced), calibrated "
          f"{res['calibrated_total_s']:.3f} s", flush=True)
    for key, r in unc[:24]:
        print(f"  {key}  n={r['n']:5d}  {r['s']:7.3f} s  "
              f"{1000*r['s']/max(1,r['n']):7.4f} ms/call  out={r.get('out_buf')}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))

    if a.no_sweep:
        return

    # ---------------- phase 2: off-fold config sweep on the uncalibrated sites ----------------
    dev = STATE["dev"]
    gx, gy = T.COMPUTE_GRID_MAIN
    WARM, REPS = 2, 5

    def med(fn):
        for _ in range(WARM):
            o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
        ts, o = [], None
        for i in range(REPS):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append(time.perf_counter() - t0)
            if i < REPS - 1:
                ttnn.deallocate(o)
        return statistics.median(ts), o

    def divisors(n, cap=None):
        return [d for d in range(1, n + 1) if n % d == 0 and (cap is None or d <= cap)]

    def cands(mt, kt, nt):
        """Candidate configs. `in0_block_w = kt` throughout: one K block is the unconfigured op's
        accumulation order, which is the mechanism that makes an entry byte-identical. Every entry
        is still checked with `torch.equal` -- the rule proposes, the check disposes."""
        out = [("shipped", "keep")]
        for g in ((gx, gy), (8, 8), (4, 4), (2, 2), (1, 1)):
            out.append((f"core_grid{g[0]}x{g[1]}", ("cg", g)))
        num = gx * gy
        # 1D M-split (the shipped `_pair_proj_program_config` family)
        for pcm in sorted({d for d in divisors(mt) if -(-mt // d) <= num} |
                          {-(-mt // num), -(-(-(-mt // num)) // 5) * 5}):
            if pcm < 1 or pcm > mt or -(-mt // pcm) > num:
                continue
            for obh in divisors(pcm, 8):
                for obw in divisors(nt, 16):
                    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
                    sw = max(w for w in range(min(8 // sh, obw), 0, -1) if obw % w == 0)
                    out.append((f"1d/pcm{pcm}/obh{obh}/obw{obw}", ("1d", pcm, obh, obw, sh, sw)))
        # 2D grid split
        for pcm in divisors(mt):
            if -(-mt // pcm) > gy:
                continue
            for pcn in divisors(nt):
                if -(-nt // pcn) > gx:
                    continue
                for sh in divisors(pcm, 4):
                    for sw in divisors(pcn, 4):
                        if sh * sw > 8:
                            continue
                        out.append((f"2d/pcm{pcm}/pcn{pcn}/{sh}x{sw}", ("2d", pcm, pcn, sh, sw)))
        # Every candidate is a fresh JIT kernel compile (~1-2 s), so the sweep is compile-bound,
        # not run-bound. Keep an even subsample of the legal set rather than all of it.
        seen, uniq = set(), []
        for lab, sp in out:
            if lab not in seen:
                seen.add(lab); uniq.append((lab, sp))
        if len(uniq) <= a.maxcfg:
            return uniq
        head = uniq[:6]                                   # the core_grid arms and the first 1D
        rest = uniq[6:]
        step = len(rest) / float(a.maxcfg - 6)
        return head + [rest[int(i * step)] for i in range(a.maxcfg - 6)]

    def build(spec, kt):
        if spec == "keep":
            return {k: v for k, v in (("core_grid", r["_cg"]), ("program_config", r["_pc"]))
                    if v is not None}
        if spec[0] == "cg":
            return {"core_grid": ttnn.CoreGrid(y=spec[1][1], x=spec[1][0])}
        if spec[0] == "1d":
            _, pcm, obh, obw, sh, sw = spec
            return {"program_config": ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=kt,
                out_subblock_h=sh, out_subblock_w=sw, out_block_h=obh, out_block_w=obw,
                per_core_M=pcm, per_core_N=obw if obw else 1, fuse_batch=True,
                fused_activation=None, mcast_in0=False)}
        _, pcm, pcn, sh, sw = spec
        return {"program_config": ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy), in0_block_w=kt,
            out_subblock_h=sh, out_subblock_w=sw, out_block_h=pcm, out_block_w=pcn,
            per_core_M=pcm, per_core_N=pcn, transpose_mcast=False, fused_activation=None)}

    swept, saving = [], 0.0
    torch.manual_seed(0)
    target = unc if a.only_unc else ordered
    for key, r in target[:a.top]:
        if r["s"] < a.min_share:
            continue
        xs, ws = r["in"], r["w"]
        if len(ws) != 2:
            swept.append({"key": key, "skip": "weight is not 2D"})
            continue
        mt = 1
        for d in xs[:-2]:
            mt *= d
        mt *= -(-xs[-2] // 32)
        kt, nt = -(-xs[-1] // 32), -(-ws[-1] // 32)
        if kt != -(-ws[-2] // 32):
            swept.append({"key": key, "skip": "K mismatch"})
            continue
        TMAP = {"BFLOAT16": torch.bfloat16, "FLOAT32": torch.float32}
        dt, wdt = TMAP.get(r["x_dtype"]), TMAP.get(r["w_dtype"])
        if dt is None or wdt is None:
            swept.append({"key": key, "skip": f"dtype {r['x_dtype']}/{r['w_dtype']}"})
            continue
        tdt = {"BFLOAT16": ttnn.bfloat16, "FLOAT32": ttnn.float32}
        memcfg = r["_memcfg"] if r["_memcfg"] is not None else (
            ttnn.L1_MEMORY_CONFIG if r.get("out_buf") == "L1" else ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.from_torch(torch.randn(*xs, dtype=dt), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=tdt[r["x_dtype"]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.from_torch(torch.randn(*ws, dtype=wdt), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=tdt[r["w_dtype"]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        base_kw = {"memory_config": memcfg, "compute_kernel_config": r["_ckc"]}
        if r["dtype"] and tdt.get(r["dtype"]) is not None:
            base_kw["dtype"] = tdt[r["dtype"]]
        if r["_cg"] is not None:
            base_kw["core_grid"] = r["_cg"]
        if r["_pc"] is not None:
            base_kw["program_config"] = r["_pc"]
        if r["activation"]:
            base_kw["activation"] = r["activation"]
        b = None
        if r["bias"]:
            b = ttnn.from_torch(torch.randn(*r["bias"], dtype=dt), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=r["_bias_dtype"],
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            base_kw["bias"] = b
        try:
            base_ms, o = med(lambda: ORIG(x, w, **base_kw))
        except Exception as e:                                                   # noqa: BLE001
            swept.append({"key": key, "skip": f"base threw {type(e).__name__}: {str(e)[:100]}"})
            ttnn.deallocate(x); ttnn.deallocate(w)
            continue
        ref = ttnn.to_torch(o); ttnn.deallocate(o)
        cl = cands(mt, kt, nt)
        row = {"key": key, "n": r["n"], "in": xs, "w": ws, "m_tiles": mt, "k_tiles": kt,
               "n_tiles": nt, "base_ms": round(1e3 * base_ms, 4), "n_candidates": len(cl),
               "arms": []}
        print(f"\n== {key}\n   mt={mt} kt={kt} nt={nt} base {1e3*base_ms:.4f} ms  "
              f"{len(cl)} candidates  n={r['n']}", flush=True)
        for label, spec in cl:
            try:
                kwc = dict(base_kw)
                kwc.pop("core_grid", None); kwc.pop("program_config", None)
                kwc.update(build(spec, kt))
                ms, o = med(lambda: ORIG(x, w, **kwc))
                got = ttnn.to_torch(o); ttnn.deallocate(o)
                row["arms"].append({"cfg": label, "ms": round(1e3 * ms, 4),
                                    "speedup": round(base_ms / ms, 4),
                                    "exact": bool(torch.equal(got, ref))})
            except Exception as e:                                               # noqa: BLE001
                row["arms"].append({"cfg": label, "error": f"{type(e).__name__}: {str(e)[:90]}"})
        ok = sorted([r2 for r2 in row["arms"] if r2.get("exact")],
                    key=lambda r2: -r2["speedup"])
        row["best"] = ok[:5]
        if ok and ok[0]["speedup"] > 1.0:
            gain = (base_ms - ok[0]["ms"] / 1e3) * r["n"]
            row["site_saving_s"] = round(gain, 4)
            saving += gain
            for r2 in ok[:3]:
                print(f"   {r2['cfg']:28s} {r2['ms']:8.4f} ms  {r2['speedup']:.4f}x", flush=True)
            print(f"   -> site saving {gain:.4f} s at n={r['n']}", flush=True)
        else:
            row["site_saving_s"] = 0.0
            print("   no bit-exact config beats the unconfigured call", flush=True)
        swept.append(row)
        ttnn.deallocate(x); ttnn.deallocate(w)
        if b is not None:
            ttnn.deallocate(b)

    res["sweep"] = swept
    res["summed_saving_s"] = round(saving, 4)
    res["kill_gate_s"] = 0.35
    res["verdict"] = "GO to fold A/B" if saving >= 0.35 else "NO-GO"
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    a.out.write_text(json.dumps(res, indent=1))
    print(f"\nSUMMED bit-exact saving {saving:.4f} s/fold against a 0.35 s gate -> "
          f"{res['verdict']}", flush=True)
    print("wrote", a.out, flush=True)


main()

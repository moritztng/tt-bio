#!/usr/bin/env python3
"""Is our Evoformer gradient as accurate at 800 tokens as it is at 288?

`bcw-accept` measured 9 of 11 trajectories at ~800 tokens dying at the screen gate against 1
of 10 at ~352. Two causes are already measured (a miscalibrated gate constant and a smaller
optimisation gain). The third would be a bug we own: a gradient that degrades with the token
axis. `bcw-land` graded the VJP against float64 at 288 only.

Same grade, same harness (`perf/bcx_afgrad/afgrad.py`'s blocks, cotangents, metrics and
device arm; `perf/bcw_land/stack_grade.py`'s `fast_round()` framing), at 800. Split in two
because the float64 reference is CPU work that does not need a chip and the device arm needs
one:

    ref   card-free. The float64 cotangent pass, then per graded block the float64 VJP and
          the torch fp32 / bf16 arms on the same bf16-rounded inputs. Caches the block inputs,
          the cotangents and the float64 gradients.
    dev   needs a Blackhole chip. Replays the cached inputs through the shipped device blocks
          under `bindcraft2.fast_round()`, plus the permuted-cotangent and zero-seed controls.

`ref` runs inside `perf/bcw_vjp800/lowmem.py`, which chunks the five reference modules whose
activations outgrow the pair representation. At 288 none of that is needed and `--floor 288`
reproduces `stack_grade.py`'s path exactly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import resource
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perf.bcx_afgrad import afgrad as A        # noqa: E402
from perf.bcw_vjp800.lowmem import lowmem      # noqa: E402

OUT = ROOT / "perf" / "bcw_vjp800"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')} rss {rss_gb():5.2f}G load {os.getloadavg()[0]:4.1f}]",
          *a, flush=True)


def _ckpt_stack(model, msa, pair, k_extra, k_evo):
    """`afgrad.ref_stack(keep=True)`, with each block checkpointed.

    Without this the float64 graph holds every block's activations at once. With it the graph
    holds one pair tensor per boundary and recomputes a block at a time in the backward, which
    is what makes the cotangent pass fit. The boundaries are ordinary graph tensors either way,
    so `retain_grad` on them still gives the teacher-forcing cotangent.
    """
    from torch.utils.checkpoint import checkpoint
    bounds = []
    for i in range(k_extra):
        bounds.append(("extra", i, None, pair))
        pair = checkpoint(lambda p, i=i: A.ref_extra(model, i, p), pair,
                          use_reentrant=False, preserve_rng_state=False)
    for i in range(k_evo):
        bounds.append(("evo", i, msa, pair))
        msa, pair = checkpoint(lambda m, p, i=i: A.ref_evo(model, i, m, p), msa, pair,
                               use_reentrant=False, preserve_rng_state=False)
    return msa, pair, bounds


# ------------------------------------------------------------------------------ ref

def cmd_ref(a):
    """Staged, because the cotangent pass and the per-block arms do not fit in one process.

    At 288 the whole thing ran in 10.91 GB and the cotangent pass set that peak. The live set
    is roughly n^2 (eight float64 boundaries, two retained boundary grads, one block's residual
    stream, the chunk transients), so 800 is 7.7x that. Running each stage in its own process
    means nothing accumulates across stages and glibc hands the arenas back at exit; the caller
    (`run_ref.sh`) also sets MALLOC_ARENA_MAX, which is most of the gap between the ~3 GB of
    live tensors at 288 and the 10.91 GB the process actually held.

      cot   the float64 stack, the readout loss, the cotangent at each graded block's output.
            Writes `b<j>.cot.pt` (bf16 inputs, float64 cotangents) and `meta_cot.json`.
      arms  one block: the float64 VJP and the torch fp32 / bf16 arms on the same inputs, plus
            the float64 forward. Writes `b<j>.pt` and `b<j>.arms.json`.
      fin   assembles `meta.json` from `meta_cot.json` and the per-block arm files.
      all   cot, then every block's arms, then fin, in this process (the 288 path).
    """
    cache = pathlib.Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    blocks = [int(b) for b in a.blocks.split(",")]
    torch.set_num_threads(a.threads)
    say(f"stage={a.stage} n={a.n} evo={a.evo} extra={a.extra} blocks={blocks} "
        f"chunk={a.chunk} floor={a.floor} threads={a.threads} cache={cache}")
    if a.stage in ("cot", "all"):
        _stage_cot(a, cache, blocks)
    if a.stage in ("arms", "all"):
        for j in ([a.block] if a.stage == "arms" else blocks):
            _stage_arms(a, cache, j)
    if a.stage in ("fin", "all"):
        _stage_fin(a, cache, blocks)


def _stage_cot(a, cache, blocks):
    _, ref = A.load_models(a.params, device_arm=False)
    say("models loaded")
    n = a.n
    # Seed HERE, not before the load: `load_models` draws from the global RNG while it builds
    # the modules (the values are then overwritten by the state dict), and it draws a different
    # amount with and without the device arm. Seeding at the input makes the inputs a function
    # of (seed, n) alone, so every stage and any re-run agree whatever was loaded first.
    torch.manual_seed(a.seed)
    logits = torch.randn(n, 20) * 2.0
    ridx = torch.arange(n)
    with torch.no_grad():
        msa0, pair0 = A.embed(ref["f64"], logits.double(), ridx)
    say(f"embedded msa {tuple(msa0.shape)} pair {tuple(pair0.shape)}")

    with lowmem(chunk=a.chunk, floor=a.floor):
        msa_t = msa0.clone().requires_grad_(True)
        pair_t = pair0.clone().requires_grad_(True)
        del msa0, pair0
        m_out, z_out, bounds = _ckpt_stack(ref["f64"], msa_t, pair_t, a.extra, a.evo)
        say(f"forward done, {len(bounds)} boundaries")
        wm = torch.randn(m_out.shape, dtype=torch.float64) / m_out.numel() ** 0.5
        wz = torch.randn(z_out.shape, dtype=torch.float64) / z_out.numel() ** 0.5
        # block j's output cotangent is boundary j+1's grad, or the readout for the last block
        for j in {j + 1 for j in blocks if j + 1 < len(bounds)}:
            for t in bounds[j][2:]:
                if t is not None and t.requires_grad:
                    t.retain_grad()
        ((wm * m_out).sum() + (wz * z_out).sum()).backward()
        say("cotangent backward done")
        del m_out, z_out, msa_t, pair_t
        gc.collect()

        for j in blocks:
            kind, i, m, z = bounds[j]
            if j + 1 < len(bounds):
                _, _, m1, z1 = bounds[j + 1]
                gm, gz = (m1.grad if kind == "evo" else None), z1.grad
            else:
                gm, gz = (wm if kind == "evo" else None), wz
            assert gz is not None, f"no cotangent for block {j}"
            blob = {"block": f"{kind}{i}", "kind": kind, "i": i,
                    "z_in": A.bf(z).to(torch.bfloat16), "gz": gz,
                    "m_in": A.bf(m).to(torch.bfloat16) if m is not None else None, "gm": gm}
            torch.save(blob, cache / f"b{j}.cot.pt")
            say(f"wrote {cache / f'b{j}.cot.pt'} "
                f"{(cache / f'b{j}.cot.pt').stat().st_size / 2**30:.2f} GB")
            del blob
            gc.collect()
    meta = {"stamp": A.stamp(a.card), "n": a.n, "seed": a.seed, "evo": a.evo, "extra": a.extra,
            "blocks": blocks, "chunk": a.chunk, "floor": a.floor,
            "loss": "fixed random linear readout of (msa_out, pair_out)",
            "lowmem": "perf/bcw_vjp800/lowmem.py", "cot_peak_rss_gb": rss_gb()}
    (cache / "meta_cot.json").write_text(json.dumps(meta, indent=1, default=str))
    say(f"cot done, peak rss {meta['cot_peak_rss_gb']:.2f} GB")


def _stage_arms(a, cache, j):
    _, ref = A.load_models(a.params, device_arm=False)
    cot = torch.load(cache / f"b{j}.cot.pt", weights_only=False)
    kind, i, tag = cot["kind"], cot["i"], cot["block"]
    zin, min_ = cot["z_in"].double(), None
    if cot["m_in"] is not None:
        min_ = cot["m_in"].double()
    gz, gm = cot["gz"], cot["gm"]
    del cot
    say(f"block {tag}: arms")
    with lowmem(chunk=a.chunk, floor=a.floor):
        arms = {}
        for arm in ("f64", "f32", "bf16"):
            mod = ref[arm]
            dt = mod.trunk_dtype
            t0 = time.time()
            if kind == "extra":
                g, _ = A.ref_vjp(lambda x, i=i, mod=mod: A.ref_extra(mod, i, x),
                                 [zin.to(dt)], [gz])
                arms[arm] = {"dz": g[0].double()}
            else:
                g, _ = A.ref_vjp(lambda x, y, i=i, mod=mod: A.ref_evo(mod, i, x, y),
                                 [min_.to(dt), zin.to(dt)], [gm, gz])
                arms[arm] = {"dm": g[0].double(), "dz": g[1].double()}
            del g
            gc.collect()
            say(f"  arm {arm} {time.time() - t0:.1f}s")
        r64 = arms["f64"]
        row = {f"{k}_torch_{arm}": dict(A.cmp(arms[arm][k], r64[k]),
                                        norm_ratio=float(arms[arm][k].norm())
                                        / float(r64[k].norm()))
               for k in r64 for arm in ("f32", "bf16")}
        del arms
        gc.collect()
        with torch.no_grad():
            if kind == "extra":
                fwd = {"z": A.ref_extra(ref["f64"], i, zin)}
            else:
                fm, fz = A.ref_evo(ref["f64"], i, min_, zin)
                fwd = {"m": fm, "z": fz}
    blob = {"block": tag, "kind": kind, "i": i,
            "z_in": zin.to(torch.bfloat16), "gz": gz.to(torch.bfloat16),
            "gz_norm_f64": float(gz.norm()),
            "dz_ref": r64["dz"], "fwd_z": fwd["z"].float()}
    if kind != "extra":
        blob.update(m_in=min_.to(torch.bfloat16), gm=gm.to(torch.bfloat16),
                    gm_norm_f64=float(gm.norm()), dm_ref=r64["dm"],
                    fwd_m=fwd["m"].float())
    torch.save(blob, cache / f"b{j}.pt")
    (cache / f"b{j}.arms.json").write_text(json.dumps({"block": tag, "arms": row,
                                                       "peak_rss_gb": rss_gb()}, indent=1))
    say(f"  wrote {cache / f'b{j}.pt'} "
        f"{(cache / f'b{j}.pt').stat().st_size / 2**30:.2f} GB, peak rss {rss_gb():.2f} GB")


def _stage_fin(a, cache, blocks):
    meta = json.loads((cache / "meta_cot.json").read_text())
    meta["arms"] = {}
    meta["peak_rss_gb"] = meta.get("cot_peak_rss_gb", 0.0)
    for j in blocks:
        one = json.loads((cache / f"b{j}.arms.json").read_text())
        meta["arms"][one["block"]] = one["arms"]
        meta["peak_rss_gb"] = max(meta["peak_rss_gb"], one["peak_rss_gb"])
    (cache / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    say(f"wrote {cache / 'meta.json'}; peak rss over stages {meta['peak_rss_gb']:.2f} GB")


# ------------------------------------------------------------------------------ dev


def cmd_dev(a):
    cache = pathlib.Path(a.cache)
    meta = json.loads((cache / "meta.json").read_text())
    blocks = [int(b) for b in a.blocks.split(",")] if a.blocks else meta["blocks"]
    torch.set_num_threads(a.threads)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio import bindcraft2, triatt_bw as TB
    dm, _ref = A.load_models(a.params, device_arm=True)
    del _ref
    dev = A.Dev(dm)
    ag = dev.ag
    trace = A.clock_trace()
    rows = []
    ctx = bindcraft2.fast_round() if a.memory == "fast" else _nullctx()
    with ctx:
        if a.memory != "fast":
            dev.arm(a.memory)
        for j in blocks:
            blob = torch.load(cache / f"b{j}.pt", weights_only=False)
            kind, i, tag = blob["kind"], blob["i"], blob["block"]
            zin = blob["z_in"].double()
            min_ = blob["m_in"].double() if "m_in" in blob else None
            gz, gm = blob["gz"].double(), blob.get("gm")
            gm = gm.double() if gm is not None else None

            def device_vjp(gm_, gz_):
                gc.collect()
                zl = dev.leaf(zin)
                ml = dev.leaf(min_) if min_ is not None else None
                with dev.tt.tape():
                    if kind == "extra":
                        zo = dev.extra(i, zl)
                        roots, seeds = [zo], [dev.seed(gz_, zo)]
                    else:
                        mo, zo = dev.evo(i, ml, zl, None)
                        roots, seeds = [mo, zo], [dev.seed(gm_, mo), dev.seed(gz_, zo)]
                census = A.node_census(ag, roots)
                ag.backward(roots, seeds)
                out = {"dz": dev.grad(zl, zin.shape)}
                fwd = {"z": dev.down(zo.value, zin.shape)}
                if ml is not None:
                    out["dm"] = dev.grad(ml, min_.shape)
                    fwd["m"] = dev.down(mo.value, min_.shape)
                del roots, seeds, zo, zl, ml
                gc.collect()
                return out, fwd, census

            t0 = time.time()
            d, fwd, census = device_vjp(gm, gz)
            t1 = time.time()
            row = {"block": tag, "n": meta["n"], "reach": census["reach"],
                   "nodes": census["nodes"], "secs": round(t1 - t0, 2),
                   "aiclk": A.window(trace, t0, t1), "memory": a.memory}
            for key in d:
                r = blob[f"d{key[1]}_ref"]
                row[key] = dict(A.cmp(d[key], r),
                                norm_ref=float(r.norm()),
                                norm_ratio=float(d[key].norm()) / float(r.norm()))
                row.update({k: v for k, v in meta["arms"][tag].items()
                            if k.startswith(f"{key}_torch")})
            for key in fwd:
                row[f"fwd_{key}"] = A.cmp(fwd[key], blob[f"fwd_{key}"])
            if a.controls:
                perm = lambda t: t.flatten()[torch.randperm(t.numel())].reshape(t.shape)
                dp, _, _ = device_vjp(perm(gm) if gm is not None else None, perm(gz))
                row["permuted"] = {k: A.cmp(dp[k], blob[f"d{k[1]}_ref"]) for k in dp}
                d0, _, _ = device_vjp(torch.zeros_like(gm) if gm is not None else None,
                                      torch.zeros_like(gz))
                row["zero_seed_max_abs"] = {k: float(v.abs().max()) for k, v in d0.items()}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del blob
            gc.collect()
    A.save(f"vjp800_n{meta['n']}{'_' + a.tag if a.tag else ''}.json",
           {"stamp": A.stamp(a.card), "ref": meta, "triatt_bw": dict(TB.STATS), "rows": rows})


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


# ------------------------------------------------------------------------------ cli


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ref", cmd_ref), ("dev", cmd_dev)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--n", type=int, default=800)
        p.add_argument("--evo", type=int, default=8)
        p.add_argument("--extra", type=int, default=0)
        p.add_argument("--blocks", default="0,3,7")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--card", type=int, default=0)
        p.add_argument("--params", default=A.DEFAULT_PARAMS)
        p.add_argument("--cache", default=str(OUT / "cache_n800"))
        if name == "ref":
            p.add_argument("--chunk", type=int, default=64)
            p.add_argument("--floor", type=int, default=128)
            p.add_argument("--stage", default="all", choices=("all", "cot", "arms", "fin"))
            p.add_argument("--block", type=int, default=0)
        else:
            p.add_argument("--tag", default="")
            p.add_argument("--memory", default="fast")
            p.add_argument("--controls", action="store_true")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

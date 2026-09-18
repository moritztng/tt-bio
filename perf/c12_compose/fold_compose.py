#!/usr/bin/env python3
"""The composed fold A/B for the campaign's two GO levers: TT_BIO_DIT_COND_HOIST + TT_BIO_UNFUSED_SILU.

Both levers concluded GO at the op/block level with a fold owed. `c12-cond-hoist-block-timing` read
+0.2134 s on a wall around the token-level DiffusionTransformer; `c12-unfused-silu-bh` read
+0.2843 s from each executed shape's own measured price times its own executed call count. Neither
number is a fold number. This harness takes one.

Four arms plus an A/A, because the standing bar approves a STACK as a stack and never by summing
individual readings, so the composed value has to be measured against the two singles in the SAME
session:

    base   hoist=0 silu=0   the shipped default
    silu   hoist=0 silu=1
    hoist  hoist=1 silu=0
    both   hoist=1 silu=1
    base   again, second position in the same rep -> the session's own A/A floor

Both flags are module-level globals read at CALL time (`tt_bio.tenstorrent._B2_DIT_COND_HOIST` and
`._UNFUSED_SILU`), so an arm is selected by setting the attribute per fold inside one process, one
device open, one program cache and one model load. Pinning either env var is refused: a pinned value
is baked in at import and would serve every arm.

Three things this harness does because they have already cost this campaign a session:

1. The cold rep runs the FULL arm list and is discarded. `c12-cond-hoist-block-timing` paid
   +2.3461 s of first-call kernel JIT inside a timed arm, 4.7x the whole effect being measured, and
   the silu row's cold fold read 92.514 s. `both` gets its own cold pass too: its JIT set is the
   union of the two levers' paths, not either one's.
2. `_cond_weights()` is forced BEFORE the first timed rep and walled, so its one-time 0.34-0.57 s
   concatenation is a recorded number outside the timing rather than a tax on whichever arm
   happened to touch it first. `cond_build_n` per fold proves it was not rebuilt.
3. Per-arm witnesses, because a flag set is not a lever fired. `hoist_norms` counts hoisted
   token-level stacks at the site that decides to take one; `silu_calls` counts the standalone
   in-place `ttnn.silu` that only exists on the unfused path (tenstorrent.py:8221 is the file's only
   `ttnn.silu` call). An arm whose witness is absent is VOID and its timing may not be reasoned
   about.

    fold_compose.py --out <json> --reps 5 --arms base,silu,hoist,both,base --size 512
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as _md
import json
import os
import shutil
import socket
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified
import clk  # noqa: E402

FLAGS = {"hoist": ("TT_BIO_DIT_COND_HOIST", "_B2_DIT_COND_HOIST"),
         "silu": ("TT_BIO_UNFUSED_SILU", "_UNFUSED_SILU")}
# The eltwise lever lives on `tt_bio.eltwise_fusion`, not on `tenstorrent`, and it is TWO flags
# gating two sites of one mechanism (AdaLN's gated conditioning write-back and the attention gate's
# residual add). They are set and scored together because that is how the owning row priced them.
EFLAGS = {"cond_muladd": ("TT_BIO_FUSE_COND_MULADD", "FUSE_COND_MULADD"),
          "attn_gate_add": ("TT_BIO_FUSE_ATTN_GATE_ADD", "FUSE_ATTN_GATE_ADD")}
# arm -> (hoist, silu, eltwise). `default` touches no attribute: it measures the SHIPPED default
# rather than a value this driver sets, which is the only arm that can say what the default does.
ARMS = {"base": (False, False, False), "silu": (False, True, False),
        "hoist": (True, False, False), "both": (True, True, False),
        "eltwise": (False, False, True), "all3": (True, True, True),
        # Pairwise, to localise which two levers break the three-lever stack. `all3` collapses
        # plDDT 0.864 -> 0.368 reproducibly, so one of these two pairs carries it.
        "hoist_elt": (True, False, True), "silu_elt": (False, True, True), "default": None}


def check(arms, T):
    """Refuse to measure a lever this checkout does not have, or one pinned by the environment.

    Setting an attribute nothing reads succeeds silently, so every arm would run the same code and
    the stack would score as a null for a reason that has nothing to do with the levers.
    """
    for arm in arms:
        assert arm in ARMS, f"unknown arm {arm!r}, known: {sorted(ARMS)}"
    for name, (env, attr) in FLAGS.items():
        assert env not in os.environ, f"{env} may not be pinned; the arm is set per fold"
        assert hasattr(T, attr), f"this checkout has no tt_bio.tenstorrent.{attr}"
        assert getattr(T, attr) is False, f"{attr} does not ship off; base would not be the default"
    assert hasattr(T, "DiffusionTransformer"), "no DiffusionTransformer to gate"
    if any(ARMS[a] is not None and ARMS[a][2] for a in arms):
        import tt_bio.eltwise_fusion as EF
        for name, (env, attr) in EFLAGS.items():
            assert env not in os.environ, f"{env} may not be pinned; the arm is set per fold"
            assert hasattr(EF, attr), f"this checkout has no tt_bio.eltwise_fusion.{attr}"
            assert getattr(EF, attr) is False, (
                f"{attr} does not ship off; base would not be the default")
        # The fusion helper the lever routes through must itself be live, or both arms run the
        # unfused chain and the lever scores as a null for a reason that is not the lever.
        assert EF.FUSE_MASK_ADD is True, "FUSE_MASK_ADD off: mask_add would not fuse on either arm"
        assert hasattr(T, "mask_add"), "tenstorrent.py does not import mask_add"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifs", type=Path, default=None)
    ap.add_argument("--arms", default="base,silu,hoist,both,base")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--size", default="512")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mhz", type=int, default=1350)
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--cold-reps", type=int, default=1,
                    help="discarded full-arm-list passes before the timed reps")
    ap.add_argument("--palindrome", action="store_true",
                    help="reverse the interior arms on odd reps, so a within-rep drift cancels")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    check(arms, T)

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    AB.SEED = args.seed
    dev = get_device()
    nodes = clk.nodes_open_by_this_process()
    assert len(nodes) == 1, f"expected exactly one open chip, got {nodes}: an unpinned open brings " \
                            "up every visible chip and breaks every co-tenant"
    node = nodes[0]
    # The AICLK sets the fold time on this part: 800 MHz reads 21.90 s and 1350 reads 14.69 s on the
    # same 512 aa cell. An unforced arm pair measures the governor, not the lever.
    clk.force(args.mhz, nodes)
    g = dev.compute_with_storage_grid_size()

    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "card_node": node, "grid": [g.x, g.y],
        "arch": str(dev.arch()), "torch": torch.__version__,
        "ttnn": _md.version("ttnn"), "python": sys.executable,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_at_start": os.getloadavg(), "clock_forced_mhz": args.mhz,
        "protocol": {"arms": arms, "reps": args.reps, "cold_reps": args.cold_reps,
                     "size": args.size, "seed": args.seed,
                     "recycling_steps": args.recycles, "sampling_steps": args.steps},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="c12-compose-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    fx = AB.FIX / f"cdk2x2_{args.size}.yaml"
    assert fx.is_file(), f"no fixture {fx}"
    AB._seed_msa(fx, (AB.FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c12-compose", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    # --- witnesses -----------------------------------------------------------------------------
    # Both counters are on paths that fire O(10^3) times per fold, so their Python cost is ~ms
    # against a 15 s fold. ttnn.linear is deliberately NOT wrapped: it is called 108,608 times per
    # fold and a wrapper there would cost 0.1-0.2 s, which is 40 % of the effect being measured.
    fired = {"hoist": 0, "silu": 0, "eltwise": 0}
    silu_shapes: dict[str, int] = {}
    _dt_call = T.DiffusionTransformer.__call__

    def _counted_dt(self, *a, **kw):
        if getattr(T, "_B2_DIT_COND_HOIST") and not self.atom_level:
            fired["hoist"] += 1
        return _dt_call(self, *a, **kw)

    T.DiffusionTransformer.__call__ = _counted_dt

    import ttnn as _tn
    _silu = _tn.silu

    def _counted_silu(x, *a, **kw):
        fired["silu"] += 1
        silu_shapes[str(list(x.shape))] = silu_shapes.get(str(list(x.shape)), 0) + 1
        return _silu(x, *a, **kw)

    _tn.silu = _counted_silu

    # `mask_add` is reached from tenstorrent.py ONLY on the two fused branches, so counting it
    # counts exactly the eltwise lever's firings -- 0 on every arm that has it off.
    #
    # The eltwise lever is not on main: `tenstorrent.mask_add` and the two FUSE_* flags live on
    # wk/c12-fused-eltwise-at-pin only, and wrapping them unconditionally is an AttributeError the
    # moment this harness runs against a checkout that does not carry them. An eltwise ARM on such
    # a checkout is still refused loudly by check(); what is dropped here is only the witness for
    # a mechanism this tree does not contain.
    import tt_bio.eltwise_fusion as EF
    HAS_ELTWISE = hasattr(T, "mask_add") and hasattr(EF, "FUSE_COND_MULADD")
    if HAS_ELTWISE:
        _mask_add = T.mask_add

        def _counted_mask_add(*a, **kw):
            fired["eltwise"] += 1
            return _mask_add(*a, **kw)

        T.mask_add = _counted_mask_add

    # --- the one-time conditioning build, walled so it is a number and not a difference ------
    # `_cond_weights()` concatenates the 24 layers' conditioning projections once per PROCESS and
    # caches them on the module. Lazily it lands in whichever fold first takes the hoisted path:
    # `c12-cond-hoist-block-timing` session 1 paid it inside a TIMED arm and that fold's block read
    # 5.3617 s against 3.0156 s for the later hoist folds. Here the cold rep's hoist arm pays it,
    # and `cond_build_n` on every warm fold is the witness that it was not rebuilt -- a rebuild
    # inside a timed arm would otherwise look like the lever getting worse.
    #
    # A walk over `state.model.__dict__` to pre-build it was tried and found 0 of them (the layer
    # stack is not reachable that way), so the build is witnessed rather than hoisted out by hand.
    build = {"n": 0, "s": 0.0}
    dtc = T.DiffusionTransformer
    assert hasattr(dtc, "_cond_weights"), "no _cond_weights to wall"
    _cw = dtc._cond_weights

    def _timed_cw(self):
        if self._cond_w is not None:
            return _cw(self)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        r = _cw(self)
        ttnn.synchronize_device(dev)
        build["n"] += 1
        build["s"] += time.perf_counter() - t
        return r

    dtc._cond_weights = _timed_cw

    cifdir = args.cifs or (work / "cifs")
    cifdir.mkdir(parents=True, exist_ok=True)

    def fold(arm, rep, pos):
        want = ARMS[arm]
        if want is not None:
            setattr(T, "_B2_DIT_COND_HOIST", want[0])
            setattr(T, "_UNFUSED_SILU", want[1])
            if HAS_ELTWISE:
                for _n, (_e, _at) in EFLAGS.items():
                    setattr(EF, _at, want[2])
        fired["hoist"] = fired["silu"] = fired["eltwise"] = 0
        silu_shapes.clear()
        rebuilt = build["n"]
        cfg["seed"] = args.seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        # Load at BOTH ends of the fold. A single reading at the start is blind to a co-tenant that
        # starts mid-fold, which is exactly benchlock's own acquire-time blind spot.
        load0 = os.getloadavg()[0]
        smp = clk.Sampler(node)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(fx, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        aiclk = smp.stop()
        load1 = os.getloadavg()[0]
        cifs = sorted(struct_dir.rglob("*.cif"))
        assert cifs, "no CIF written"
        dst = cifdir / f"{args.size}_r{rep}_p{pos}_{arm}.cif"
        shutil.copy2(cifs[0], dst)
        return {"arm": arm, "rep": rep, "pos": pos, "cold": rep < 0,
                "fold_s": round(wall, 3), "aiclk": aiclk,
                "load0": round(load0, 2), "load1": round(load1, 2),
                "hoist": bool(getattr(T, "_B2_DIT_COND_HOIST")),
                "silu": bool(getattr(T, "_UNFUSED_SILU")),
                "hoist_norms": fired["hoist"], "silu_calls": fired["silu"],
                "eltwise_calls": fired["eltwise"],
                "eltwise": bool(getattr(EF, "FUSE_COND_MULADD", False)),
                "silu_shapes": dict(silu_shapes),
                "cond_build_n": build["n"] - rebuilt,
                "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()[:16],
                "plddt": round(float(metrics.get("plddt",
                                                 metrics.get("confidence_score", 0))), 6),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # A fixed arm order inside a rep aliases any within-rep drift into the arm effect: base sits
    # at both ends and averages the drift out, but silu/hoist/both sit at one position each and do
    # not. Reversing the interior on odd reps gives every interior arm both an early and a late
    # slot across the session, so a linear drift cancels for them too. Position is still recorded
    # per fold, so the balance is checkable rather than asserted.
    def order(rep):
        return arms if (not args.palindrome or rep % 2 == 0) else \
            [arms[0]] + list(reversed(arms[1:-1])) + [arms[-1]]

    for rep in range(-args.cold_reps, args.reps):
        for pos, arm in enumerate(order(rep)):
            r = fold(arm, rep, pos)
            out["runs"].append(r)
            print(f"  rep{rep:<3d} {arm:6s} pos{pos} fold {r['fold_s']:7.3f}s  "
                  f"clk {r['aiclk']['min']}-{r['aiclk']['max']} (n={r['aiclk']['n']})  "
                  f"load {r['load0']:5.2f}->{r['load1']:5.2f}  "
                  f"hoist_norms={r['hoist_norms']:<4d} silu={r['silu_calls']:<5d} "
                  f"elt={r['eltwise_calls']:<5d} "
                  f"sha={r['sha256']} plddt={r['plddt']}"
                  + ("  [COLD, discarded]" if r["cold"] else ""), flush=True)
            dump()
    clk.release()

    built = [r for r in out["runs"] if r["cond_build_n"]]
    out["cond_weights_build"] = {
        "total_s": round(build["s"], 4), "n": build["n"],
        "paid_in": [(r["rep"], r["arm"], r["cond_build_n"]) for r in built],
        "all_in_cold_reps": all(r["cold"] for r in built)}
    print(f"  _cond_weights() built {build['n']}x for {build['s']:.4f}s, in "
          f"{[(r['rep'], r['arm']) for r in built]} "
          f"(all cold: {out['cond_weights_build']['all_in_cold_reps']})", flush=True)
    out["summary"] = summarise(out["runs"], arms, args)
    dump()
    report(out["summary"], args)
    return 0


def summarise(runs, arms, args):
    warm = [r for r in runs if not r["cold"]]
    order = list(dict.fromkeys(arms))
    by = {a: [r["fold_s"] for r in warm if r["arm"] == a] for a in order}
    base = st.median(by["base"])
    s = {"n_warm_folds": len(warm), "base_median_fold_s": round(base, 4), "arms": {}}
    for a in order:
        v = by[a]
        med = st.median(v)
        s["arms"][a] = {
            "n": len(v), "reps_s": v, "median_fold_s": round(med, 4),
            "delta_s_vs_base": round(base - med, 4),
            # cycles at the forced clock: seconds are clock-dependent on this fixture, so the
            # campaign prices every lever in cycles as well.
            "delta_Mcycles": round((base - med) * args.mhz, 1),
            "ratio_vs_base": round(base / med, 5) if med else None,
            "spread_pct": round(100 * (max(v) - min(v)) / med, 2) if med else None,
        }

    s["paired"] = paired(warm, order)

    # The median-of-medians A/A floor is KEPT but is no longer the floor a verdict is read against.
    # It understates the session's own noise because taking medians first cancels the within-rep
    # swings the floor exists to measure: on session s2 it read 0.1080 s where the paired A/A 95 %
    # CI was +/-0.79 s, a 7x understatement. `s["paired"]["aa"]` is the number an effect must clear.
    p0 = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] == 0]
    pn = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] != 0]
    if p0 and pn:
        s["aa_floor"] = {
            "pos0_median_s": round(st.median(p0), 4), "posN_median_s": round(st.median(pn), 4),
            "delta_s": round(abs(st.median(p0) - st.median(pn)), 4),
            "ratio": round(st.median(p0) / st.median(pn), 5),
            "worst_pairwise_delta_s": round(max(abs(a - b) for a, b in zip(p0, pn)), 4)
            if len(p0) == len(pn) else None,
        }

    # SUBADDITIVITY, measured in this session and never inferred: the composed delta against the
    # sum of the two singles taken in the same arm list.
    if all(a in s["arms"] for a in ("silu", "hoist", "both")):  # noqa: median-based, see paired()
        d_s = s["arms"]["silu"]["delta_s_vs_base"]
        d_h = s["arms"]["hoist"]["delta_s_vs_base"]
        d_b = s["arms"]["both"]["delta_s_vs_base"]
        larger = max(d_s, d_h)
        s["subadditivity"] = {
            "silu_delta_s": d_s, "hoist_delta_s": d_h, "both_delta_s": d_b,
            "sum_of_singles_s": round(d_s + d_h, 4),
            "composed_minus_sum_s": round(d_b - (d_s + d_h), 4),
            "fraction_of_sum": round(d_b / (d_s + d_h), 4) if (d_s + d_h) else None,
            "larger_single_s": round(larger, 4),
            "below_larger_single": bool(d_b < larger),
        }

    w = {}
    for a in order:
        rs = [r for r in warm if r["arm"] == a]
        w[a] = {"hoist_norms": sorted({r["hoist_norms"] for r in rs}),
                "silu_calls": sorted({r["silu_calls"] for r in rs}),
                "eltwise_calls": sorted({r.get("eltwise_calls", 0) for r in rs}),
                "silu_shapes": sorted({k for r in rs for k in r["silu_shapes"]}),
                "cond_rebuilt": sorted({r["cond_build_n"] for r in rs}),
                "digests": sorted({r["sha256"] for r in rs}),
                "plddt": sorted({r["plddt"] for r in rs})}
        # An arm is VOID when its lever did not fire. Recorded, not asserted: a void arm still
        # carries information (it says the flag does not reach the code) and the verdict has to be
        # able to quote it.
        want = ARMS[a]
        if want is not None:
            bad = []
            if want[0] and w[a]["hoist_norms"] == [0]:
                bad.append("hoist flag set, hoist_norms 0")
            if not want[0] and w[a]["hoist_norms"] != [0]:
                bad.append(f"hoist flag off, hoist_norms {w[a]['hoist_norms']}")
            if want[1] and w[a]["silu_calls"] == [0]:
                bad.append("silu flag set, no standalone ttnn.silu")
            if not want[1] and w[a]["silu_calls"] != [0]:
                bad.append(f"silu flag off, silu_calls {w[a]['silu_calls']}")
            if want[2] and w[a]["eltwise_calls"] == [0]:
                bad.append("eltwise flags set, no mask_add reached")
            if not want[2] and w[a]["eltwise_calls"] != [0]:
                bad.append(f"eltwise flags off, mask_add {w[a]['eltwise_calls']}")
            w[a]["void"] = bad
    s["witness"] = w
    s["clock"] = {"min": min(r["aiclk"]["min"] for r in warm),
                  "max": max(r["aiclk"]["max"] for r in warm),
                  "samples": sum(r["aiclk"]["n"] for r in warm)}
    s["load"] = {"min": min(min(r["load0"], r["load1"]) for r in warm),
                 "max": max(max(r["load0"], r["load1"]) for r in warm)}
    return s


# two-sided 95 % t critical values; df -> t. Beyond the table the normal value is close enough
# (t(120) = 1.980 against z = 1.960) and is used.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 14: 2.145, 16: 2.120, 18: 2.101, 20: 2.086,
        25: 2.060, 30: 2.042, 35: 2.030, 40: 2.021, 45: 2.014, 50: 2.009, 60: 2.000,
        80: 1.990, 100: 1.984, 120: 1.980}


def _t95(df):
    if df < 1:
        return float("inf")
    return _T95.get(df) or next((v for k, v in sorted(_T95.items()) if k >= df), 1.960)


def paired(warm, order):
    """Every arm differenced against its OWN rep's base mean, pooled across reps with a CI.

    This is the statistic that carries the session's power, and the reason it replaces the
    difference of two medians is measured rather than argued: session s2's medians said
    base 14.8235 s, both -0.0135 s, A/A floor 0.1080 s -- a clean-looking NO-GO -- while the same
    session's paired A/A arm had a 95 % CI of +/-0.79 s, so every lever arm sat inside its own
    control and the session could not resolve its pre-registered 0.4977 s at all.

    Two estimates are reported and a verdict needs both to agree in sign:

      mean  paired mean with a t-based 95 % CI. Unbiased, and the interval is the honest width of
            what the session can see.
      min   the per-arm minimum against the base minimum. Contention noise on this box is
            one-sided -- a co-tenant can only ever make a fold slower, never faster -- so the floor
            of each arm's distribution is the least contaminated fold it achieved. It has no clean
            interval, so it is a cross-check on the sign and not a headline.
    """
    reps = sorted({r["rep"] for r in warm})
    interior = [a for a in order if a != "base"]
    acc = {a: [] for a in interior}
    aa, rows = [], []
    for rp in reps:
        g = [r for r in warm if r["rep"] == rp]
        b = sorted([r for r in g if r["arm"] == "base"], key=lambda r: r["pos"])
        if len(b) != 2:
            continue
        bm = (b[0]["fold_s"] + b[1]["fold_s"]) / 2
        row = {"rep": rp, "base_first": b[0]["fold_s"], "base_last": b[1]["fold_s"],
               "base_mean": round(bm, 4), "aa": round(b[0]["fold_s"] - b[1]["fold_s"], 4)}
        aa.append(b[0]["fold_s"] - b[1]["fold_s"])
        for a in interior:
            v = [r for r in g if r["arm"] == a]
            if len(v) == 1:
                acc[a].append(bm - v[0]["fold_s"])
                row[a] = round(bm - v[0]["fold_s"], 4)
        rows.append(row)

    def stat(v, name):
        if len(v) < 2:
            return {"n": len(v), "mean_s": round(v[0], 4) if v else None}
        m = st.mean(v); sd = st.stdev(v); se = sd / len(v) ** 0.5; t = _t95(len(v) - 1)
        return {"n": len(v), "mean_s": round(m, 4), "sd_s": round(sd, 4), "se_s": round(se, 4),
                "t95": t, "ci95_half_width_s": round(t * se, 4),
                "ci95_s": [round(m - t * se, 4), round(m + t * se, 4)],
                "resolved": bool(abs(m) > t * se)}

    bmin = min(r["fold_s"] for r in warm if r["arm"] == "base")
    out = {"n_reps": len(rows), "per_rep": rows,
           "aa": stat(aa, "aa"),
           "arms": {a: dict(stat(acc[a], a),
                            min_fold_s=round(min(r["fold_s"] for r in warm if r["arm"] == a), 4),
                            delta_min_s=round(bmin - min(r["fold_s"] for r in warm
                                                         if r["arm"] == a), 4))
                    for a in interior},
           "base_min_fold_s": round(bmin, 4)}
    def subadd(singles, composed):
        """Composed arm against the sum of its own singles, differenced PER REP.

        Per-rep is the point: it gives the interaction its own interval instead of doing
        arithmetic on point estimates that each carry a CI nobody propagated.
        """
        if composed not in out["arms"] or any(a not in out["arms"] for a in singles):
            return None
        d = {a: out["arms"][a]["mean_s"] for a in (*singles, composed)}
        ssum = sum(d[a] for a in singles)
        n = min(len(acc[a]) for a in (*singles, composed))
        inter = [acc[composed][i] - sum(acc[a][i] for a in singles) for i in range(n)]
        return {"singles": list(singles), "composed_arm": composed,
                "single_deltas_s": {a: round(d[a], 4) for a in singles},
                "sum_of_singles_s": round(ssum, 4), "both_s": round(d[composed], 4),
                "fraction_of_sum": round(d[composed] / ssum, 4) if ssum else None,
                "interaction": stat(inter, "interaction"),
                "below_larger_single": bool(d[composed] < max(d[a] for a in singles)),
                "larger_single_s": round(max(d[a] for a in singles), 4)}

    for key, singles, composed in (("subadditivity", ("silu", "hoist"), "both"),
                                   ("subadditivity3", ("silu", "hoist", "eltwise"), "all3")):
        v = subadd(singles, composed)
        if v is not None:
            out[key] = v
    return out


def report(s, args):
    print(f"\n  base {s['base_median_fold_s']:.4f}s  "
          f"clock {s['clock']['min']}-{s['clock']['max']} MHz "
          f"({s['clock']['samples']} samples)  load {s['load']['min']}-{s['load']['max']}")
    for a, r in s["arms"].items():
        print(f"  {a:6s} n={r['n']} {r['median_fold_s']:8.4f}s  "
              f"{r['delta_s_vs_base']:+8.4f}s  {r['delta_Mcycles']:+8.1f}Mc  "
              f"{r['ratio_vs_base']}x  spread {r['spread_pct']}%")
    if "aa_floor" in s:
        print(f"  A/A floor (medians, UNDERSTATES) {s['aa_floor']['delta_s']:.4f}s  "
              f"{s['aa_floor']['ratio']}x")
    p = s.get("paired")
    if p:
        print(f"\n  PAIRED, each arm against its own rep's base mean, n={p['n_reps']} reps")
        a = p["aa"]
        print(f"  {'A/A':6s} {a['mean_s']:+8.4f}s  sd {a['sd_s']:.4f}  "
              f"95% CI +/-{a['ci95_half_width_s']:.4f}s  <- the floor an effect must clear")
        for k, v in p["arms"].items():
            print(f"  {k:6s} {v['mean_s']:+8.4f}s  sd {v['sd_s']:.4f}  "
                  f"95% CI [{v['ci95_s'][0]:+.4f},{v['ci95_s'][1]:+.4f}]  "
                  f"resolved={v['resolved']}   min-based {v['delta_min_s']:+.4f}s")
        for key in ("subadditivity", "subadditivity3"):
            sb = p.get(key)
            if not sb:
                continue
            i = sb["interaction"]
            print(f"  SUBADDITIVITY paired [{'+'.join(sb['singles'])} -> {sb['composed_arm']}]: "
                  f"{sb['both_s']:+.4f}s vs singles sum "
                  f"{sb['sum_of_singles_s']:+.4f}s = {sb['fraction_of_sum']} of it; "
                  f"interaction {i['mean_s']:+.4f}s 95% CI +/-{i['ci95_half_width_s']:.4f}s "
                  f"(resolved={i['resolved']})")
    if "subadditivity" in s:
        sb = s["subadditivity"]
        print(f"  SUBADDITIVITY  both {sb['both_delta_s']:+.4f}s against silu+hoist "
              f"{sb['sum_of_singles_s']:+.4f}s = {sb['fraction_of_sum']} of the sum, "
              f"{sb['composed_minus_sum_s']:+.4f}s")
        print(f"  below the larger single ({sb['larger_single_s']:.4f}s)? "
              f"{sb['below_larger_single']}")
    for a, r in s["witness"].items():
        print(f"  witness {a:7s} hoist_norms={r['hoist_norms']} silu_calls={r['silu_calls']} "
              f"eltwise_calls={r['eltwise_calls']} "
              f"rebuilt={r['cond_rebuilt']} digests={r['digests']} "
              + (f"VOID: {r['void']}" if r.get("void") else ""))


if __name__ == "__main__":
    raise SystemExit(main())

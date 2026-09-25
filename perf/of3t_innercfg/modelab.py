#!/usr/bin/env python3
"""D55 at model scale: the OF3 trunk's per-parameter gradients, ship vs fix, with an A/A floor.

`VJP.json` and `TIGHT.json` say the config on `softmax_bw_inner`'s numerator reduction is
bit-identical to leaving it off, on the op the trunk runs and over six decades of cancellation
tightness. This carries that to the real thing: the shipped OF3 trunk, its real weights, its
real parameter paths, a taped forward and backward, three arms in one process on one device
open.

  A1  shipped `softmax_bw_inner`
  A2  shipped again -- the A/A FLOOR, which is what any A/B has to beat to mean anything
  B1  the one-line fix: `compute_kernel_config=config or precise_config()` on the numerator

Reported as a worst case LOCATED BY PARAMETER PATH, never as a mean. The comparison code is
`perf/of3t_zerosfill/gradcompare.py`'s, imported rather than copied, so the two rows' numbers
are produced by one scorer.
"""
from __future__ import annotations

import argparse, gc, json, os, socket, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import tt_bio as _tt_bio  # noqa: E402
assert Path(_tt_bio.__file__).resolve().parents[1] == REPO, _tt_bio.__file__

from perf.clocksample import during                        # noqa: E402
from perf.of3t_perf import step as S                       # noqa: E402
from perf.of3t_zerosfill.gradcompare import compare, run_arm  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--exact", default="on", choices=("on", "off"))
    ap.add_argument("--arms", default="ship,ship,fix",
                    help="comma list of ship/fix; every PAIR is compared, so an A/A floor is "
                         "read off adjacent ship arms rather than assumed")
    ap.add_argument("--out", type=Path, default=REPO / "perf/of3t_innercfg/MODELAB.json")
    a = ap.parse_args()

    out = {"doc": __doc__.splitlines()[0], "argv": sys.argv[1:], "arms": {}, "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board": "pc card 0 -- Blackhole p150a"},
        "config": {"crop": a.tokens, "cycles": a.cycles, "exact_training": a.exact}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))  # noqa: E731
    dump()

    with during() as clk:
        try:
            import torch, ttnn
            from tt_bio import autograd as ag
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device
            from tt_bio.autograd import precise_config

            _ship = ag.softmax_bw_inner
            FIRES = {"n": 0}
            # WHICH ROUTE the trunk's attention takes under a tape, counted rather than read
            # off the constructors. `softmax_bw_inner` fires from `triangle_attention` and from
            # `_v_softmax`; if the first is 0 the trunk is not on the fused-SDPA route at all.
            ROUTE = {"triangle_attention": 0, "v_sdpa": 0, "host_f64_softmax": 0}
            _real_ta, _real_hf = ag.triangle_attention, ag.host_f64_softmax
            _real_sdpa = TT._VERBS.get("transformer.scaled_dot_product_attention")

            def _ta(*x, **k):
                ROUTE["triangle_attention"] += 1
                return _real_ta(*x, **k)

            def _hf(*x, **k):
                ROUTE["host_f64_softmax"] += 1
                return _real_hf(*x, **k)

            def _sdpa(*x, **k):
                ROUTE["v_sdpa"] += 1
                return _real_sdpa(*x, **k)

            ag.triangle_attention = _ta
            ag.host_f64_softmax = _hf
            if _real_sdpa is not None:
                TT._VERBS["transformer.scaled_dot_product_attention"] = _sdpa

            def _fix(y, g, dim=-1, config=None):
                """The one-line change: the numerator reduction gets the config too."""
                FIRES["n"] += 1
                cfg = config or precise_config()
                inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True,
                                 compute_kernel_config=cfg)
                if not ag.SOFTMAX_BW_RENORM:
                    return inner
                return ttnn.divide(inner, ttnn.sum(y, dim=dim, keepdim=True,
                                                   compute_kernel_config=cfg))

            def _ship_counting(y, g, dim=-1, config=None):
                FIRES["n"] += 1
                return _ship(y, g, dim=dim, config=config)

            held, _meta = S.capture(a.tokens, out)
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = S.declare_weights(held["trunk"][0], out)

            ctx = ag.exact_training(a.exact == "on")
            ctx.__enter__()
            out["env"]["exact_training_ops"] = list(ag.exact_training_ops())

            spec = [x.strip() for x in a.arms.split(",") if x.strip()]
            ARMS = [(f"{i+1}{x[0].upper()}", _fix if x == "fix" else _ship_counting)
                    for i, x in enumerate(spec)]
            kept = {}
            for tag, impl in ARMS:
                FIRES["n"] = 0
                for _k in ROUTE:
                    ROUTE[_k] = 0
                ag.softmax_bw_inner = impl
                # `softmax_bw` calls the module-global name, so rebinding it here is what the
                # backward closures will see. Counted, because an arm whose site never fired
                # is not an arm -- it is the other arm run twice.
                t0 = time.perf_counter()
                g, meta = run_arm("host", held, dev, params, S, ag, TT, ttnn, torch, a.cycles)
                meta["softmax_bw_inner_fires"] = FIRES["n"]
                meta["route"] = dict(ROUTE)
                meta["exact_softmax_stats"] = dict(ag.EXACT_SOFTMAX_STATS)
                meta["exact_layer_norm_stats"] = dict(ag.EXACT_LAYER_NORM_STATS)
                meta["renorm_stats"] = dict(ag.SOFTMAX_BW_RENORM_STATS)
                meta["sdpa_taped_calls"] = getattr(TT, "SDPA_TAPED_CALLS", None)
                meta["impl"] = "fix" if impl is _fix else "ship"
                meta["wall_s"] = round(time.perf_counter() - t0, 2)
                out["arms"][tag] = meta
                print(f"[{tag}] {meta['impl']} fires={FIRES['n']} fwd {meta['forward_s']}s "
                      f"bwd {meta['backward_s']}s grads {meta['params_with_grad']} "
                      f"route={meta['route']} exact_sm={meta['exact_softmax_stats']} "
                      f"renorm={meta['renorm_stats']}", flush=True)
                kept[tag] = g
                dump()
            ag.softmax_bw_inner = _ship

            # EVERY pair. A single A/A floor cannot tell a fix that moves nothing from a
            # backward whose arms all differ; the matrix can.
            tags = [t for t, _ in ARMS]
            out["comparisons"] = {}
            for i, x in enumerate(tags):
                for y in tags[i + 1:]:
                    lab = (f"{x}({out['arms'][x]['impl']}) vs {y}"
                           f"({out['arms'][y]['impl']})")
                    out["comparisons"][lab] = compare(kept[x], kept[y], torch)
            ctx.__exit__(None, None, None)
        except Exception:                                   # noqa: BLE001
            import traceback
            out["error"] = traceback.format_exc()
            print(out["error"], flush=True)
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()
    for k, v in out.get("comparisons", {}).items():
        print(f"{k:28s} bitident {v['n_bit_identical']:5d}/{v['n_compared']} "
              f"worst {v['worst_rel_l2']:.6g} at {v['worst_rel_l2_param']} "
              f"mincos {v['min_cosine']:.6g} p50 {v['rel_l2_p50']:.3g}", flush=True)
    print("wrote", a.out)
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())

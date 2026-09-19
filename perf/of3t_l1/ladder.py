#!/usr/bin/env python3
"""Does the taped OF3 trunk fit the card at a crop their recipe trains at?

384, 640 and 768 are the three crops upstream uses and nothing smaller exists, so a trunk
that does not fit 384 cannot be trained at all. This runs ONE taped trunk cycle at a crop and
answers in BOTH units at once: the live allocation COUNT and the byte high-water, DRAM and L1
separately. Reporting one without the other is what this campaign's R16 warns against --
between 384 and 640 the count curve overtakes the byte curve, so a rung given in bytes hides
which limit bound it.

The input is the shipped pipeline's own: `perf/of3t_perf/step.py` captures a real
`predict_one` at `OF3Trunk.__call__` and replays it, so there is no second featurisation to
drift. That row's file is imported, not forked.

WHERE it stops matters as much as whether. The trunk runs three stacks of pairformer-shaped
blocks -- the MSA module's 4, the template pair stack's 2 per template, and the 48-block
pairformer -- and the peak is tagged with the stack that was running when it was set, so a
failure names a stack instead of a verb.

    ladder.py --tokens 384 --out perf/of3t_l1/out/ladder_384.json
"""
from __future__ import annotations

import argparse
import gc
import re
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                                    # noqa: E402
from perf.of3t_memory.alloc_profile import Peak, _swap_watch           # noqa: E402
from perf.of3t_l1.holders import l1_holders                            # noqa: E402
from perf.of3t_perf import step as S                                   # noqa: E402


def _clash_addr(msg):
    """The address a circular-buffer clash names, so the holder census can point at it."""
    m = re.search(r"L1 buffer allocated at (\d+)", msg or "")
    return int(m.group(1)) if m else None


def _tag_stacks(peak):
    """Label the peak with the block stack that is running, and count each stack's entries.

    Three stacks, one seam. `Pairformer.__call__` goes through `ops.checkpoint_segment`;
    `MSAModule.__call__` and `TemplatePairStack.__call__` called their blocks directly. Which
    one holds the peak is the whole question, so it is recorded rather than inferred from a
    traceback.
    """
    from tt_bio import openfold3_msa_embedder as ME
    from tt_bio import openfold3_template as TE
    from tt_bio import tenstorrent as TS

    peak.stage = "enter"
    peak.stage_at_peak = None
    seen: dict = {}
    origs: list = []

    def wrap(cls, name, label):
        fn = getattr(cls, name)
        origs.append((cls, name, fn))

        def call(self, *a, _f=fn, _l=label, **k):
            prev, peak.stage = peak.stage, _l
            seen[_l] = seen.get(_l, 0) + 1
            try:
                return _f(self, *a, **k)
            finally:
                peak.stage = prev
        setattr(cls, name, call)

    wrap(ME.MSAModule, "__call__", "msa_module")
    wrap(TE.TemplatePairStack, "__call__", "template_pair_stack")
    wrap(TS.Pairformer, "__call__", "pairformer")
    return seen, origs


def _restore(origs):
    for cls, name, fn in origs:
        setattr(cls, name, fn)


def _rung(peak, census, free):
    """One rung, in both units. `count` is the allocator's live list, not a shape estimate."""
    r = {"dram_peak_b": peak.dram_hw, "l1_peak_b": peak.l1_hw,
         "at_verb": peak.at_verb, "at_stage": peak.stage_at_peak,
         "verb_calls": peak.calls}
    if census:
        r["dram_live_allocs"] = census["DRAM"]["count"]
        r["l1_live_allocs"] = census["L1"]["count"]
        r["dram_largest_b"] = census["DRAM"]["largest_b"]
        r["l1_largest_b"] = census["L1"]["largest_b"]
        r["census"] = census
    if free:
        r["free"] = free
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1,
                    help="trunk cycles to PIN; only the last carries a gradient, so 1 is the "
                         "smallest honest training draw, num_recycles=0")
    ap.add_argument("--backward", action="store_true")
    ap.add_argument("--probe-every", type=int, default=1,
                    help="allocator reads per verb; above 1 subsamples a long run")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "branch": os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            import ttnn
            from ttnn._ttnn import reports
            from tt_bio import autograd as ag
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device

            held, _meta = S.capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = S.declare_weights(trunk, out)
            dump()

            peak = Peak(dev, ttnn, reports)
            peak.every = a.probe_every
            blocks, origs = _tag_stacks(peak)
            _probe = peak.probe

            def probe(verb):
                before = peak.dram_hw
                _probe(verb)
                if peak.dram_hw != before:
                    peak.stage_at_peak = peak.stage
            peak.probe = probe

            out["resting"] = {"dram_b": peak.dram_now(), "census": peak.take_census()}
            gc.collect()
            saved: list = []
            peak.reset()
            peak.stage_at_peak = None

            snap_args, snap_kwargs = held["trunk_snap"]
            args_ = S._rehydrate(snap_args, dev)
            kwargs_ = {k: v for k, v in S._rehydrate(snap_kwargs, dev).items()
                       if k != "progress_fn"}
            trunk.num_cycles = a.cycles
            out["forward"] = {"cycles": a.cycles, "inputs_dram_b": peak.dram_now()}

            z = None
            t0 = time.perf_counter()
            try:
                with ag.tape():
                    _swap_watch(peak, True, saved)
                    try:
                        _s, z = trunk(*args_, **kwargs_)
                    finally:
                        _swap_watch(peak, False, saved)
                ttnn.synchronize_device(dev)
                out["forward"]["ok"] = True
            except Exception as e:                                       # noqa: BLE001
                # The census AT the wall, which is the only moment the answer is the
                # allocator's rather than a reconstruction.
                out["forward"]["ok"] = False
                out["forward"]["error"] = traceback.format_exc()[-6000:]
                out["forward"]["error_head"] = str(e).split("backtrace")[0][:900]
                out["forward"]["stage_at_failure"] = peak.stage
                try:
                    out["forward"]["l1_holders"] = l1_holders(
                        ag, _clash_addr(out["forward"]["error_head"]))
                except Exception:                                        # noqa: BLE001
                    pass
                try:
                    out["forward"]["at_failure"] = {
                        "census": peak.take_census(), "free": peak.free_now(),
                        "dram_b": peak.dram_now()}
                except Exception:                                        # noqa: BLE001
                    pass
            out["forward"]["s"] = round(time.perf_counter() - t0, 2)
            out["forward"].update(_rung(peak, peak.census, peak.free))
            out["forward"]["blocks_entered"] = dict(blocks)
            dump()

            if a.backward and out["forward"].get("ok") and isinstance(z, ag.Tensor):
                import torch
                zr = z.value
                seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                       layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
                peak.reset()
                peak.stage_at_peak = None
                out["backward"] = {"tape_nodes": len(ag._reverse_topo([z]))}
                t0 = time.perf_counter()
                # One outer recompute_scope for the whole backward, so the watcher is not
                # disarmed by `checkpoint`'s per-segment swap (alloc_profile.py documents it).
                rs = TT.recompute_scope()
                rs.__enter__()
                try:
                    _swap_watch(peak, True, saved)
                    try:
                        ag.backward([z], [seed])
                        ttnn.synchronize_device(dev)
                    finally:
                        _swap_watch(peak, False, saved)
                    out["backward"]["ok"] = True
                except Exception as e:                                   # noqa: BLE001
                    out["backward"]["ok"] = False
                    out["backward"]["error"] = traceback.format_exc()[-6000:]
                    out["backward"]["error_head"] = str(e).split("backtrace")[0][:900]
                    out["backward"]["stage_at_failure"] = peak.stage
                    try:
                        out["backward"]["l1_holders"] = l1_holders(
                            ag, _clash_addr(out["backward"]["error_head"]))
                    except Exception:                                        # noqa: BLE001
                        pass
                    try:
                        out["backward"]["at_failure"] = {
                            "census": peak.take_census(), "free": peak.free_now()}
                    except Exception:                                    # noqa: BLE001
                        pass
                finally:
                    rs.__exit__(None, None, None)
                out["backward"]["s"] = round(time.perf_counter() - t0, 2)
                out["backward"].update(_rung(peak, peak.census, peak.free))
                # K52: how many declared weights RECEIVED a gradient, never a global norm.
                out["backward"]["params_with_grad"] = sum(
                    1 for t in params.values() if getattr(t, "grad", None) is not None)
                out["backward"]["params_declared"] = len(params)
            ag.release_pins()
            _restore(origs)
        except Exception:                                                # noqa: BLE001
            out["error"] = traceback.format_exc()[-6000:]
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()

    f = out.get("forward", {})
    b = out.get("backward", {})
    print("tokens %d  fwd %s  dram %.3f GB / %s allocs  L1 %.3f GB / %s allocs  at %s in %s"
          % (a.tokens, "PASS" if f.get("ok") else "FAIL",
             f.get("dram_peak_b", 0) / 1e9, f.get("dram_live_allocs"),
             f.get("l1_peak_b", 0) / 1e9, f.get("l1_live_allocs"),
             f.get("at_verb"), f.get("at_stage") or f.get("stage_at_failure")), flush=True)
    if b:
        print("          bwd %s  dram %.3f GB / %s allocs  L1 %.3f GB / %s allocs  "
              "grads %s/%s"
              % ("PASS" if b.get("ok") else "FAIL",
                 b.get("dram_peak_b", 0) / 1e9, b.get("dram_live_allocs"),
                 b.get("l1_peak_b", 0) / 1e9, b.get("l1_live_allocs"),
                 b.get("params_with_grad"), b.get("params_declared")), flush=True)
    print(out["env"]["aiclk_line"], flush=True)
    for phase in ("forward", "backward"):
        if out.get(phase) and not out[phase].get("ok"):
            print(phase, "failed:", (out[phase].get("error_head") or "")[:400], flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

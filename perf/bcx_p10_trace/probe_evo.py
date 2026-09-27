#!/usr/bin/env python3
"""Leg 1: can the Evoformer forward be trace-captured, is replay equal to eager, and what
does replay save?

Runs the tritraj round unchanged (`duo_round.py`, one trajectory) and hijacks the Evoformer
taped seam on round 2, where every program is compiled. It takes the seam's own inputs from
two consecutive calls (two recycles: same padded shape, different values), then on the card:

  eager   the untaped 48-block forward, enqueue and total seconds, twice (the second is read)
  capture the same forward inside begin/end_trace_capture on persistent input tensors
  replay  copy each call's inputs in place, execute_trace, compare torch.equal to eager

The forward is untaped here (recompute=False, no tape) because it isolates dispatch: the
taped seam's tape is leg 2. Writes one JSON and exits the process, so nothing captured here
leaks into a timed round.
"""
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
sys.path.insert(0, str(_ROOT / "perf" / "bcx_p10_duotraj"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = os.environ.get("TRACE_PROBE_OUT", "probe_evo.json")
REGION = int(os.environ.get("TRACE_PROBE_REGION_MB", "256")) << 20
SEEN: list = []


def aiclk():
    try:
        return int(open("/sys/class/tenstorrent/tenstorrent!" + os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0] + "/tt_aiclk").read().split()[0])
    except Exception:
        return -1


def experiment(evo, slot, calls):
    import torch
    import ttnn
    trunk = evo._trunk(slot)
    dev = trunk.device
    res = {"aiclk_start": aiclk()}
    m0, z0, mask, pair_mask, n = calls[0]
    msa_mask = evo._msa_mask(trunk, mask)
    pm = evo._pair_masks(trunk, pair_mask)
    res["shapes"] = {"m": list(m0.shape), "z": list(z0.shape), "n": n}

    def body(mi, zi):
        # The blocks deallocate what they are handed; the persistent inputs must survive.
        return trunk.evoformer(ttnn.clone(mi), ttnn.clone(zi), msa_mask, pm, recompute=False)

    def host(t):
        return ttnn.from_torch(t.detach().unsqueeze(0).to(torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    eager, times = [], []
    for m, z, *_ in calls:
        for rep in range(2):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            mi, zi = trunk.up(m), trunk.up(z)
            mo, zo = body(mi, zi)
            t1 = time.perf_counter()
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            out = (ttnn.to_torch(mo).clone(), ttnn.to_torch(zo).clone())
            for t in (mi, zi, mo, zo):
                ttnn.deallocate(t)
            times.append({"enqueue_s": t1 - t0, "total_s": t2 - t0, "aiclk": aiclk()})
        eager.append(out)
    res["eager"] = times

    mi, zi = trunk.up(m0), trunk.up(z0)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    mo, zo = body(mi, zi)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    res["capture_s"] = time.perf_counter() - t0

    replays = []
    for rep in range(3):
        for k, (m, z, *_) in enumerate(calls):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            ttnn.copy_host_to_device_tensor(host(m), mi)
            ttnn.copy_host_to_device_tensor(host(z), zi)
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            t1 = time.perf_counter()
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            got = (ttnn.to_torch(mo), ttnn.to_torch(zo))
            em, ez = eager[k]
            replays.append({"input": k, "enqueue_s": t1 - t0, "total_s": t2 - t0,
                            "aiclk": aiclk(),
                            "equal_m": bool(torch.equal(got[0], em)),
                            "equal_z": bool(torch.equal(got[1], ez)),
                            "maxabs_z": float((got[1].float() - ez.float()).abs().max())})
    res["replay"] = replays
    ttnn.release_trace(dev, tid)
    return res


def install():
    from tt_bio import bindcraft2, tenstorrent
    tenstorrent.TRACE_REGIONS["bindcraft2_probe"] = {"blackhole": REGION, "wormhole_b0": REGION}
    tenstorrent.get_device(trace="bindcraft2_probe")
    cls = bindcraft2.EvoformerOnDevice
    taped = cls._taped

    def wrapper(self, slot, *a):
        SEEN.append(self._inputs(*a))
        # calls 0,1 are round 1 (compile); 2,3 are round 2's two recycles.
        if len(SEEN) == 4:
            try:
                res = experiment(self, slot, SEEN[2:4])
            except Exception as exc:
                import traceback
                res = {"error": repr(exc), "tb": traceback.format_exc()}
            res["aiclk_end"] = aiclk()
            res["loadavg"] = os.getloadavg()
            pathlib.Path(OUT).write_text(json.dumps(res, indent=1))
            print(json.dumps(res, indent=1), flush=True)
            os._exit(0)
        return taped(self, slot, *a)
    cls._taped = wrapper


if __name__ == "__main__":
    install()
    import duo_round
    duo_round.main()

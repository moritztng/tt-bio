#!/usr/bin/env python3
"""Leg 2: can the Evoformer backward be traced when its tape crosses other work?

A taped forward's saved tensors are written by the forward and read by the backward, and in a
round other seams run in between. Traced, those tensors sit at addresses the capture chose,
freed afterwards, where any later replay or eager op may write. So the forward trace ends by
copying the tape into a BANK allocated before any capture, and the backward trace starts by
copying it back. Inputs, cotangent seeds, outputs and gradients are pre-allocated the same way,
so a trace owns nothing a later allocation can take from it.

On round 2's Evoformer backward this takes the round's two recycle inputs and two real
cotangents (round 1's and round 2's), then:

  eager   taped forward + backward per (input, cotangent), warm, enqueue and total seconds
  capture forward trace (tape -> bank, outputs -> fixed buffers), backward trace (bank ->
          tape, gradients -> fixed buffers); audits the tape against the live-buffer report
  replay  forward, then an eager taped forward of the OTHER input that is dropped (it writes
          over the freed capture addresses, which is what another trajectory's seam does),
          then backward; outputs and gradients compared torch.equal to eager

Writes one JSON and exits the process.
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

OUT = os.environ.get("TRACE_PROBE_OUT", "probe_bwd.json")
REGION = int(os.environ.get("TRACE_PROBE_REGION_MB", "256")) << 20
CARD = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]


def aiclk():
    try:
        return int(open(f"/sys/class/tenstorrent/tenstorrent!{CARD}/tt_aiclk").read().split()[0])
    except Exception:
        return -1


def experiment(evo, slot, fwd_calls, cots):
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.autograd import _reverse_topo
    trunk = evo._trunk(slot)
    dev = trunk.device
    res = {"aiclk_start": aiclk()}
    _, _, mask, pair_mask, n = fwd_calls[0]
    msa_mask = evo._msa_mask(trunk, mask)
    pm = evo._pair_masks(trunk, pair_mask)

    def forward(mi, zi):
        ml = ag.Tensor(ttnn.clone(mi), requires_grad=True)
        zl = ag.Tensor(ttnn.clone(zi), requires_grad=True)
        with trunk.taped.tape():
            mo, zo = trunk.evoformer(ml, zl, msa_mask, pm, recompute=evo.recompute)
        ag.release_pins()
        return ml, zl, mo, zo

    def tape_of(mo, zo):
        """Every device value the backward can read, once each, in a fixed order."""
        seen, out = set(), []
        for t in _reverse_topo([mo, zo]):
            v = t.value
            if v is None or not v.is_allocated():
                continue
            a = v.buffer_address()
            if a in seen:
                continue
            seen.add(a)
            out.append(v)
        return out

    def host(t):
        return ttnn.from_torch(t.detach().unsqueeze(0).to(torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    def seeds(k, like_m, like_z):
        gm, gz = cots[k]
        return trunk.seed(gm, like_m), trunk.seed(gz, like_z)

    # ---- eager reference, warm (first pass per case discarded)
    eager, et = {}, []
    for k in range(2):
        m, z = fwd_calls[k][0], fwd_calls[k][1]
        for rep in range(2):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            mi, zi = trunk.up(m), trunk.up(z)
            ml, zl, mo, zo = forward(mi, zi)
            t1 = time.perf_counter()
            ttnn.synchronize_device(dev)
            t2 = time.perf_counter()
            fo = (ttnn.to_torch(mo.value).clone(), ttnn.to_torch(zo.value).clone())
            sm, sz = seeds(k, mo, zo)
            ttnn.synchronize_device(dev)
            t3 = time.perf_counter()
            ag.backward([mo, zo], [sm, sz])
            t4 = time.perf_counter()
            ttnn.synchronize_device(dev)
            t5 = time.perf_counter()
            go = (ttnn.to_torch(ml.grad).clone(), ttnn.to_torch(zl.grad).clone())
            ag.release_pins()
            et.append({"case": k, "fwd_enqueue_s": t1 - t0, "fwd_total_s": t2 - t0,
                       "bwd_enqueue_s": t4 - t3, "bwd_total_s": t5 - t3, "aiclk": aiclk()})
            del ml, zl, mo, zo, sm, sz
            ttnn.deallocate(mi)
            ttnn.deallocate(zi)
        eager[k] = (fo, go)
    res["eager"] = et

    # ---- fixed buffers, all allocated before the first capture
    m0, z0 = fwd_calls[0][0], fwd_calls[0][1]
    mi, zi = trunk.up(m0), trunk.up(z0)
    ml, zl, mo, zo = forward(mi, zi)
    ttnn.synchronize_device(dev)
    ref_tape = tape_of(mo, zo)
    bank = [ttnn.clone(v) for v in ref_tape]
    mo_o, zo_o = ttnn.clone(mo.value), ttnn.clone(zo.value)
    sm_i, sz_i = seeds(0, mo, zo)
    # Gradients do not share the leaves' spec, so their buffers come from a real backward.
    ag.backward([mo, zo], [ttnn.clone(sm_i), ttnn.clone(sz_i)])
    gm_o, gz_o = ttnn.clone(ml.grad), ttnn.clone(zl.grad)
    ag.release_pins()
    ml, zl, mo, zo = forward(mi, zi)
    ref_tape = tape_of(mo, zo)
    res["tape"] = {"tensors": len(ref_tape),
                   "elements": int(sum(v.volume() for v in ref_tape)),
                   "memory": sorted({str(v.memory_config().buffer_type) for v in ref_tape}),
                   "dtypes": sorted({str(v.dtype) for v in ref_tape})}
    # Every copy the traces issue compiles here: a program load is a device write, and a write
    # inside a capture is a TT_FATAL.
    for v, b in zip(ref_tape, bank):
        ttnn.copy(v, b)
        ttnn.copy(b, v)
    ttnn.copy(mo.value, mo_o)
    ttnn.copy(zo.value, zo_o)
    ttnn.copy(gm_o, gm_o)
    ttnn.copy(gz_o, gz_o)
    del ml, zl, mo, zo, ref_tape
    ttnn.synchronize_device(dev)

    def live():
        return {(b.address, b.max_size_per_bank) for b in ttnn._ttnn.reports.get_buffers(dev)
                if str(b.buffer_type).endswith("DRAM")}

    before = live()
    t0 = time.perf_counter()
    tid_f = ttnn.begin_trace_capture(dev, cq_id=0)
    ml, zl, mo, zo = forward(mi, zi)
    tape = tape_of(mo, zo)
    if [tuple(v.shape) for v in tape] != [tuple(b.shape) for b in bank]:
        raise RuntimeError("the captured tape does not line up with the bank")
    for v, b in zip(tape, bank):
        ttnn.copy(v, b)
    ttnn.copy(mo.value, mo_o)
    ttnn.copy(zo.value, zo_o)
    ttnn.end_trace_capture(dev, tid_f, cq_id=0)
    # Everything alive now that was not before must be the tape (or the roots/leaves in it).
    extra = live() - before
    tape_addrs = {v.buffer_address() for v in tape}
    res["audit"] = {"live_new": len(extra),
                    "not_in_tape": sorted([a, s] for a, s in extra if a not in tape_addrs)[:20]}
    tid_b = ttnn.begin_trace_capture(dev, cq_id=0)
    for v, b in zip(tape, bank):
        ttnn.copy(b, v)
    ag.backward([mo, zo], [ttnn.clone(sm_i), ttnn.clone(sz_i)])
    ttnn.copy(ml.grad, gm_o)
    ttnn.copy(zl.grad, gz_o)
    ag.release_pins()
    ttnn.end_trace_capture(dev, tid_b, cq_id=0)
    res["capture_s"] = time.perf_counter() - t0
    del ml, zl, mo, zo, tape
    ttnn.synchronize_device(dev)
    res["live_after_capture"] = len(live() - before)

    replays = []
    for rep, k in enumerate([0, 1, 0, 1]):
        m, z = fwd_calls[k][0], fwd_calls[k][1]
        gm, gz = cots[k]
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ttnn.copy_host_to_device_tensor(host(m), mi)
        ttnn.copy_host_to_device_tensor(host(z), zi)
        ttnn.execute_trace(dev, tid_f, cq_id=0, blocking=False)
        t1 = time.perf_counter()
        ttnn.synchronize_device(dev)
        t2 = time.perf_counter()
        fo = (ttnn.to_torch(mo_o), ttnn.to_torch(zo_o))
        # Another seam's work on the freed capture addresses.
        o = fwd_calls[1 - k]
        xi, yi = trunk.up(o[0]), trunk.up(o[1])
        junk = forward(xi, yi)
        ttnn.synchronize_device(dev)
        del junk
        ttnn.deallocate(xi)
        ttnn.deallocate(yi)
        ttnn.synchronize_device(dev)
        t3 = time.perf_counter()
        ttnn.copy_host_to_device_tensor(trunk.ttnn.from_torch(
            gm.detach().reshape(list(sm_i.shape)).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), sm_i)
        ttnn.copy_host_to_device_tensor(trunk.ttnn.from_torch(
            gz.detach().reshape(list(sz_i.shape)).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), sz_i)
        ttnn.execute_trace(dev, tid_b, cq_id=0, blocking=False)
        t4 = time.perf_counter()
        ttnn.synchronize_device(dev)
        t5 = time.perf_counter()
        go = (ttnn.to_torch(gm_o), ttnn.to_torch(gz_o))
        (efm, efz), (egm, egz) = eager[k]
        replays.append({"case": k, "fwd_enqueue_s": t1 - t0, "fwd_total_s": t2 - t0,
                        "bwd_enqueue_s": t4 - t3, "bwd_total_s": t5 - t3, "aiclk": aiclk(),
                        "equal_fwd": bool(torch.equal(fo[0], efm) and torch.equal(fo[1], efz)),
                        "equal_grad_m": bool(torch.equal(go[0], egm)),
                        "equal_grad_z": bool(torch.equal(go[1], egz)),
                        "maxabs_grad_z": float((go[1].float() - egz.float()).abs().max()),
                        "cos_grad_z": float(torch.nn.functional.cosine_similarity(
                            go[1].float().flatten(), egz.float().flatten(), dim=0))})
    res["replay"] = replays
    ttnn.release_trace(dev, tid_f)
    ttnn.release_trace(dev, tid_b)
    return res


FWD: list = []
COT: list = []


def install():
    import numpy as np
    import torch
    from tt_bio import bindcraft2, tenstorrent
    tenstorrent.TRACE_REGIONS["bindcraft2_probe"] = {"blackhole": REGION, "wormhole_b0": REGION}
    tenstorrent.get_device(trace="bindcraft2_probe")
    cls = bindcraft2.EvoformerOnDevice
    taped, backward = cls._taped, cls._backward

    def taped_w(self, slot, *a):
        FWD.append(self._inputs(*a))
        return taped(self, slot, *a)

    def backward_w(self, slot, token, g_msa_np, g_pair_np):
        m, z, _, _, n = FWD[-1]
        gm, gz = torch.zeros(m.shape), torch.zeros(z.shape)
        gm[:, :n] = torch.from_numpy(np.asarray(g_msa_np).copy()).float()
        gz[:n, :n] = torch.from_numpy(np.asarray(g_pair_np).copy()).float()
        COT.append((gm, gz))
        if len(COT) == 2:           # round 2: both recycles' inputs and two real cotangents
            try:
                res = experiment(self, slot, FWD[-2:], [COT[1], COT[0]])
            except Exception as exc:
                import traceback
                res = {"error": repr(exc), "tb": traceback.format_exc()}
            res["aiclk_end"] = aiclk()
            res["loadavg"] = os.getloadavg()
            pathlib.Path(OUT).write_text(json.dumps(res, indent=1))
            print(json.dumps(res, indent=1), flush=True)
            os._exit(0)
        return backward(self, slot, token, g_msa_np, g_pair_np)

    cls._taped, cls._backward = taped_w, backward_w


if __name__ == "__main__":
    install()
    import duo_round
    duo_round.main()

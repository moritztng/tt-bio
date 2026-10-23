"""Triangle attention per forced padded fused pair, or (arm `q10`) with lever triatt_q10 on: OpenDDE at 608 tokens (19 tiles, prime): the padded fused pair the fold serves vs pairs
whose q chunk pads less.

At 608 the gated call is served by `_tri_att_fused_padded`, whose `fused_pairs(padded=True)` offers q chunks of
16/12/8/6/4 tiles and ranks (q256, k640) first: 3 q chunks x 12 heads x 2 batch splits lights all 72 cores, but
pads the q axis to 768 (21 % of the rows computed are padding). q224 (7 tiles) also lights 72 cores and pads to
672 (10 %); `per_core_cost` would rate it 12.5 % cheaper, but 7 is not on the offered tile list. Each arm forces
one pair through the same gated entry point the block calls. q_chunk only splits output rows, so an arm at the
shipped k chunk should be torch.equal to the shipped arm; this records that, rel_rms vs float64, and ms/call.

Arm `q10` runs the ship route with lever `triatt_q10` added, so its picks say whether the lever reaches the device
route at this (seq, heads) at all; arm `+<lever>` does the same for any lever (`+triatt_k1`); `--model protenix` takes Protenix-v2's mode sets for the 4-head calls.
`--qsweep` adds, per mode, one arm per q chunk (2..24 tiles, dividing S unless the ship route is padded) at the k chunk the ship arm picked: every such arm should be
torch.equal to ship, so the fastest one is a bit-exact pick the cost model missed. An arm the kernel refuses is dropped.

usage: TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ta608q.py OUT.json [--seq 608] [--heads 12] [--model opendde]
       [--pair 224x640 ... | q10 | +triatt_k1] [--qsweep]
"""
import argparse, json, os, statistics, time, types

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--seq", type=int, default=608)
ap.add_argument("--pair", nargs="*", default=["224x640", "192x640", "128x640", "320x640", "224x320"]); ap.add_argument("--heads", type=int, default=12)
ap.add_argument("--model", default="opendde", choices=("opendde", "protenix"))
ap.add_argument("--qsweep", action="store_true")
ap.add_argument("--ncall", type=int, default=6); ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import opendde as OD

dev = T.get_device()
S, H, D = a.seq, a.heads, 32
torch.manual_seed(S)
hq, hk, hv, hg = (torch.randn(S, H, S, D) for _ in range(4))
hb = torch.randn(1, H, S, S) * 2
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
q, k, v, g, b = (up(t) for t in (hq, hk, hv, hg, hb))
sc = D ** -0.5
sel = slice(0, S, 23)
q64, k64, v64, g64 = (ttnn.to_torch(t)[sel].double() for t in (q, k, v, g))
# the fused kernel adds the bias before the scale: exp((qk + b - max) * scale)
ref = torch.sigmoid(g64) * (torch.softmax((q64 @ k64.transpose(-1, -2) + ttnn.to_torch(b).double()) * sc, -1) @ v64)
att = types.SimpleNamespace(subtile=False, biased=False, fp32_softmax=False)
if a.model == "protenix":
    from tt_bio import protenix as OD
MODES = {"normal": OD.NORMAL_LEVERS, "fast": OD.WH_FAST_LEVERS}


def gated(q_, k_, v_, b_, g_):
    o = T._tri_att_gated_sdpa(att, q_, k_, v_, b_, sc, g_)
    if o is None:   # the block's own fallback: ungated ladder, then the gate multiply
        u = T._tri_att_sdpa(q_, k_, v_, b_, sc)
        o = ttnn.multiply(u, ttnn.sigmoid(g_))
        u.deallocate()
    return o


_ship_pairs = T._triatt_sdpa.fused_pairs


def call(arm, lv):
    if arm == "q10" or arm.startswith("+"):
        with T.levers(set(lv) | {"triatt_q10" if arm == "q10" else arm[1:]}):
            return gated(q, k, v, b, g)
    if arm:
        qc, kc = (int(x) for x in arm.split("x"))
        T._triatt_sdpa.fused_pairs = lambda *a_, **k_: ((qc, kc),)
    try:
        return gated(q, k, v, b, g)
    finally:
        T._triatt_sdpa.fused_pairs = _ship_pairs


res = {"seq": S, "heads": H, "arch": str(dev.arch()), "grid": list(T.COMPUTE_GRID_MAIN), "modes": {}}
for mode, lv in MODES.items():
    with T.levers(lv):
        out = res["modes"][mode] = {}
        arms = [""] + a.pair
        i = 0
        while i < len(arms):
            to = arms[i]; i += 1
            T.SDPA_CHUNK_PICKS.clear()
            try:
                o = call(to, lv)
                full = ttnn.to_torch(o)
                o.deallocate()
            except Exception as e:   # a forced pair the kernel refuses
                out[str(to)] = {"error": repr(e)[:200]}
                continue
            picks = {str(kk): vv for kk, vv in T.SDPA_CHUNK_PICKS.items()}
            if to == "":
                ship = full
                if a.qsweep:
                    qs, ks = next(iter(T.SDPA_CHUNK_PICKS.values()))[:2]
                    # a dividing route only takes dividing q chunks; a padded one (ship q does not divide S) takes any
                    arms += [f"{32 * t}x{ks}" for t in range(2, 25) if 32 * t != qs and f"{32 * t}x{ks}" not in arms
                             and (S % (32 * t) == 0 or S % qs)]
            got = full[sel].double()
            out[str(to)] = {"equal_ship": bool(torch.equal(full, ship)),
                            "max_abs_vs_ship": float((full.float() - ship.float()).abs().max()), "rel_rms": float(((got - ref).pow(2).mean() / ref.pow(2).mean()).sqrt()),
                            "picks": picks, "ms": []}
        live = [to for to in arms if "error" not in out[str(to)]]
        for _ in range(a.reps):
            for to in live:
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(a.ncall):
                    call(to, lv).deallocate()
                ttnn.synchronize_device(dev)
                out[str(to)]["ms"].append((time.perf_counter() - t0) / a.ncall * 1e3)
        for to in live:
            r = out[str(to)]
            r["ms_median"] = statistics.median(r["ms"])
            print(f"{mode:6s} S{S} {to or 'ship'}: {r['ms_median']:7.3f} ms  equal {r['equal_ship']}  rel_rms {r['rel_rms']:.4e}  picks {r['picks']}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)

"""Small-head SDPA on one Wormhole chip at the two shapes spd-bet-shsdpa is graded on.

  esmc  B7 H15 S544 D64, key-padding mask [B,1,S,S] (ESMC/SaProt language-model attention)
  ta    S512 H12 D32, pair bias [1,H,S,S] (OpenDDE triangle attention)

Each arm times NCALL back-to-back calls between two syncs, REPS rounds, arms interleaved. Accuracy is
rel_rms against a float64 softmax(q k^T * scale + mask) v on a row subset. AICLK is sampled from sysfs
on the opened node every 0.5 s and the window of each arm's timing is kept with it.

usage: TT_VISIBLE_DEVICES=N python perf/shsdpa/bench.py OUT.json [--shape esmc,ta] [--arms a,b] [--lens 544,...]
"""
import argparse, json, os, statistics, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--shape", default="esmc,ta")
ap.add_argument("--arms", default="")
ap.add_argument("--lens", default="544,544,520,480,430,390,300", help="ESMC per-sequence key lengths")
ap.add_argument("--ncall", type=int, default=10)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--bh", default="7,15", help="ESMC batch,heads (300M 7,15; 600M 7,18; SaProt 650M 7,20; 6B 1,40)")
ap.add_argument("--host", action="store_true", help="time eager host dispatch instead of trace replay")
ap.add_argument("--proto", default="", help="extra ESMC proto arms qc:kc:b,h,q;... (qc 0 = whole, split - = factory)")
a = ap.parse_args()

samples = []


def _sampler():
    nodes = [p for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*")]
    while True:
        row = {}
        for p in nodes:
            try:
                row[int(p.name.split("!")[1])] = int((p / "tt_aiclk").read_text().split()[0])
            except Exception:
                pass
        samples.append((time.monotonic(), row))
        time.sleep(0.5)


threading.Thread(target=_sampler, daemon=True).start()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
from tt_bio import runtime, worker as W
from tt_bio.host_controller import worker_payload
slot = runtime.build_local_workers("tenstorrent", [object()], [int(os.environ["TT_VISIBLE_DEVICES"])])[0]
W._apply_tt_environment(worker_payload(slot)); W._bind_host_threads()
import torch, ttnn
import tt_bio.tenstorrent as T
from tt_bio import esmc as E, triatt_sdpa as TS

dev = T.get_device(trace="protenix")


def opened_nodes():
    out = set()
    for fd in os.listdir("/proc/self/fd"):
        try:
            t = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            out.add(int(t.rsplit("/", 1)[1]))
    return sorted(out)


NODES = opened_nodes()


def clock(t0, t1):
    v = sorted(r[n] for ts, r in samples if t0 <= ts <= t1 for n in NODES if n in r)
    return dict(median=v[len(v) // 2], min=v[0], max=v[-1], n=len(v)) if v else None


up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
ARMS, REF = {}, {}

if "esmc" in a.shape:
    (B, H), S, D = map(int, a.bh.split(",")), 544, 64
    lens = ([int(x) for x in a.lens.split(",")] * B)[:B]
    assert max(lens) <= S
    torch.manual_seed(544)
    hq, hk, hv = (torch.randn(B, H, S, D) for _ in range(3))
    mask = torch.zeros(B, S, S)
    for i, li in enumerate(lens):
        mask[i, :, li:] = float("-inf")
    eq, ek, ev = (up(t) for t in (hq, hk, hv))
    em = E._sdpa_mask(mask, dev)
    sc = D ** -0.5
    qd, kd, vd = (ttnn.to_torch(t).double() for t in (eq, ek, ev))
    REF["esmc"] = (torch.softmax(qd @ kd.transpose(-1, -2) * sc + mask.unsqueeze(1).double(), -1) @ vd,
                   [(b, i) for b in range(B) for i in range(lens[b])])
    ARMS["e_ship"] = ("esmc", lambda: E.lm_sdpa(eq, ek, ev, em, sc))
    ARMS["e_stock"] = ("esmc", lambda: T.fused_sdpa(eq, ek, ev, attn_mask=em, scale=sc))
    try:
        from tt_bio import shsdpa as P
        PROTO = {}
        EPAD = P.key_pad(lens, S, dev)      # built outside any capture: an upload is a host write
        ARMS["e_auto"] = ("esmc", lambda: P.sdpa(eq, ek, ev, sc, EPAD))
        if a.proto:
            PROTO = {}
            for t in a.proto.split(";"):
                qc, kc, spl, *dfs = t.split(":")
                dfs = {d: 1 for d in dfs[0].split("+")} if dfs else None
                PROTO[f"e_p{qc}_{kc}_{spl.replace(',', '')}" + ("_" + "_".join(dfs) if dfs else "")] = (
                    int(qc) or None, int(kc), None if spl == "-" else tuple(map(int, spl.split(","))), dfs)
        for nm, (qc, kc, spl, *dfs) in PROTO.items():
            d = dict(dfs[0]) if dfs and dfs[0] else {}
            ckc = (ttnn.MathFidelity.LoFi if d.pop("LOFI", 0) else ttnn.MathFidelity.HiFi2, True, False,
                   bool(d.pop("FULLSYNC", 0)))
            ARMS[nm] = ("esmc", lambda qc=qc, kc=kc, spl=spl, d=(d or None), ckc=ckc, pad=P.key_pad(lens, S, dev, kc):
                        P.sdpa(eq, ek, ev, sc, pad, q_chunk=qc, k_chunk=kc, split=spl, defines=d, ckc=ckc))
    except ImportError:
        pass

if "ta" in a.shape:
    S2, H2, D2 = 512, 12, 32
    torch.manual_seed(512)
    tq, tk, tv = (torch.randn(S2, H2, S2, D2) for _ in range(3))
    tb = torch.randn(1, H2, S2, S2) * 2
    q, k, v, bb = (up(t) for t in (tq, tk, tv, tb))
    sc2 = D2 ** -0.5
    sel = slice(0, S2, 23)
    q64, k64, v64 = (ttnn.to_torch(t)[sel].double() for t in (q, k, v))
    # the fused kernel adds the bias before the scale, so the bias is scaled in the reference too
    REF["ta"] = (torch.softmax((q64 @ k64.transpose(-1, -2) + ttnn.to_torch(bb).double()) * sc2, -1) @ v64, sel)
    HIFI2 = (ttnn.MathFidelity.HiFi2, True, False, False)
    ARMS["t_ship"] = ("ta", lambda: TS.sdpa(q, k, v, bb, sc2, 512, 256, ckc_default=HIFI2, q_split_cap=0, padded_mask=True))
    ARMS["t_k1"] = ("ta", lambda: TS.sdpa(q, k, v, bb, sc2, 256, 512, ckc_default=HIFI2, q_split_cap=0, padded_mask=True))

if a.arms:
    ARMS = {k_: v_ for k_, v_ in ARMS.items() if k_ in a.arms.split(",")}


def err(shape, o):
    ref, sel_ = REF[shape]
    if shape == "esmc":
        got = ttnn.to_torch(o).double()
        bi = torch.tensor(sel_)
        g, r = got[bi[:, 0], :, bi[:, 1]], ref[bi[:, 0], :, bi[:, 1]]
    else:
        g, r = ttnn.to_torch(o)[sel_].double(), ref
    return float(((g - r).pow(2).mean() / r.pow(2).mean()).sqrt())


res = {"bh": a.bh, "arch": str(dev.arch()), "grid": list(T.COMPUTE_GRID_MAIN), "nodes": NODES, "lens": a.lens, "arms": {}}
for name, (shape, fn) in list(ARMS.items()):
    try:
        o = fn()
    except Exception as exc:  # an L1 refusal or a compile error on one arm
        print(name, "FAILED", str(exc)[:600], flush=True)
        res["arms"][name] = {"error": str(exc)[:2000]}
        del ARMS[name]
        continue
    assert o is not None, (name, getattr(TS, "REJECTS", None))
    o2 = fn()
    res["arms"][name] = {"shape": shape, "rel_rms": err(shape, o),
                         "repeat_equal": bool(torch.equal(ttnn.to_torch(o), ttnn.to_torch(o2)))}
    o.deallocate(); o2.deallocate()
    print(name, res["arms"][name], flush=True)
# Device time: each arm is one captured call, replayed NCALL times back to back, so the host's per-call Python
# (generic_op descriptor binding) cannot starve the device. --host times eager dispatch instead.
TRACES = {}
if not a.host:
    for name, (shape, fn) in ARMS.items():
        fn().deallocate()   # programs compiled and cached outside the capture
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        o = fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        TRACES[name] = (tid, o)
times = {n: [] for n in ARMS}
t_all0 = time.monotonic()
for _ in range(a.reps):
    for name, (shape, fn) in ARMS.items():
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(a.ncall):
            if a.host:
                fn().deallocate()
            else:
                ttnn.execute_trace(dev, TRACES[name][0], cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        times[name].append((time.perf_counter() - t0) / a.ncall * 1e3)
for name, (tid, o) in TRACES.items():
    res["arms"][name]["trace_rel_rms"] = err(ARMS[name][0], o)
res["timing"] = "host" if a.host else "trace"

# Trace safety: replay e_auto's trace after rewriting its length table and mask row in place with other lengths.
if "e_auto" in TRACES and "esmc" in a.shape:
    lens2 = list(reversed(lens)) if len(set(lens)) > 1 else [max(1, S - 37 * i) for i in range(B)]
    mask2 = torch.zeros(B, S, S)
    for i, li in enumerate(lens2):
        mask2[i, :, li:] = float("-inf")
    ref2 = torch.softmax(qd @ kd.transpose(-1, -2) * sc + mask2.unsqueeze(1).double(), -1) @ vd
    for src, dst in zip(P.key_pad(lens2, S), EPAD):
        ttnn.copy_host_to_device_tensor(src, dst)
    tid, o = TRACES["e_auto"]
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    got = ttnn.to_torch(o).double()
    sel2 = torch.tensor([(b, i) for b in range(B) for i in range(lens2[b])])
    g2, r2 = got[sel2[:, 0], :, sel2[:, 1]], ref2[sel2[:, 0], :, sel2[:, 1]]
    res["trace_relens"] = {"lens2": lens2, "rel_rms": float(((g2 - r2).pow(2).mean() / r2.pow(2).mean()).sqrt())}
    print("trace_relens", res["trace_relens"], flush=True)

res["aiclk"] = clock(t_all0, time.monotonic())
for name, t in times.items():
    res["arms"][name].update(ms_median=statistics.median(t), ms_min=min(t), ms=t)
    print(f"{name:10s} {statistics.median(t):7.3f} ms  rel_rms {res['arms'][name]['rel_rms']:.3e}", flush=True)
print("aiclk", res["aiclk"], flush=True)
json.dump(res, open(a.out, "w"), indent=1)

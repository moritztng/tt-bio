"""Step 2 of bc4-bet-sigmoid: every sigmoid the gates can run, graded against float64 sigmoid of the same bf16 input.

Arms (one process, one chip):
  ttnn.sigmoid accurate and FastApproximate (standalone, bf16)
  gate_bw with fast off and on (the real fp32-dest gate VJP kernel, the same LUT the trimul gate kernels take): with d = o = 1 its
  outputs are s and s(1 - s), so the kernel's own sigmoid and the derivative the backward uses are both graded.
Inputs: a dense grid over [-24, 24], the tails (0, +-1e-6 .. 1e-2, +-30, 88, 1e4) and the production-like N(0, 3^2)
at [1, 224, 224, 128] bf16. A hard stop is any output outside [0, 1], a NaN, or a sign error of s - 0.5.
Times: gate_bw off/on and ttnn.sigmoid accurate/fast at the production shape, per call, synchronized bursts.
"""
import argparse, json, os, statistics, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--burst", type=int, default=10)
A = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio import gate_bw as GB

dev = T.get_device()
CARD = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
out = open(A.out, "w")
CLK = []


def clk_loop():
    while True:
        try:
            CLK.append(int(open(f"/sys/class/tenstorrent/tenstorrent!{CARD}/tt_aiclk").read()))
        except Exception:
            pass
        time.sleep(1)


threading.Thread(target=clk_loop, daemon=True).start()


def emit(r):
    r = dict(r, aiclk=CLK[-1] if CLK else None, load=os.getloadavg())
    out.write(json.dumps(r) + "\n"); out.flush(); print(json.dumps(r), flush=True)


def put(t):
    return ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def get(t):
    return ttnn.to_torch(t).double()


grid = torch.linspace(-24, 24, 32 * 32 * 96)
tails = torch.tensor([0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 16.0,
                      30.0, 88.0, 1e4])
tails = torch.cat([tails, -tails])
special = torch.cat([grid, tails, torch.zeros(32 * 32 - tails.numel() % (32 * 32))])[: (grid.numel() + 1024) // 1024 * 1024]
gen = torch.Generator().manual_seed(0)
inputs = {
    "grid": special.reshape(1, 1, -1, 32),
    "prod_n3": (3 * torch.randn(1, 224, 224, 128, generator=gen)),
}
BINS = [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9)]


def grade(name, x, s, ds=None):
    xr = x.bfloat16().double()
    ref = torch.sigmoid(xr)
    r = dict(arm=name, n=int(x.numel()))
    e = (s - ref)
    r["rel_l2"] = float(e.norm() / ref.norm())
    r["max_abs"] = float(e.abs().max())
    r["argmax_x"] = float(xr.flatten()[e.abs().flatten().argmax()])
    r["out_min"], r["out_max"] = float(s.min()), float(s.max())
    r["outside_01"] = int(((s < 0) | (s > 1)).sum())
    r["nan"] = int(torch.isnan(s).sum())
    r["sign_err"] = int(((s - 0.5) * xr < 0).sum())
    r["bins_max_abs"] = {f"{lo}-{hi}": float(e[(xr.abs() >= lo) & (xr.abs() < hi)].abs().max())
                         for lo, hi in BINS if ((xr.abs() >= lo) & (xr.abs() < hi)).any()}
    if ds is not None:
        dref = ref * (1 - ref)
        de = ds - dref
        r["d_rel_l2"] = float(de.norm() / dref.norm())
        r["d_max_abs"] = float(de.abs().max())
        r["d_zero_where_ref_gt_1e-3"] = int(((ds == 0) & (dref > 1e-3)).sum())
    if name.endswith("grid"):
        r["tails"] = {f"{float(a):g}": float(b) for a, b in zip(xr.flatten()[grid.numel():grid.numel() + tails.numel()],
                                                             s.flatten()[grid.numel():grid.numel() + tails.numel()])}
    emit(r)


for iname, x in inputs.items():
    xt = put(x)
    grade(f"ttnn_accurate/{iname}", x, get(ttnn.sigmoid(xt)))
    grade(f"ttnn_fast/{iname}", x, get(ttnn.sigmoid(xt, mode=ttnn.SigmoidMode.FastApproximate)))
    ones = put(torch.ones_like(x))
    # the binary gates' forward: multiply with a fused sigmoid, as shipped and with the approx flag set
    for aname, act in (("binary_act_accurate", ttnn.UnaryOpType.SIGMOID),
                       ("binary_act_approx", ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID, 4.0, 1.0))):
        try:
            grade(f"{aname}/{iname}", x, get(ttnn.multiply(ones, xt, input_tensor_b_activations=[act])))
        except Exception as e:
            emit(dict(arm=f"{aname}/{iname}", error=f"{type(e).__name__}: {str(e)[:300]}"))
    for fast in (False, True):
        do, dg = GB.gate_bw(ones, ones, xt, fast=fast)
        grade(f"gate_bw_{'lut' if fast else 'accurate'}/{iname}", x, get(do), get(dg))
        ttnn.deallocate(do); ttnn.deallocate(dg)

def timed(name, f):
    for _ in range(2):
        r = f(); [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(A.reps):
        t0 = time.perf_counter()
        for _ in range(A.burst):
            r = f(); [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]
        ttnn.synchronize_device(dev)
        ms.append((time.perf_counter() - t0) * 1e3 / A.burst)
    emit(dict(timing=name, ms_median=statistics.median(ms), ms_min=min(ms), ms_max=max(ms), n=len(ms)))


x = inputs["prod_n3"]; xt = put(x); ones = put(torch.ones_like(x))
for fast in (False, True, False):
    timed(f"gate_bw_{'lut' if fast else 'accurate'}", lambda: GB.gate_bw(ones, ones, xt, fast=fast))
timed("ttnn_sigmoid_accurate", lambda: ttnn.sigmoid(xt))
timed("ttnn_sigmoid_fast", lambda: ttnn.sigmoid(xt, mode=ttnn.SigmoidMode.FastApproximate))
# the trimul gate itself: reblock_permute_gated(xw, p=0, g=128, 128) with p = 1 is permute(sigmoid(g)), flag off/on
from tt_bio import reblock_permute as RP, gate_sigmoid as GS
gq = inputs["prod_n3"]
xw = put(torch.cat([torch.ones_like(gq), gq], dim=-1))
for fast in (False, True):
    GS.SIGMOID_FAST = fast
    o = RP.reblock_permute_gated(xw, 0, 128, 128)
    grade(f"gated_move_{'lut' if fast else 'accurate'}/prod_n3", gq.permute(0, 3, 1, 2), get(o))
    ttnn.deallocate(o)
GS.SIGMOID_FAST = False
for fast in (False, True, False):
    GS.SIGMOID_FAST = fast
    timed(f"gated_move_{'lut' if fast else 'accurate'}", lambda: RP.reblock_permute_gated(xw, 0, 128, 128))
GS.SIGMOID_FAST = False

emit(dict(done=True, aiclk_samples=len(CLK), aiclk_median=statistics.median(CLK) if CLK else None,
          aiclk_min=min(CLK) if CLK else None))

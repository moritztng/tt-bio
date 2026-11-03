# triatt_ln.BIAS_HEADS diagnosis: the fused kernel's q, k, v, g and bias with the head-major bias on and off, same
# input, and where they differ (by head, row and column inside the tile, and tile). bh_ab.py found the block output
# unequal; the change moves data only, so any difference is a bug and this says which output carries it.
# bh_diag.py S:c:H:ending ...  (TT_VISIBLE_DEVICES pinned; sets the p300 mesh descriptor itself)
import sys
from pathlib import Path
W = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(W), str(W / "perf"), str(W / "perf/mol1_triattn")]
cells = [tuple(map(int, c.split(":"))) for c in sys.argv[1:]]
sys.argv = [sys.argv[0]]
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import bench as B
import tt_bio.tenstorrent as T, tt_bio.triatt_ln as TL
dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
T._MM_DEFAULT_WIDE = B.ARMS["served"][1]
real = TL.ln_qkvgb_heads
cap = []


def spy(*a, **k):
    r = real(*a, **k)
    if r is not None:
        (q, kk, v), g, b = r
        cap.append([ttnn.to_torch(t).float() for t in (q, kk, v, g, b)])
    return r


TL.ln_qkvgb_heads = spy


def where(d, name):
    bad = d.nonzero()
    if not len(bad):
        return f"{name} equal"
    s = f"{name} {len(bad)}/{d.numel()} differ"
    for ax, lab in ((1, "h"), (2, "i%32"), (3, "j%32")):
        v = bad[:, ax] % 32 if lab.endswith("%32") else bad[:, ax]
        s += f"\n   {lab}: {torch.bincount(v).tolist()}"
    tiles = torch.unique(bad[:, 1] * 10**6 + (bad[:, 2] // 32) * 1000 + bad[:, 3] // 32)
    s += f"\n   tiles (h, it, jt) {len(tiles)}: {[(int(t) // 10**6, int(t) // 1000 % 1000, int(t) % 1000) for t in tiles[:12]]}"
    return s


for S, c, H, end in cells:
    sd = B.weights(c, H, 7)
    torch.manual_seed(S + c + end)
    x = torch.randn(1, S, S, c).bfloat16()
    with T.levers(B.ARMS["served"][0]):
        mod = T.TriangleAttention(32, H, bool(end), sd, ckc, softmax_site="pairformer")
        outs = {}
        for arm in ("0", "1"):
            TL.BIAS_HEADS = {"wormhole": arm == "1", "blackhole": arm == "1"}
            xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            cap.clear()
            mod(xt, add_to_input=True)
            ttnn.deallocate(xt)
            outs[arm] = cap[0] if cap else None
        del mod
    print(f"== {S}:{c}:{H}:{end}", flush=True)
    if outs["0"] is None or outs["1"] is None:
        print("   fused kernel not served", flush=True)
        continue
    for i, n in enumerate("qkvg"):
        a, b = outs["0"][i], outs["1"][i]
        print(f"   {n} {tuple(a.shape)} equal {torch.equal(a, b)}", flush=True)
    b0, b1 = outs["0"][4], outs["1"][4]
    print(f"   bias plain {tuple(b0.shape)} hm {tuple(b1.shape)}", flush=True)
    b0 = b0.reshape(S, S, -1).permute(2, 0, 1).unsqueeze(0)
    b1 = b1[:, :b0.shape[1]]
    print("   " + where((b0 != b1), "bias"), flush=True)
    if not torch.equal(b0, b1):
        # is it a misplaced value (a permutation) or a wrong one?
        print(f"   bias sorted-equal {torch.equal(b0.flatten().sort().values, b1.flatten().sort().values)}"
              f" max_abs {float((b0 - b1).abs().max()):.4f}", flush=True)
T.cleanup()

"""spd-m-esmfold: ESMFold2's MSA pair-weighted averaging on the shared device PWA vs the host einsum it replaces.

Both arms against a float64 reference of the vendored forward (modeling_esmfold2_common.MSAPairWeightedAveraging),
same bf16 inputs and weights, plus warm per-call time synced. ESMFold2 shapes: d_msa 64, d_pair 256, 8 heads x 16.

usage: TT_VISIBLE_DEVICES=<chip> python esmf_mpwa.py OUT [L=512] [M=1024] [REPS=3] [D_M=64]
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
L = int(sys.argv[2]) if len(sys.argv) > 2 else 512
M = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
LOG = open(OUT / "mpwa.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            out[p.name.split("!")[1]] = int((p / "tt_aiclk").read_text().split()[0])
        except Exception:
            pass
    return out


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.esmfold2 as E

torch.manual_seed(0)
dev = T.get_device()
ckc = T.TorchWrapper().compute_kernel_config
D_M = int(sys.argv[5]) if len(sys.argv) > 5 else 64
D_Z, H, HD = 256, 8, 16
bf = lambda t: t.to(torch.bfloat16)
sd = {"norm_single.weight": 1 + 0.1 * torch.randn(D_M), "norm_single.bias": 0.1 * torch.randn(D_M),
      "compute_bias.0.weight": 1 + 0.1 * torch.randn(D_Z), "compute_bias.0.bias": 0.1 * torch.randn(D_Z),
      "compute_bias.1.weight": torch.randn(H, D_Z) / D_Z ** 0.5 * 3,
      "Wv.weight": torch.randn(H * HD, D_M) / D_M ** 0.5, "Wgate.weight": torch.randn(H * HD, D_M) / D_M ** 0.5,
      "Wout.weight": torch.randn(D_M, H * HD) / (H * HD) ** 0.5}
sd = {k: bf(v) for k, v in sd.items()}
m = bf(torch.randn(1, L, M, D_M)); z = bf(torch.randn(1, L, L, D_Z))
log(ev="start", L=L, M=M, arch=str(dev.arch()), aiclk=aiclk())


def ref64():
    d = {k: v.double() for k, v in sd.items()}
    mm, zz = m.double(), z.double()
    mn = torch.nn.functional.layer_norm(mm, (D_M,), d["norm_single.weight"], d["norm_single.bias"], 1e-5)
    zn = torch.nn.functional.layer_norm(zz, (D_Z,), d["compute_bias.0.weight"], d["compute_bias.0.bias"], 1e-5)
    attn = torch.softmax(zn @ d["compute_bias.1.weight"].t(), dim=-2)
    v = (mn @ d["Wv.weight"].t()).reshape(1, L, M, H, HD)
    g = torch.sigmoid(mn @ d["Wgate.weight"].t()).reshape(1, L, M, H, HD)
    o = torch.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, g).reshape(1, L, M, H * HD)
    return mm + o @ d["Wout.weight"].t()


class Host(T.Module):
    """origin/main 4e4b105ea esmfold2.MSAPairWeightedAveraging (host einsum), single row block."""

    def __init__(self):
        super().__init__(T.WeightScope(sd), ckc)
        tt = self.torch_to_tt
        self.ns_w, self.ns_b = tt("norm_single.weight"), tt("norm_single.bias")
        self.cb_ln_w, self.cb_ln_b, self.cb_w = tt("compute_bias.0.weight"), tt("compute_bias.0.bias"), tt("compute_bias.1.weight")
        self.Wv, self.Wgate, self.Wout = tt("Wv.weight"), tt("Wgate.weight"), tt("Wout.weight")

    def __call__(self, m, pair):
        ck, lin = self.compute_kernel_config, self._lin
        bias = lin(ttnn.layer_norm(pair, weight=self.cb_ln_w, bias=self.cb_ln_b, epsilon=1e-5, compute_kernel_config=ck), self.cb_w)
        attn = torch.softmax(torch.Tensor(ttnn.to_torch(bias)).float(), dim=-2)
        mn = ttnn.layer_norm(m, weight=self.ns_w, bias=self.ns_b, epsilon=1e-5, compute_kernel_config=ck)
        v = torch.Tensor(ttnn.to_torch(lin(mn, self.Wv))).float().reshape(1, L, M, H, HD)
        g = torch.Tensor(ttnn.to_torch(ttnn.sigmoid(lin(mn, self.Wgate)))).float().reshape(1, L, M, H, HD)
        out = torch.einsum("bijh,bjmhd,bimhd->bimhd", attn, v, g).reshape(1, L, M, H * HD)
        out_tt = ttnn.from_torch(out, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16)
        return ttnn.add(m, lin(out_tt, self.Wout))


ref = ref64()
arms = {"host": Host(), "device": E.MSAPairWeightedAveraging(H, HD, T.WeightScope(sd), ckc)}
up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
for name, mod in arms.items():
    ts = []
    for r in range(REPS + 1):
        mt, zt = up(m), up(z)
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        o = mod(mt, zt)
        ttnn.synchronize_device(dev); ts.append(time.perf_counter() - t0)
        if r == 0:
            out = ttnn.to_torch(o).double().reshape(ref.shape)
            du, ru = out - ref, ref - m.double()       # error relative to the update, not the residual
            log(ev="acc", arm=name, rel_l2_update=float(du.norm() / ru.norm()),
                max_abs=float(du.abs().max()), rel_l2_out=float(du.norm() / ref.norm()))
        for t in (o, mt, zt):
            ttnn.deallocate(t)
    log(ev="time", arm=name, cold_s=ts[0], warm_s=sorted(ts[1:]), aiclk=aiclk())
log(ev="end")

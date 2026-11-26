"""The float64 reference trunk with bf16 rounding at op boundaries (bf16_mixed = fp32 residuals; bf16 = everything
rounded), on batch08's glue inputs: what a correct bf16 trunk does to the insertion term.
autocast = upstream's own bf16 shape (D45): only matmul/einsum INPUTS are bf16 (accumulated wide), every other op
boundary (adds, norms, softmax, sigmoid, residuals) fp32. autocast_bf16out also rounds the matmul output to bf16,
as torch autocast returns it."""
import contextlib, os, sys, torch, numpy as np
from torch.overrides import TorchFunctionMode
B = "/home/ttuser/r4t/tape/b08/"
sys.path.insert(0, B + "src"); sys.path.insert(0, B + "src/tests"); sys.path.insert(0, B + "lpr")
import rfd4_reference as REF
from loss_parity_reference import load_batch
from tt_bio.rfd4 import network as N, engine as E
from tt_bio.train.rfd4_objective import insertion_loss
state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS)); cfg = config["nn"]
glue = N.FlowGlue(state, config)
bt, meta, rec, _ = load_batch("/home/ttuser/r4t/step/batch08.npz")
depth = max(1, int(meta["depth"]))
with torch.no_grad():
    cs = [glue.conditioning(bt, None)] + [glue.conditioning(bt, depth - c - 1) if glue.condition_on_recycle_remaining else None for c in range(1, depth)]
    cs = [c if c is not None else cs[0] for c in cs]
    s0, z0 = glue.initial_states(bt)
mask = bt["padding_mask"].float(); ins = bt["insertion_source_idx"]
w = {k[3:]: v.double() for k, v in state.items() if k.startswith("nn.")}
ref = torch.load(B + "out_P/ref.pt")["sref"].double()
batch = {"insertion_source_idx": ins.numpy(), "padding_mask": mask.bool().numpy()}
def lg(s):
    with torch.no_grad():
        q, k = glue.insertion_q_proj(s.float()), glue.insertion_k_proj(s.float())
        return (torch.bmm(q, k.transpose(1, 2)) * glue._insertion_head_scale).double().numpy()
Lref = insertion_loss(lg(ref), batch)[0]
MM = {torch.matmul, torch.Tensor.__matmul__, torch.einsum, torch.bmm, torch.nn.functional.linear}


class BF16MatmulInputs(TorchFunctionMode):
    def __init__(self, out_bf16):
        super().__init__()
        self.out_bf16 = out_bf16

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in MM:
            args = tuple(REF.bf16(a) if torch.is_tensor(a) else a for a in args)
            out = func(*args, **kwargs)
            return REF.bf16(out) if self.out_bf16 else out
        return func(*args, **kwargs)


mode = sys.argv[1]
if mode.startswith("autocast"):
    rnd, hook = REF.f32, BF16MatmulInputs(mode == "autocast_bf16out")
else:
    rnd, hook = getattr(REF, mode), contextlib.nullcontext()
for b in [int(x) for x in sys.argv[2].split(",")]:
    sl = slice(b, b + 1)
    w_r = {k: rnd(v) for k, v in w.items()}   # weights held in bf16 too
    with torch.no_grad(), hook:
        s, _ = REF.trunk(w_r, rnd(s0[sl].double()), rnd(z0[sl].double()), [rnd(c[sl].double()) for c in cs], mask[sl].double(),
                         cfg["nlayers"], every_n=cfg["update_pair_repr_every_n"], n_heads=cfg["nheads"], rnd=rnd)
    m = mask[b].bool()
    e = float((s[0][m] - ref[b][m]).norm() / ref[b][m].norm())
    full = torch.cat([ref[:b], s, ref[b + 1:]])
    L = insertion_loss(lg(full), batch)[0]
    print(f"{mode} cand{b}: s rel {e:.3e} | ins loss {L[b]:.4e} vs ref {Lref[b]:.4e} rel {abs(L[b]-Lref[b])/Lref[b]:.3e}", flush=True)

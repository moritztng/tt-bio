"""Is the new on-card `single_activations` projection right, forward and VJP?

At n=256 dL/dsingle dominates dL/dpair, so this one op carries most of the gradient and it is
the only thing bcx-e2e added to the trunk. Graded against float64 torch on the same inputs.
"""
import sys, numpy as np, torch
sys.path.insert(0, "perf/bcx_e2e")
sys.path.insert(0, "perf/bcx_afgrad")
sys.path.insert(0, "perf/bcx_stack")
import afgrad as A, stack as S

torch.set_num_threads(8)
lv = S.Levers()
dm, ref = A.load_models(A.DEFAULT_PARAMS)
dev = A.Dev(dm.to_device())
lv.arm("stack")
ag, tt = dev.ag, dev.tt

for n in (128, 256):
    torch.manual_seed(0)
    msa = torch.randn(1, n, 256) * 0.5
    g_single = torch.randn(n, 384) * 0.01
    lin = ref["f64"].single_activations

    # float64 reference: forward and VJP of the same op
    m64 = msa.double().clone().requires_grad_(True)
    s64 = lin(m64[0])
    s64.backward(g_single.double())

    # bf16 torch, the precision the device runs at
    mb = msa.to(torch.bfloat16).float().clone().requires_grad_(True)
    sb = ref["bf16"].single_activations(mb[0].to(torch.bfloat16))
    sb.float().backward(g_single)

    # device
    ml = dev.leaf(msa)
    with tt.tape():
        so = dev.dm.device_single(ml)
    dev.sync()
    s_dev = dev.down(so.value, (n, 384))
    ag.backward([so], [dev.seed(g_single, so)])
    dev.sync()
    gm = dev.grad(ml, msa.shape)
    ag.release_pins()

    print(f"n={n} forward  dev vs f64 {A.cmp(s_dev, s64.detach())}  bf16 vs f64 {A.cmp(sb.detach().float(), s64.detach())}")
    print(f"n={n} VJP      dev vs f64 {A.cmp(gm, m64.grad)}  bf16 vs f64 {A.cmp(mb.grad, m64.grad)}")

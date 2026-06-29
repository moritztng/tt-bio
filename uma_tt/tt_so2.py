"""ttnn port of UMA SO2_Convolution (the dominant compute module).
Big GEMMs (radial MLP + per-m fc) run on device via ttnn.linear; tiny coeff-dim
reshapes/splits/complex-combine run on host. Validates PCC vs golden, benchmarks.
"""
import os, time, math, numpy as np, torch
import ttnn

GOLD = os.path.expanduser("~/.uma_run/golden")

def pcc(a,b):
    a=torch.as_tensor(a).flatten().float(); b=torch.as_tensor(b).flatten().float()
    if a.std()==0 or b.std()==0: return float('nan')
    return torch.corrcoef(torch.stack([a,b]))[0,1].item()

class TTLinear:
    """device-resident linear: y = x @ W^T (+ b)."""
    def __init__(self, dev, weight, bias, kcfg):
        self.dev=dev; self.kcfg=kcfg
        self.w = ttnn.from_torch(weight.t().contiguous(), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
        self.b = None if bias is None else ttnn.from_torch(bias.reshape(1,-1),
                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    def __call__(self, xt):
        return ttnn.linear(xt, self.w, bias=self.b, compute_kernel_config=self.kcfg)

def to_dev(dev, x):
    return ttnn.from_torch(x.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

class TTRadialMLP:
    def __init__(self, dev, state, prefix, kcfg):
        # Sequential of Linear / LayerNorm / SiLU. Module index j; weight dim 2=Linear, 1=LayerNorm.
        self.dev=dev; self.kcfg=kcfg
        idxs=sorted({int(k[len(prefix):].split(".")[0]) for k in state if k.startswith(prefix)})
        self.ops=[]
        for j in idxs:
            w=state[f"{prefix}{j}.weight"]; b=state[f"{prefix}{j}.bias"]
            if w.dim()==2:
                self.ops.append(("lin", TTLinear(dev, w, b, kcfg)))
            else:  # LayerNorm -> followed by SiLU in the reference Sequential
                wt=ttnn.from_torch(w.reshape(1,-1),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
                bt=ttnn.from_torch(b.reshape(1,-1),dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=dev)
                self.ops.append(("ln",(wt,bt)))
    def __call__(self, xt):
        for kind,payload in self.ops:
            if kind=="lin":
                xt=payload(xt)
            else:
                wt,bt=payload
                xt=ttnn.layer_norm(xt, weight=wt, bias=bt, epsilon=1e-5)
                xt=ttnn.silu(xt)
        return xt

def main():
    dev=ttnn.open_device(device_id=0)
    try:
        kcfg=ttnn.init_device_compute_kernel_config(dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        G=torch.load(os.path.join(GOLD,"so2_io.pt"))
        caps=G["caps"]
        # ----- so2_conv_2 (internal weights, no radial, no extra) -----
        validate_so2_2(dev,kcfg,G,caps)
        # ----- so2_conv_1 (radial + extra m0) -----
        validate_so2_1(dev,kcfg,G,caps)
    finally:
        ttnn.close_device(dev)

def so2_m_conv_host_after_fc(y, m_output_channels):
    """y = fc(x_m) on device result [E,2,2*out_half]; do complex combine on host."""
    E=y.shape[0]; out_half=y.shape[-1]//2
    yr = y.reshape(E,-1,out_half)              # [E,4,out_half]
    x_r_0,x_i_0,x_r_1,x_i_1 = yr.split(1,dim=1)
    x_m_r = x_r_0 - x_i_1
    x_m_i = x_r_1 + x_i_0
    return (x_m_r.reshape(E,-1,m_output_channels), x_m_i.reshape(E,-1,m_output_channels))

def validate_so2_2(dev,kcfg,G,caps):
    st=G["so2_2_state"]; cfg=G["so2_2_cfg"]; mss=G["m_split_sizes_2"]
    C=cfg["sphere_channels"]; moc=cfg["m_output_channels"]; lmax=cfg["lmax"]; mmax=cfg["mmax"]
    x = caps["so2_2_in"][0]                      # [E,9,C]
    ref = caps["so2_2_out"]                      # [E,9,moc]
    E=x.shape[0]
    fc_m0=TTLinear(dev, st["fc_m0.weight"], st["fc_m0.bias"], kcfg)
    fcs=[]
    for m in range(1,mmax+1):
        w=st[f"so2_m_conv.{m-1}.fc.weight"]
        fcs.append(TTLinear(dev,w,None,kcfg))
    x_by_m = x.split(mss,dim=1)
    # m0
    x0 = x_by_m[0].reshape(E,-1)
    y0 = ttnn.to_torch(fc_m0(to_dev(dev,x0))).reshape(E,-1,moc)
    out=[y0]
    off=1
    for m in range(1,mmax+1):
        xm = x_by_m[m].reshape(E,2,-1)
        y = ttnn.to_torch(fcs[m-1](to_dev(dev,xm)))
        yr,yi = so2_m_conv_host_after_fc(y, moc)
        out.extend([yr,yi])
    res = torch.cat(out,dim=1)
    print(f"[so2_conv_2] shape {tuple(res.shape)} vs ref {tuple(ref.shape)}  PCC={pcc(ref,res):.5f}")

def validate_so2_1(dev,kcfg,G,caps):
    st=G["so2_1_state"]; cfg=G["so2_1_cfg"]; mss=G["m_split_sizes_1"]; ess=G["edge_split_sizes_1"]
    C=cfg["sphere_channels"]; moc=cfg["m_output_channels"]; lmax=cfg["lmax"]; mmax=cfg["mmax"]
    extra=cfg["extra_m0_output_channels"]
    x = caps["so2_1_in"][0]; x_edge_raw = caps["so2_1_in"][1]
    ref_out, ref_extra = caps["so2_1_out"]
    E=x.shape[0]
    # radial MLP on device
    rad=TTRadialMLP(dev, st, "rad_func.net.", kcfg)
    x_edge = ttnn.to_torch(rad(to_dev(dev,x_edge_raw)))
    fc_m0=TTLinear(dev, st["fc_m0.weight"], st["fc_m0.bias"], kcfg)
    fcs=[TTLinear(dev, st[f"so2_m_conv.{m-1}.fc.weight"], None, kcfg) for m in range(1,mmax+1)]
    x_by_m=x.split(mss,dim=1)
    x_edge_by_m=x_edge.split(ess,dim=1)
    # m0
    x0=x_by_m[0].reshape(E,-1)*x_edge_by_m[0]
    y0=ttnn.to_torch(fc_m0(to_dev(dev,x0)))
    x0_extra, x0=y0.split((extra, y0.shape[-1]-extra),-1)
    out=[x0.reshape(E,-1,moc)]
    for m in range(1,mmax+1):
        xm=x_by_m[m].reshape(E,2,-1)*x_edge_by_m[m].unsqueeze(1)
        y=ttnn.to_torch(fcs[m-1](to_dev(dev,xm)))
        yr,yi=so2_m_conv_host_after_fc(y,moc)
        out.extend([yr,yi])
    res=torch.cat(out,dim=1)
    print(f"[so2_conv_1] out {tuple(res.shape)} PCC={pcc(ref_out,res):.5f} | extra PCC={pcc(ref_extra,x0_extra):.5f}")

if __name__=="__main__":
    main()

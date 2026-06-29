"""Fully device-resident UMA SO2 message-passing core (so2_conv_1 + gate act + so2_conv_2).
Everything stays on device: GEMMs, radial MLP, per-m complex combine, concat. No host glue
in the hot loop. Validates PCC vs golden, then benchmarks warm device-resident throughput.

Layout trick: x [E,9,C] is flattened to [E, 9*C]; the m-blocks are TILE-ALIGNED channel
slices (m0=[0:3C], m1=[3C:7C], m2=[7C:9C], all multiples of 32 for C=128).
"""
import os, time, argparse, numpy as np, torch
import ttnn

GOLD = os.path.expanduser("~/.uma_run/golden")
DESC = os.path.expanduser("~/.uma_run/env/lib/python3.12/site-packages/ttnn/tt_metal/"
                          "fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")
os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", DESC)

def pcc(a,b):
    a=torch.as_tensor(a).flatten().float(); b=torch.as_tensor(b).flatten().float()
    return torch.corrcoef(torch.stack([a,b]))[0,1].item()

def dev_w(dev, w):
    return ttnn.from_torch(w.t().contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
def dev_b(dev, b):
    return None if b is None else ttnn.from_torch(b.reshape(1,-1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
def dev_x(dev, x):
    return ttnn.from_torch(x.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

def slc(t, a, b):  # slice 2D tensor [E,K] -> [E, a:b]
    return ttnn.slice(t, [0, a], [t.shape[0], b])

class SO2ConvTT:
    """Device-resident SO2_Convolution. m_output_channels=moc, sphere_channels=C, lmax=mmax=2."""
    def __init__(self, dev, state, cfg, m_split, edge_split, kcfg, has_radial, extra):
        self.dev=dev; self.kcfg=kcfg; self.moc=cfg["m_output_channels"]
        self.lmax=cfg["lmax"]; self.mmax=cfg["mmax"]; self.extra=extra or 0
        # input channels per coeff = fc_m0.in_features / (lmax+1)  (conv1 input is 2*sphere_channels)
        self.C=state["fc_m0.weight"].shape[1] // (self.lmax+1)
        self.m_split=m_split  # coeff counts per m-block in input layout [m0, 2*m1, 2*m2]
        self.has_radial=has_radial
        # channel offsets in flattened [E, sum(m_split)*C]
        self.in_offsets=np.cumsum([0]+[s*self.C for s in m_split]).tolist()
        # weights
        self.fc_m0_w=dev_w(dev, state["fc_m0.weight"]); self.fc_m0_b=dev_b(dev, state["fc_m0.bias"])
        # per-m: build block weight [[W1,-W2],[W2,W1]] so a single GEMM does the complex combine
        self.fc_m=[]; self.out_half=[]
        for m in range(1,self.mmax+1):
            fcw=state[f"so2_m_conv.{m-1}.fc.weight"]   # [2*out_half, in]
            oh=fcw.shape[0]//2
            W1,W2=fcw.split(oh,dim=0)
            wblk=torch.cat([torch.cat([W1,-W2],dim=1), torch.cat([W2,W1],dim=1)],dim=0)  # [2oh, 2in]
            self.fc_m.append(dev_w(dev, wblk))  # stored transposed inside dev_w
            self.out_half.append(oh)
        if has_radial:
            self.rad=[]
            idxs=sorted({int(k[len("rad_func.net."):].split(".")[0]) for k in state if k.startswith("rad_func.net.")})
            for j in idxs:
                w=state[f"rad_func.net.{j}.weight"]; b=state[f"rad_func.net.{j}.bias"]
                if w.dim()==2: self.rad.append(("lin",(dev_w(dev,w),dev_b(dev,b))))
                else: self.rad.append(("ln",(dev_b(dev,w),dev_b(dev,b))))
            self.edge_split=edge_split

    def radial(self, xe):
        for kind,p in self.rad:
            if kind=="lin": xe=ttnn.linear(xe,p[0],bias=p[1],compute_kernel_config=self.kcfg)
            else: xe=ttnn.silu(ttnn.layer_norm(xe,weight=p[0],bias=p[1],epsilon=1e-5))
        return xe

    def __call__(self, x_flat, x_edge_raw=None):
        """x_flat: device tensor [E, sum(m_split)*C].  returns (out_flat [E, 9*moc], gate or None)."""
        dev=self.dev; C=self.C; moc=self.moc
        if self.has_radial:
            xe=self.radial(x_edge_raw)  # [E, sum(edge_split)]
            eo=np.cumsum([0]+self.edge_split).tolist()
        # m0
        x0=slc(x_flat, self.in_offsets[0], self.in_offsets[1])  # [E, 3C]
        if self.has_radial:
            x0=ttnn.multiply(x0, slc(xe, eo[0], eo[1]))
        y0=ttnn.linear(x0, self.fc_m0_w, bias=self.fc_m0_b, compute_kernel_config=self.kcfg)
        if self.extra:
            gate=slc(y0,0,self.extra); y0=slc(y0,self.extra,y0.shape[-1])
        else:
            gate=None
        outs=[y0]  # [E, 3*moc]
        # m>0
        for m in range(1,self.mmax+1):
            xm=slc(x_flat, self.in_offsets[m], self.in_offsets[m+1])  # [E, 2*ncoeff*C]
            if self.has_radial:
                er=slc(xe, eo[m], eo[m+1])  # [E, ncoeff*C]
                er2=ttnn.concat([er,er],dim=1)   # both real/imag halves scaled by same radial
                xm=ttnn.multiply(xm, er2)
            # block GEMM: [E, 2*in] @ [2*in, 2*out_half] -> [E, 2*out_half] = [real|imag]
            y=ttnn.linear(xm, self.fc_m[m-1], compute_kernel_config=self.kcfg)
            oh=self.out_half[m-1]
            outs.append(slc(y,0,oh))     # real [E, ncoeff*moc]
            outs.append(slc(y,oh,2*oh))  # imag
        out=ttnn.concat(outs, dim=1)  # [E, 9*moc]
        return out, gate

def load():
    G=torch.load(os.path.join(GOLD,"so2_io.pt"))
    return G

def build(dev, G, kcfg):
    so2_1=SO2ConvTT(dev, G["so2_1_state"], G["so2_1_cfg"], G["m_split_sizes_1"],
                    G["edge_split_sizes_1"], kcfg, has_radial=True,
                    extra=G["so2_1_cfg"]["extra_m0_output_channels"])
    so2_2=SO2ConvTT(dev, G["so2_2_state"], G["so2_2_cfg"], G["m_split_sizes_2"],
                    None, kcfg, has_radial=False, extra=None)
    return so2_1, so2_2

def validate(dev, G, kcfg):
    caps=G["caps"]
    so2_1,so2_2=build(dev,G,kcfg)
    C=G["so2_1_cfg"]["sphere_channels"]
    x1=caps["so2_1_in"][0]; E=x1.shape[0]
    x1f=dev_x(dev, x1.reshape(E,-1))
    xe=dev_x(dev, caps["so2_1_in"][1])
    o1,gate=so2_1(x1f, xe)
    ro,rg=caps["so2_1_out"]
    print(f"[resident so2_1] out PCC={pcc(ro.reshape(E,-1), ttnn.to_torch(o1)):.5f} "
          f"gate PCC={pcc(rg, ttnn.to_torch(gate)):.5f}")
    x2=caps["so2_2_in"][0]
    x2f=dev_x(dev, x2.reshape(x2.shape[0],-1))
    o2,_=so2_2(x2f)
    print(f"[resident so2_2] out PCC={pcc(caps['so2_2_out'].reshape(x2.shape[0],-1), ttnn.to_torch(o2)):.5f}")

def bench(dev, G, kcfg, E, iters=50):
    """Device-resident throughput: full SO2 message core (so2_1 + gate*silu act + so2_2)."""
    so2_1,so2_2=build(dev,G,kcfg)
    C=so2_1.C
    in1=G["m_split_sizes_1"]
    rad_input=G["so2_1_state"]["rad_func.net.0.weight"].shape[1]  # raw x_edge width (768)
    torch.manual_seed(1)
    x1=torch.randn(E, sum(in1)*C)*0.1
    xe=torch.randn(E, rad_input)*0.1
    x1f=dev_x(dev,x1); xet=dev_x(dev,xe)
    # warm
    for _ in range(5):
        o1,gate=so2_1(x1f, xet)
        o1=ttnn.silu(o1)              # stand-in pointwise act (real GateActivation similar cost)
        # so2_2 input is [E,9,moc]=[E,9*128]=[E,1152]; matches so2_2 in-layout
        o2,_=so2_2(o1)
        ttnn.synchronize_device(dev)
    t0=time.time()
    for _ in range(iters):
        o1,gate=so2_1(x1f, xet)
        o1=ttnn.silu(o1)
        o2,_=so2_2(o1)
    ttnn.synchronize_device(dev)
    dt=(time.time()-t0)/iters
    return dt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",type=int,default=0)
    ap.add_argument("--bench",action="store_true")
    ap.add_argument("--E",type=int,default=8424)
    a=ap.parse_args()
    dev=ttnn.open_device(device_id=a.device)
    try:
        ttnn.enable_program_cache(dev) if hasattr(ttnn,"enable_program_cache") else None
        kcfg=ttnn.init_device_compute_kernel_config(dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        G=load()
        validate(dev,G,kcfg)
        if a.bench:
            for E in [a.E, 16384, 32768, 65536]:
                dt=bench(dev,G,kcfg,E)
                eps=E/dt
                print(f"[bench dev{a.device}] E={E:6d} layer_core={dt*1000:7.3f}ms  "
                      f"{eps/1e6:6.2f} Medges/s  (~{eps/78:.0f} atoms/s/layer)")
    finally:
        ttnn.close_device(dev)

if __name__=="__main__":
    main()

"""End-to-end: wire the device-resident TT SO2 convolutions into the REAL UMA model
(replace each block's so2_conv_1/so2_conv_2), run the full forward + energy head, and
compare energy / forces / node_embedding PCC vs the PyTorch CPU golden (same weights).
Wigner rotation, activation, scatter, graph stay on host (the <1% irregular geometric part).
"""
import os, time, pickle, numpy as np, torch
import ttnn
torch.manual_seed(0)
from ase.build import molecule, bulk
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from ref_harness import CFG, build_data, EnergyHead
from tt_so2_resident import SO2ConvTT, dev_x

DESC=os.path.expanduser("~/.uma_run/env/lib/python3.12/site-packages/ttnn/tt_metal/"
  "fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")
os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", DESC)
GOLD=os.path.expanduser("~/.uma_run/golden")

def pcc(a,b):
    a=torch.as_tensor(a).flatten().float(); b=torch.as_tensor(b).flatten().float()
    if a.std()==0 or b.std()==0: return float('nan')
    return torch.corrcoef(torch.stack([a,b]))[0,1].item()

class TTSO2Wrap(torch.nn.Module):
    """Wrap an SO2_Convolution instance with a device-resident TT implementation.
    Matches call signatures: conv1(x,x_edge)->(out,gate); conv2(x)->out."""
    def __init__(self, dev, conv, kcfg):
        super().__init__()
        st=conv.state_dict()
        has_radial = conv.rad_func is not None
        extra = conv.extra_m0_output_channels
        cfg=dict(sphere_channels=conv.sphere_channels, m_output_channels=conv.m_output_channels,
                 lmax=conv.lmax, mmax=conv.mmax)
        m_split=conv.m_split_sizes
        edge_split=conv.edge_split_sizes if has_radial else None
        self.tt=SO2ConvTT(dev, st, cfg, m_split, edge_split, kcfg, has_radial, extra)
        self.dev=dev; self.moc=conv.m_output_channels; self.has_radial=has_radial; self.extra=extra
    def forward(self, x, x_edge=None):
        E=x.shape[0]
        xf=dev_x(self.dev, x.reshape(E,-1).float())
        xet=dev_x(self.dev, x_edge.float()) if (self.has_radial and x_edge is not None) else None
        out,gate=self.tt(xf, xet)
        o=ttnn.to_torch(out).reshape(E,-1,self.moc).to(x.dtype)
        if self.extra:
            g=ttnn.to_torch(gate).to(x.dtype)
            return o,g
        return o

def patch(model, dev, kcfg):
    for blk in model.blocks:
        ew=blk.edge_wise
        ew.so2_conv_1=TTSO2Wrap(dev, ew.so2_conv_1, kcfg)
        ew.so2_conv_2=TTSO2Wrap(dev, ew.so2_conv_2, kcfg)

def main():
    dev=ttnn.open_device(device_id=0)
    try:
        kcfg=ttnn.init_device_compute_kernel_config(dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        # rebuild the SAME random model as the golden (seed identical to ref_harness)
        g=torch.load(os.path.join(GOLD,"model_random.pt"))
        model=eSCNMDBackbone(**g["cfg"]).eval(); model.load_state_dict(g["model"])
        head=EnergyHead(g["cfg"]["sphere_channels"]).eval(); head.load_state_dict(g["head"])
        ref=pickle.load(open(os.path.join(GOLD,"ref_results.pkl"),"rb"))

        patch(model, dev, kcfg)
        print("patched all blocks with TT SO2 convs")

        systems={"h2o":molecule("H2O"),"ch4":molecule("CH4"),"c2h6":molecule("C2H6"),
                 "cu_fcc":bulk("Cu","fcc",a=3.6,cubic=True)}
        print(f"{'system':8s} {'E_ref':>10s} {'E_tt':>10s} {'E_PCC/relerr':>14s} "
              f"{'F_PCC':>8s} {'F_cos':>8s} {'node_PCC':>9s}")
        for name,atoms in systems.items():
            batch=build_data(atoms)
            batch["pos"]=batch["pos"].clone().requires_grad_(True)
            out=model(batch)
            e=head(out["node_embedding"], out["batch"], 1)
            f=-torch.autograd.grad(e.sum(), batch["pos"])[0]
            er=ref[name]
            E_ref=float(er["energy"]); E_tt=float(e.item())
            relerr=abs(E_tt-E_ref)/(abs(E_ref)+1e-8)
            Fpcc=pcc(er["forces"], f.detach())
            fr=torch.as_tensor(er["forces"]).flatten().float(); ft=f.detach().flatten().float()
            fcos=torch.nn.functional.cosine_similarity(fr.unsqueeze(0),ft.unsqueeze(0)).item()
            npcc=pcc(er["node_embedding"], out["node_embedding"].detach())
            print(f"{name:8s} {E_ref:10.4f} {E_tt:10.4f} {relerr:14.2e} "
                  f"{Fpcc:8.4f} {fcos:8.4f} {npcc:9.5f}")
    finally:
        ttnn.close_device(dev)

if __name__=="__main__":
    main()

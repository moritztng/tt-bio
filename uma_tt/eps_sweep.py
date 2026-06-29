import os, pickle, numpy as np, torch
import ttnn
torch.manual_seed(0)
from ase.build import molecule
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from ref_harness import build_data, EnergyHead
from tt_e2e import patch, pcc
GOLD=os.path.expanduser("~/.uma_run/golden")
os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", os.path.expanduser("~/.uma_run/env/lib/python3.12/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"))
dev=ttnn.open_device(device_id=0)
try:
    kcfg=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
    g=torch.load(GOLD+"/model_random.pt")
    model=eSCNMDBackbone(**g["cfg"]).eval(); model.load_state_dict(g["model"])
    head=EnergyHead(g["cfg"]["sphere_channels"]).eval(); head.load_state_dict(g["head"])
    ref=pickle.load(open(GOLD+"/ref_results.pkl","rb"))
    patch(model,dev,kcfg)
    atoms=molecule("H2O"); pos0=atoms.get_positions().astype(np.float64)
    def en(pp):
        b=build_data(atoms); b["pos"]=torch.tensor(pp,dtype=torch.float32)
        with torch.no_grad():
            o=model(b); return float(head(o["node_embedding"],o["batch"],1).item())
    for eps in [2e-3,5e-3,1e-2,2e-2]:
        F=np.zeros_like(pos0)
        for i in range(3):
            for d in range(3):
                pp=pos0.copy();pp[i,d]+=eps;pm=pos0.copy();pm[i,d]-=eps
                F[i,d]=-(en(pp)-en(pm))/(2*eps)
        print(f"eps={eps:.0e} PCC={pcc(ref['h2o']['forces'],F):.4f}")
finally:
    ttnn.close_device(dev)

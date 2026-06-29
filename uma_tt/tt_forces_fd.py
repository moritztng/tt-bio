"""Autograd-free forces: central finite-difference of the TT energy surface.
Proves the TT forward yields correct forces without on-device autograd. Compares to the
PyTorch CPU autograd reference forces (same random weights). Validation tool; the production
analytic path is the chain rule (transpose-matmuls on TT) but FD confirms the surface."""
import os, pickle, numpy as np, torch
import ttnn
torch.manual_seed(0)
from ase.build import molecule
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from ref_harness import CFG, build_data, EnergyHead
from tt_e2e import patch, pcc
GOLD=os.path.expanduser("~/.uma_run/golden")
DESC=os.path.expanduser("~/.uma_run/env/lib/python3.12/site-packages/ttnn/tt_metal/"
  "fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")
os.environ.setdefault("TT_MESH_GRAPH_DESC_PATH", DESC)

def main():
    dev=ttnn.open_device(device_id=0)
    try:
        kcfg=ttnn.init_device_compute_kernel_config(dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        g=torch.load(os.path.join(GOLD,"model_random.pt"))
        model=eSCNMDBackbone(**g["cfg"]).eval(); model.load_state_dict(g["model"])
        head=EnergyHead(g["cfg"]["sphere_channels"]).eval(); head.load_state_dict(g["head"])
        ref=pickle.load(open(os.path.join(GOLD,"ref_results.pkl"),"rb"))
        patch(model, dev, kcfg)

        def tt_energy(atoms, pos_override=None):
            batch=build_data(atoms)
            if pos_override is not None:
                batch["pos"]=torch.tensor(pos_override, dtype=torch.float32)
            with torch.no_grad():
                out=model(batch)
                e=head(out["node_embedding"], out["batch"], 1)
            return float(e.item())

        for name in ["h2o","c2h6"]:
            atoms=molecule("H2O" if name=="h2o" else "C2H6")
            pos0=atoms.get_positions().astype(np.float64)
            N=pos0.shape[0]; eps=5e-3  # bf16-energy sweet spot (see eps sweep: too-small amplifies bf16 noise, too-large adds curvature truncation)
            F=np.zeros_like(pos0)
            for i in range(N):
                for d in range(3):
                    pp=pos0.copy(); pp[i,d]+=eps
                    pm=pos0.copy(); pm[i,d]-=eps
                    ep=tt_energy(atoms, pp); em=tt_energy(atoms, pm)
                    F[i,d]=-(ep-em)/(2*eps)
            Fref=ref[name]["forces"]
            p=pcc(Fref, F)
            fr=torch.as_tensor(Fref).flatten().float(); ft=torch.as_tensor(F).flatten().float()
            cos=torch.nn.functional.cosine_similarity(fr.unsqueeze(0),ft.unsqueeze(0)).item()
            mae=np.abs(np.asarray(Fref)-F).mean()
            print(f"[{name}] FD-forces vs ref autograd: PCC={p:.4f} cos={cos:.4f} MAE={mae:.4e} "
                  f"|F|max ref={np.abs(Fref).max():.3f} tt={np.abs(F).max():.3f}")
    finally:
        ttnn.close_device(dev)

if __name__=="__main__":
    main()

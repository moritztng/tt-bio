"""UMA reference harness: instantiate eSCNMDBackbone (random weights) + energy head,
build AtomicData inputs via fairchem, run forward + autograd forces on CPU.
Saves golden inputs/outputs for the ttnn port to validate against.
"""
import os, sys, time, pickle
import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

from ase.build import molecule, bulk
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md import eSCNMDBackbone

GOLD = os.path.expanduser("~/.uma_run/golden")
os.makedirs(GOLD, exist_ok=True)

# ---- config: representative uma-s-ish but small enough for fast CPU iteration ----
CFG = dict(
    max_num_elements=100,
    sphere_channels=128,
    lmax=2, mmax=2,
    num_layers=4,
    hidden_channels=128,
    edge_channels=128,
    num_distance_basis=512,
    cutoff=6.0,
    max_neighbors=300,
    norm_type="rms_norm_sh",
    act_type="gate",
    ff_type="grid",
    direct_forces=False,      # autograd forces (energy-conserving) for reference
    regress_forces=True,
    otf_graph=False,          # use edges built by fairchem ASE/pymatgen neighborlist
    always_use_pbc=False,     # molecule path; periodic system sets pbc in data
    use_dataset_embedding=False,
    use_quaternion_wigner=False,  # Euler+Jd path (pure torch, no custom kernels)
    execution_mode="general",
)

def build_data(atoms, charge=0, spin=0):
    has_cell = atoms.cell.volume > 0.1
    kw = {} if has_cell else {"molecule_cell_size": 60.0}
    d = AtomicData.from_ase(
        atoms, task_name=None, r_edges=True,
        r_data_keys=["spin", "charge"], radius=6.0, max_neigh=300,
        target_dtype=torch.float32, **kw,
    )
    # collate single sample into a batch dict
    from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
    batch = atomicdata_list_to_batch([d])
    batch["charge"] = torch.tensor([charge], dtype=torch.long)
    batch["spin"] = torch.tensor([spin], dtype=torch.long)
    return batch

class EnergyHead(torch.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(C, C), torch.nn.SiLU(), torch.nn.Linear(C, 1)
        )
    def forward(self, node_emb, batch_idx, nsys):
        scalar = node_emb[:, 0, :]            # L=0 component [N, C]
        node_e = self.mlp(scalar).view(-1)    # [N]
        e = torch.zeros(nsys, dtype=node_e.dtype)
        e = e.index_add(0, batch_idx, node_e)
        return e

def main():
    torch.set_default_dtype(torch.float32)
    model = eSCNMDBackbone(**CFG).eval()
    head = EnergyHead(CFG["sphere_channels"]).eval()
    nparams = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"model params: {nparams/1e6:.2f}M")

    systems = {
        "h2o": molecule("H2O"),
        "ch4": molecule("CH4"),
        "c2h6": molecule("C2H6"),
    }
    # periodic: small Cu fcc cell
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    systems["cu_fcc"] = cu

    results = {}
    for name, atoms in systems.items():
        try:
            batch = build_data(atoms)
            batch["pos"] = batch["pos"].clone().requires_grad_(True)
            t0 = time.time()
            out = model(batch)
            nsys = 1
            energy = head(out["node_embedding"], out["batch"], nsys)
            grad = torch.autograd.grad(energy.sum(), batch["pos"], create_graph=False)[0]
            forces = -grad
            dt = time.time() - t0
            print(f"[{name}] natoms={len(atoms)} edges={batch['edge_index'].shape[1]} "
                  f"E={energy.item():.4f} |F|max={forces.abs().max().item():.4f} t={dt*1000:.1f}ms")
            results[name] = dict(
                natoms=len(atoms),
                nedges=int(batch["edge_index"].shape[1]),
                energy=energy.detach().numpy(),
                forces=forces.detach().numpy(),
                node_embedding=out["node_embedding"].detach().numpy(),
                pos=batch["pos"].detach().numpy(),
                atomic_numbers=batch["atomic_numbers"].detach().numpy(),
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] FAILED: {e}")

    # save model state + head + results as golden
    torch.save({"model": model.state_dict(), "head": head.state_dict(), "cfg": CFG},
               os.path.join(GOLD, "model_random.pt"))
    with open(os.path.join(GOLD, "ref_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("saved golden to", GOLD)

if __name__ == "__main__":
    main()

"""Profile the UMA backbone forward to find the op-time breakdown:
host/geometric (graph + Wigner construction) vs matmul-heavy (SO2 conv, atomwise, grid)."""
import os, time, numpy as np, torch
torch.manual_seed(0)
from ase.build import molecule, bulk
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from ref_harness import CFG, build_data, EnergyHead

def make_system(n_rep):
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)  # 4 atoms
    return cu.repeat((n_rep, n_rep, n_rep))

def run(model, head, batch, force=True):
    if force:
        batch["pos"] = batch["pos"].clone().requires_grad_(True)
    out = model(batch)
    e = head(out["node_embedding"], out["batch"], 1)
    if force:
        g = torch.autograd.grad(e.sum(), batch["pos"])[0]
    return e

def main():
    model = eSCNMDBackbone(**CFG).eval()
    head = EnergyHead(CFG["sphere_channels"]).eval()
    for nrep, label in [(2, "Cu-32"), (3, "Cu-108"), (4, "Cu-256")]:
        atoms = make_system(nrep)
        batch0 = build_data(atoms)
        nat = len(atoms); ned = int(batch0["edge_index"].shape[1])
        # warm
        run(model, head, build_data(atoms))
        # timing: full e+f
        N = 5
        t0 = time.time()
        for _ in range(N):
            run(model, head, build_data(atoms))
        dt = (time.time()-t0)/N
        # profiler breakdown
        from torch.profiler import profile, record_function, ProfilerActivity
        b = build_data(atoms)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            run(model, head, b)
        # aggregate by record_function name
        ka = prof.key_averages()
        def tot(name):
            s = 0.0
            for e in ka:
                if e.key == name:
                    s += e.cpu_time_total
            return s/1e3  # ms
        regions = ["generate_graph","obtain rotmat wigner original","obtain wigner",
                   "SO2Conv","edgewise","atomwise","message passing 0","message passing 1",
                   "message passing 2","message passing 3","edge embedding","atom embedding"]
        print(f"\n=== {label}: natoms={nat} edges={ned} | full e+f = {dt*1000:.1f} ms/eval ===")
        for r in regions:
            v = tot(r)
            if v > 0: print(f"   {r:32s} {v:8.2f} ms")
        # top ops by self time
        print("   -- top aten ops by self CPU time --")
        rows = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:12]
        for e in rows:
            print(f"   {e.key:40s} self={e.self_cpu_time_total/1e3:8.2f}ms count={e.count}")

if __name__ == "__main__":
    main()

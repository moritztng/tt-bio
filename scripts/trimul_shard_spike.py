"""Multi-card sharding spike: measure the real ceiling of sharding a single
large trimul across cards, versus running it whole on one card.

The trimul core is a per-channel batched matmul [1, C, N, N] @ [1, C, N, N]
(C=128 channels on the batch axis, N the sequence). It is embarrassingly
parallel across channels: card k computes a disjoint C/K-channel slice with NO
cross-card reduction inside the matmul. The only inter-card traffic is
(a) replicating the layer-normed input to every card and (b) gathering the
per-card output channels back to reassemble the full pair tensor for the output
projection. We measure both against the single-card compute they replace.

Numbers are warm, from real hardware. compute is timed on the mesh (SPMD: one
matmul call, each card runs its own shard in parallel). Fabric cost is the real
ttnn.all_gather collective over the 2- and 4-card mesh.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import ttnn

C_Z = 256  # trimul pair channels (ESMFold2 / Protenix trunk)


def _bench(fn, device, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


def _mk(n: int, c: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, c, n, n, generator=g) * 0.1)


def run(n: int, k: int, mesh_shape: tuple[int, int]) -> dict:
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    ) if hasattr(ttnn, "WormholeComputeKernelConfig") else None

    result = {"N": n, "K": k, "channels": C_Z}

    # ---- Single card: full C=128 batched matmul (the thing being parallelized)
    dev1 = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        a = ttnn.from_torch(_mk(n, C_Z, 11), layout=ttnn.TILE_LAYOUT,
                            device=dev1, dtype=ttnn.bfloat8_b)
        b = ttnn.from_torch(_mk(n, C_Z, 22), layout=ttnn.TILE_LAYOUT,
                            device=dev1, dtype=ttnn.bfloat8_b)

        def full():
            o = ttnn.matmul(a, b, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc)
            ttnn.deallocate(o)

        result["single_card_compute_s"] = _bench(full, dev1)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    finally:
        ttnn.close_mesh_device(dev1)

    # ---- K-card mesh: parallel sharded compute + real fabric all_gather
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
    try:
        cshard = C_Z // k
        # Shard channels across the mesh: each card holds [1, C/K, N, N].
        shard_map = ttnn.ShardTensorToMesh(mesh, dim=1)
        a = ttnn.from_torch(_mk(n, C_Z, 11), layout=ttnn.TILE_LAYOUT,
                            device=mesh, dtype=ttnn.bfloat8_b, mesh_mapper=shard_map)
        b = ttnn.from_torch(_mk(n, C_Z, 22), layout=ttnn.TILE_LAYOUT,
                            device=mesh, dtype=ttnn.bfloat8_b, mesh_mapper=shard_map)

        def par():
            o = ttnn.matmul(a, b, dtype=ttnn.bfloat16,
                            compute_kernel_config=ckc)
            ttnn.deallocate(o)

        result["parallel_compute_s"] = _bench(par, mesh)

        # (1) On-fabric collective — attempted, but qb2's cards have no working
        # card-to-card ethernet, so this is expected to fail (router sync).
        out_shard = ttnn.from_torch(_mk(n, C_Z, 33), layout=ttnn.TILE_LAYOUT,
                                    device=mesh, dtype=ttnn.bfloat16,
                                    mesh_mapper=shard_map)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

            def gather():
                g = ttnn.all_gather(out_shard, dim=1, topology=ttnn.Topology.Ring)
                ttnn.deallocate(g)

            result["all_gather_s"] = _bench(gather, mesh, warmup=1, iters=3)
            result["fabric_all_gather_ok"] = True
        except Exception as e:  # noqa: BLE001
            result["fabric_all_gather_ok"] = False
            result["fabric_all_gather_err"] = repr(e)[:180]
        finally:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

        # (2) Host-mediated comms — the ONLY inter-card path on this box.
        # A sharded trimul must, per call: gather the C/K-per-card output shards
        # to a full host tensor (device->host over PCIe), and replicate the
        # layer-normed input back out to every card (host->device). Measure both
        # on the real PCIe path with the true N=1024 pair tensor size.
        concat = ttnn.ConcatMeshToTensor(mesh, dim=1)
        replicate = ttnn.ReplicateTensorToMesh(mesh)
        host_full = _mk(n, C_Z, 44)  # [1, 128, N, N]

        def gather_to_host():
            _ = ttnn.to_torch(out_shard, mesh_composer=concat)

        def scatter_replicate():
            t = ttnn.from_torch(host_full, layout=ttnn.TILE_LAYOUT, device=mesh,
                                dtype=ttnn.bfloat16, mesh_mapper=replicate)
            ttnn.deallocate(t)

        result["host_gather_s"] = _bench(gather_to_host, mesh, warmup=2, iters=6)
        result["host_replicate_s"] = _bench(scatter_replicate, mesh, warmup=2, iters=6)

        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(out_shard)
    finally:
        ttnn.close_mesh_device(mesh)

    # ---- Derived ceiling
    sc = result["single_card_compute_s"]
    pc = result["parallel_compute_s"]
    result["compute_scaling"] = sc / pc
    # Host-mediated per-trimul-call comms: replicate input + gather output.
    comms = result["host_gather_s"] + result["host_replicate_s"]
    result["host_comms_model_s"] = comms
    result["host_net_shard_s"] = pc + comms
    result["host_net_speedup"] = sc / (pc + comms)
    result["host_comms_over_single"] = comms / sc
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--k", type=int, required=True, choices=[2, 4])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    shape = {2: (1, 2), 4: (2, 2)}[args.k]
    rec = run(args.n, args.k, shape)
    print("RESULT " + json.dumps(rec, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

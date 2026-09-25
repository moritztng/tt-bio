"""Inference trimul op trace: the ttnn ops a TriangleMultiplication dispatches, per config.

Run once from each tree (cwd = the tree). The inference A/B for this row's tenstorrent.py edit
asks whether a fold got slower. Device time is set by the ops dispatched, so an identical op
sequence and identical outputs answer the device half on any host load. The host half is the
guards the edit put on the inference path; their per-call wall is timed separately.
"""
import hashlib, json, os, re, sys, time
import torch, ttnn
from tt_bio import tenstorrent as T


def ck(dev):
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def weights(c=128, seed=0):
    """Random weights in tt-bio's fused layout (what remap_triangle_multiplication emits)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.1  # noqa: E731
    return {"norm_in.weight": 1 + r(c), "norm_in.bias": r(c), "norm_out.weight": 1 + r(c),
            "norm_out.bias": r(c), "g_in.weight": r(2 * c, c), "p_in.weight": r(2 * c, c),
            "g_out.weight": r(c, c), "p_out.weight": r(c, c)}


_VOLATILE = re.compile(r'0x[0-9a-fA-F]+|\\?"address\\?":\s*\d+|tensor_id')


def norm(gr):
    """The graph minus what differs between two runs of the same code: pointers, buffer addresses
    and tensor ids, and the tree root in a custom kernel's source path. Op names, their
    arguments (shapes, dtypes, memory configs, program attributes) and the buffers they
    allocate are kept, so a memory-config or shape change still shows."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(T.__file__))) + "/"
    keep = []
    for n in gr:
        p = {k: v for k, v in (n.get("params") or {}).items()
             if k not in ("address", "tensor_id", "device_tensors", "inputs")}
        keep.append([n.get("node_type"), n.get("connections"), p,
                     [_VOLATILE.sub("", str(a)).replace(root, "ROOT/") for a in n.get("arguments") or []]])
    return json.dumps(keep, sort_keys=True, default=str)


def h(s):
    return hashlib.sha256(s).hexdigest()[:16]


def main(out):
    dev = T.get_device()
    rows, host = [], {}
    for L in (64, 256, 384, 512):
        for ending in (False, True):
            for gated in (False, True):
                for masked in (False, True):
                    tm = T.TriangleMultiplication(ending=ending, state_dict=weights(),
                                                  compute_kernel_config=ck(dev), gated_move=gated)
                    g = torch.Generator().manual_seed(L)
                    x = ttnn.from_torch(torch.randn(1, L, L, 128, generator=g), layout=ttnn.TILE_LAYOUT,
                                        device=dev, dtype=ttnn.bfloat16)
                    m = None
                    if masked:
                        mm = torch.ones(1, L, L)
                        mm[:, L - 7:, :] = 0
                        mm[:, :, L - 7:] = 0
                        m = ttnn.from_torch(mm, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
                    tm(x, m)  # warm: fills the caches and compiles
                    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                    y = tm(x, m)
                    gr = ttnn.graph.end_graph_capture()
                    ops = [str((n.get("params") or {}).get("name")) for n in gr
                           if n.get("node_type") == "function_start"]
                    th = ttnn.to_torch(y).float().contiguous()
                    rows.append({"L": L, "ending": ending, "gated": gated, "masked": masked,
                                 "n_ops": len(ops), "ops_sha": h("\n".join(ops).encode()),
                                 "args_sha": h(norm(gr).encode()),
                                 "out_sha": h(th.numpy().tobytes())})
                    print(rows[-1], flush=True)
                    if L == 384 and not masked:
                        t0 = time.perf_counter()
                        for _ in range(20000):
                            tm._gp_in_chunks(128, 1)
                        host["gp_in_chunks_hit_us_ending%d_gated%d" % (ending, gated)] = (
                            (time.perf_counter() - t0) / 20000 * 1e6)
                    del tm, x, y, m
    json.dump({"tenstorrent_sha": hashlib.sha256(open(T.__file__, "rb").read()).hexdigest()[:12],
               "tt_bio": T.__file__, "rows": rows, "host_us": host}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

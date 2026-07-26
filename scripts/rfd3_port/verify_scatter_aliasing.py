"""Is the RFD3 attention-bias `ttnn.scatter` safe to replay from a captured trace?

The atom blocks build their dense attention bias as
`ttnn.scatter(dense_bias, 3, attn_idx, pair_bias)`, where `dense_bias` is a -1e4
mask template. The eager path allocates a fresh template every call, so it is safe
by construction. The traced decoder does not: `_capture_sparse_trace` allocates ONE
template and replays the captured scatter against it on every step, while `attn_idx`
is restaged each step because the neighbour graph moves with the coordinates.

That is only correct if the scatter leaves its input template pristine. Test 1 asks
that directly. Test 2 reproduces the trace lifetime exactly -- capture once, replay
with index set A then index set B -- and compares against the eager reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ttnn  # noqa: E402

from tt_bio.tenstorrent import get_device  # noqa: E402

L, H, K, B = 256, 4, 32, 1
MASK = -1e4


def _idx_host(indices):
    return ttnn.from_torch(
        indices.view(1, 1, L, K).expand(B, H, L, K).to(torch.int32).contiguous(),
        layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32)


def _template(dev):
    return ttnn.full((B, H, L, L), MASK, dtype=ttnn.bfloat16,
                     layout=ttnn.TILE_LAYOUT, device=dev)


def main() -> None:
    dev = get_device(trace_region_size=1 << 26)
    torch.manual_seed(0)
    a_idx = torch.stack([torch.randperm(L)[:K] for _ in range(L)])
    b_idx = torch.stack([torch.randperm(L)[:K] for _ in range(L)])
    src_host = ttnn.from_torch(torch.full((B, H, L, K), 1.0), layout=ttnn.TILE_LAYOUT,
                               dtype=ttnn.bfloat16)

    # --- test 1: does one scatter mutate its input template? -------------------
    dense = _template(dev)
    src = ttnn.to_device(src_host, dev)
    idx_a = ttnn.to_device(_idx_host(a_idx), dev)
    out = ttnn.scatter(dense, 3, idx_a, src)
    ttnn.synchronize_device(dev)
    after = ttnn.to_torch(dense).float()
    # compare against the bf16 round-trip of -1e4, not -1e4 itself: bf16 has 8
    # mantissa bits, so the stored constant is -9984.0 and a naive `!= -1e4`
    # reports every entry as mutated.
    stored_mask = torch.tensor(MASK, dtype=torch.bfloat16).float()
    n_mutated = int((after != stored_mask).sum())
    print(f"test 1  template entries mutated by one scatter: {n_mutated} "
          f"(of {after.numel()}, stored mask = {stored_mask.item()})")
    del out

    # --- reference: the eager path, a fresh template per index set -------------
    ref_a = ttnn.to_torch(ttnn.scatter(_template(dev), 3, idx_a, src)).float()
    idx_b = ttnn.to_device(_idx_host(b_idx), dev)
    ref_b = ttnn.to_torch(ttnn.scatter(_template(dev), 3, idx_b, src)).float()

    # --- test 2: the trace lifetime -- one persistent template, restaged index --
    persist_bias = _template(dev)
    persist_idx = ttnn.allocate_tensor_on_device(_idx_host(a_idx).spec, dev)
    persist_src = ttnn.allocate_tensor_on_device(src_host.spec, dev)
    ttnn.copy_host_to_device_tensor(_idx_host(a_idx), persist_idx)
    ttnn.copy_host_to_device_tensor(src_host, persist_src)
    for _ in range(2):  # warmup: capture forbids kernel compilation
        _ = ttnn.scatter(persist_bias, 3, persist_idx, persist_src)
    # restore the template the warmups may have dirtied
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(torch.full((B, H, L, L), MASK), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16), persist_bias)
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    traced_out = ttnn.scatter(persist_bias, 3, persist_idx, persist_src)
    ttnn.end_trace_capture(dev, tid, cq_id=0)

    ttnn.copy_host_to_device_tensor(_idx_host(a_idx), persist_idx)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    got_a = ttnn.to_torch(traced_out).float()
    ttnn.copy_host_to_device_tensor(_idx_host(b_idx), persist_idx)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    got_b = ttnn.to_torch(traced_out).float()

    ok_a = torch.equal(got_a, ref_a)
    ok_b = torch.equal(got_b, ref_b)
    print(f"test 2  replay 1 (index set A) matches eager: {ok_a}")
    print(f"        replay 2 (index set B) matches eager: {ok_b}")
    if not ok_b:
        bad = (got_b != ref_b)
        wrong = got_b[bad]
        print(f"        mismatching entries: {int(bad.sum())} of {ref_b.numel()}; "
              f"got e.g. {wrong[:5].tolist()} where eager has "
              f"{ref_b[bad][:5].tolist()}")

    print(f"\nSCATTER_TRACE_SAFE {bool(ok_a and ok_b)}")


if __name__ == "__main__":
    main()

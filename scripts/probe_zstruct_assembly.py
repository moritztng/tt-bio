#!/usr/bin/env python3
"""Do the expander's two z_struct assemblies produce the same tensor, in a real fold?

`11e596e92` changed `StructuralTokenExpander` from a device `ttnn.concat` of its row chunks to a
host `torch.cat` + one `from_torch`, and 9i3p's structure moved. Padding content, a dtype
narrowing, DRAM placement and free-memory-dependent branching are all refuted, so the next thing
to check is the obvious one nobody has checked: whether the two assemblies actually agree on real
expander output rather than on synthetic blocks.

This replaces `tt_bio.opendde._acc_concat` (the name the expander holds, so nothing else is
touched) with a version that builds BOTH assemblies from the same blocks, compares them, writes
the verdict, and returns the host one so the fold proceeds normally.

Run it on a SMALL target with the byte gate forced low, so both full-size copies are cheap:

    TT_BIO_CONCAT_HOST_BYTES=1 TT_VISIBLE_DEVICES=26 \
      python3 scripts/probe_zstruct_assembly.py predict examples/abag_xm/9ncy.yaml \
      --model opendde-abag ...
"""
import os
import sys

import torch
import ttnn

SINK = os.environ.get("TT_BIO_ZSTRUCT_MARK", "/tmp/zstruct_assembly.txt")


def _mark(msg):
    with open(SINK, "a") as fh:
        fh.write(msg + "\n")


def _install():
    import tt_bio.opendde as od
    from tt_bio.tenstorrent import get_device
    orig = od._acc_concat

    def comparing(acc, dim, host):
        if not host or len(acc) < 2 or not torch.is_tensor(acc[0]):
            _mark(f"[Z] skipped: host={host} n={len(acc)} "
                  f"torch={torch.is_tensor(acc[0]) if acc else None}")
            return orig(acc, dim, host)
        dev = get_device()
        # Host assembly, exactly what _acc_concat does.
        host_cat = ttnn.from_torch(torch.cat(acc, dim=dim), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
        # Device assembly from the same bytes: re-uploading a bf16 block is bit-preserving,
        # so these are the blocks the old code concatenated.
        blocks = [ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
                  for b in acc]
        dev_cat = ttnn.concat(blocks, dim=dim)
        a, b = ttnn.to_torch(dev_cat), ttnn.to_torch(host_cat)
        same = torch.equal(a, b)
        msg = f"[Z] blocks={len(acc)} dim={dim} shape={tuple(a.shape)} equal={same}"
        if not same:
            d = (a.float() - b.float()).abs()
            msg += (f" max|diff|={d.max().item():.6g} "
                    f"mismatch={(d > 0).sum().item()}/{d.numel()}")
        _mark(msg)
        ttnn.deallocate(dev_cat)
        for t in blocks:
            ttnn.deallocate(t)
        return host_cat

    od._acc_concat = comparing


from tt_bio.main import cli  # noqa: E402

_install()

if __name__ == "__main__":
    sys.exit(cli(standalone_mode=True))

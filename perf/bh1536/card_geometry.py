#!/usr/bin/env python3
"""What the chip in front of us actually is, and what its allocator says when it refuses.

`size_limits.py` has no Blackhole row and its geometry reasoning compares a p150a against a
Galaxy Wormhole chip. Nothing in the repo has ever read the DRAM shape off a p300c, so a
p150a ceiling carried over to qb2 would be an assertion. This prints the measured shape and
then asks the allocator for a tensor no bank could hold, which does two things at once: it
records this card's real bank size, and it is the negative control for the OOM classifier in
`run_rung.py` (a sweep whose classifier is never shown to fire has not classified anything).

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:... \
        python3 perf/bh1536/card_geometry.py --card 1
"""
import argparse
import json
import socket
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--out_tag", default=ladder_paths.tag_from_env())
    a = ap.parse_args()

    import ttnn  # noqa: E402
    from tt_bio import tenstorrent as tt  # noqa: E402
    dev = tt.get_device()
    # Each accessor on its own: a MeshDevice does not expose the set a Device does
    # (num_dram_channels is absent on the mesh handle, which is what a lone P300 chip opens as),
    # and one missing name must not cost the rest. Channel count comes off dram_grid_size, which
    # both handles have.
    geo = {"host": socket.gethostname(), "card": a.card, "arch": str(ttnn.get_arch_name())}
    for name in ("dram_grid_size", "num_dram_channels", "dram_size_per_channel",
                 "l1_size_per_core", "compute_with_storage_grid_size"):
        try:
            v = getattr(dev, name)()
            geo[name] = [v.x, v.y] if hasattr(v, "x") else v
        except Exception as exc:
            geo[name] = f"unavailable: {type(exc).__name__}"
    grid = geo.get("dram_grid_size")
    chans = (grid[0] * grid[1] if isinstance(grid, list)
             else geo.get("num_dram_channels") if isinstance(geo.get("num_dram_channels"), int)
             else None)
    per_chan = geo.get("dram_size_per_channel")
    geo["dram_channels"] = chans
    geo["dram_total_GiB"] = (round(chans * per_chan / 2**30, 3)
                             if isinstance(chans, int) and isinstance(per_chan, int) else None)

    # One tensor larger than any single bank can hold: the refusal names the bank size, and
    # its mechanism must classify as oversized_tensor and not as fragmentation.
    import torch
    import run_rung
    # Sized off the measured bank, or off the refusal esmfold2 recorded (4278190016 B per bank,
    # 8 banks) when the handle will not say -- the probe still has to be able to overflow.
    per_bank_target = (per_chan if isinstance(per_chan, int) else 4278190016) + 2**20
    elems = (per_bank_target * (chans if isinstance(chans, int) else 8)) // 2   # bfloat16
    side = int(elems ** 0.5) // 32 * 32
    refusal = None
    try:
        t = torch.zeros((side, side), dtype=torch.bfloat16)
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        print(f"NO REFUSAL: {side}x{side} bf16 was allocated; the probe is too small",
              file=sys.stderr)
    except Exception as exc:
        refusal = run_rung._oom_of(str(exc))
    geo["probe"] = {"side": side, "bytes": side * side * 2, "refusal": refusal}
    out = ladder_paths.ROOT / (f"card_geometry.{a.out_tag}.json" if a.out_tag
                               else "card_geometry.json")
    out.write_text(json.dumps(geo, indent=2) + "\n")
    print(json.dumps(geo, indent=2))
    return 0 if refusal and refusal["mechanism"] == "oversized_tensor" else 1


if __name__ == "__main__":
    sys.exit(main())

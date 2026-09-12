#!/usr/bin/env python3
"""Split the committed step site map into its three atom blocks and its token block.

`b2z2-step-program-fusion`'s `site_cost.py` aligned all 1066 device programs of one diffusion step
to the 1590 ttnn calls that issue them. That table is ordered, so the step's structure -- 3 atom
encoder layers, 24 token layers, 3 atom decoder layers -- can be read straight out of it by the
one call no other site makes: the atom branch's `to_memory_config` into DRAM at the top of
`AttentionPairBias`, immediately followed by the key-window build. Offline, no device.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-cost", type=Path,
                    default=Path(__file__).resolve().parents[1] / "b2z2_step_fusion"
                    / "site_cost_wh_c2.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tb = json.loads(a.site_cost.read_text())["table"]

    # An atom layer starts at the DRAM copy whose next costed op is the window build's reshape.
    starts = [t["i"] for t in tb
              if t["ttnn"] == "ttnn.to_memory_config" and t["code"] is None
              and tb[t["i"] + 1]["ttnn"] == "ttnn.reshape"
              and tb[t["i"] + 1]["code"] == "ReshapeView"]
    out = {"atom_layer_starts": starts, "n_atom_layers": len(starts)}
    # each atom layer runs to the next atom-layer start, or (for the last of a block) to the end
    # of its residual add; take the span up to the next start and label the 6 in order
    # An atom layer is a fixed call sequence, so its length is the gap between two adjacent
    # starts inside a block. Reading it off the first pair keeps the third encoder layer from
    # swallowing the 24 token layers that follow it.
    span_len = starts[1] - starts[0]
    spans, labels = [], []
    for n, i0 in enumerate(starts):
        spans.append((i0, min(i0 + span_len, len(tb))))
        labels.append(f"{'encoder' if n < 3 else 'decoder'}[{n % 3}]")

    per_layer = []
    for (i0, i1), lab in zip(spans, labels):
        rows = [t for t in tb[i0:i1] if t["us"]]
        per_layer.append({"label": lab, "i0": i0, "i1": i1,
                          "us": round(sum(t["us"] for t in rows), 3),
                          "programs": len(rows)})
    out["layers"] = per_layer

    # the first three spans are contiguous atom encoder layers; the token block is what sits
    # between the third encoder layer's end and the first decoder layer's start
    enc = sum(l["us"] for l in per_layer[:3])
    out["atom_encoder_us"] = round(enc, 3)
    out["atom_path_us"] = round(sum(l["us"] for l in per_layer), 3)
    i_tok0, i_tok1 = spans[2][1], spans[3][0]
    out["token_block"] = {"i0": i_tok0, "i1": i_tok1,
                          "us": round(sum(t["us"] for t in tb[i_tok0:i_tok1]), 3)}
    out["step_us"] = round(sum(t["us"] for t in tb), 3)

    # per-op-role breakdown of ONE settled atom layer (the second encoder layer)
    i0, i1 = spans[1]
    roles = [{"i": t["i"] - i0, "ttnn": t["ttnn"], "code": t["code"], "us": t["us"]}
             for t in tb[i0:i1] if t["us"]]
    out["atom_layer_detail"] = roles
    by = defaultdict(float)
    for t in roles:
        by[t["code"]] += t["us"]
    out["atom_layer_by_code"] = {k: round(v, 3)
                                 for k, v in sorted(by.items(), key=lambda kv: -kv[1])}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"atom layers {len(starts)}  encoder {out['atom_encoder_us']:.1f} us  "
          f"token {out['token_block']['us']:.1f} us  step {out['step_us']:.1f} us")
    for l in per_layer:
        print(f"  {l['label']:12s} {l['us']:9.3f} us  {l['programs']:4d} programs  [{l['i0']}:{l['i1']}]")
    for k, v in out["atom_layer_by_code"].items():
        print(f"    {k:18s} {v:9.3f} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

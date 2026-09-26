#!/usr/bin/env python3
"""How many taped verbs one replicate spends inside the 24-block DiT, counted from source.

The number that predicts a batching win is the VERB count, not the FLOP count: the OF3
replicate sits 245x off the compute roof and 403x off the bandwidth roof (`of3t-perf10`'s
leg 0), so there is no arithmetic to remove and no bytes to save -- there is fixed per-op
cost paid 48 times over work that would fit in one op. A batched DiT must issue the SAME
verbs with taller operands. If the verb count does not fall, the batching did not reach.

The MEASURED total is banked and needs no card to read: `perf/of3t_p10samples/arm8-chunktable.txt`
prints `6980 nodes` for every 4-replicate chunk, so a replicate is 1745 taped verbs.

What this script adds is the DiT's share of that total, counted off the source. It is an
ESTIMATE, not a measurement: it assumes one tape node per ttnn call, three for
`nlp_create_qkv_heads` (one per output), two for the eltwise-fusion helpers (which decline
to fuse while a tape is open and emit the chain they replace), and it expands `AdaLN` and
`ops.linear` by their own bodies. The exact figure is free on the card pass -- it is the
difference in the chunk line's node count between `--dit-batch 1` and `--dit-batch 4`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Verbs a call expands into. Anything not named here counts as one.
EXPAND = {
    "nlp_create_qkv_heads": 3,      # one tape node per output (taped_ttnn._v_create_qkv_heads)
    "scale_add": 2,                 # declines to fuse under a tape -> multiply + add
    "mask_add": 2,                  # likewise -> multiply + add
    "deallocate": 0,                # the tape declines it
}
# Sub-modules whose bodies are counted once and then substituted at their call sites.
FREE = {"_cached"}                  # the pair bias is cached across the chunk: 0 per replicate


def count(fn: ast.FunctionDef, *, expand=None) -> int:
    n = 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None)
        if name is None or name in FREE:
            continue
        if expand and name in expand:
            n += expand[name]
        elif name in EXPAND:
            n += EXPAND[name]
        elif _is_verb(node):
            n += 1
    return n


def _is_verb(node: ast.Call) -> bool:
    """A device call: ttnn.*, ops.*, self._lin/lin(...), or a helper we listed."""
    f = node.func
    if isinstance(f, ast.Attribute):
        base = f.value
        if isinstance(base, ast.Name) and base.id in ("ttnn", "ops"):
            return True
        if isinstance(base, ast.Attribute) and getattr(base, "attr", "") == "experimental":
            return True
        if f.attr in ("_lin", "linear", "single", "adaln_a", "adaln_t"):
            return True
        return False
    return isinstance(f, ast.Name) and f.id in ("lin", "scale_add", "mask_add", "site_softmax",
                                                "batched_matmul")


def body(path: Path, cls: str, fn: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef) and c.name == cls:
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == fn:
                    return f
    raise KeyError(f"{cls}.{fn} not in {path}")


def main() -> int:
    tt = ROOT / "tt_bio/tenstorrent.py"
    dit = ROOT / "tt_bio/openfold3_diffusion_transformer.py"

    adaln = count(body(tt, "AdaLN", "__call__")) + count(body(tt, "AdaLN", "s_terms"))
    blk = count(body(dit, "_DiTBlock", "__call__"),
                expand={"adaln_a": adaln, "adaln_t": adaln})
    stack = count(body(dit, "OF3DiffusionTransformer", "__call__"))

    per_rep = 1745          # measured: 6980 nodes / 4 replicates, arm8-chunktable.txt
    n_blocks = 24
    dit_total = blk * n_blocks + stack
    print(f"AdaLN (__call__ + s_terms)          {adaln:6d} verbs")
    print(f"one _DiTBlock                        {blk:6d} verbs")
    print(f"24 blocks + the stack wrapper        {dit_total:6d} verbs   (estimate)")
    print(f"whole replicate, MEASURED            {per_rep:6d} verbs   "
          f"(6980 nodes / 4, arm8-chunktable.txt)")
    print(f"DiT share of the replicate           {dit_total / per_rep:6.1%}   (estimate)")
    for S in (2, 4, 8, 12):
        batched = dit_total + S * (per_rep - dit_total)
        print(f"  S={S:<3d} verbs for {S} replicates {batched:6d} against {S * per_rep:6d} "
              f"unbatched  -> {S * per_rep / batched:.2f}x fewer dispatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())

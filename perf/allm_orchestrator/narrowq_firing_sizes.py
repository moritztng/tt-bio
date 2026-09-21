#!/usr/bin/env python3
"""Which sequence lengths does `TT_BIO_TRIATT_NARROW_Q_FALLBACK` actually change anything at?

`land-standing` holds this lever as the biggest item in its backlog -- **16.7 s, 6.8 % at 896 aa**
on a path its own comment says is "shared by rf3, boltz-2, protenix-v2, openfold3 and opendde" --
and says it "needs a Blackhole A/B at more than one size" without saying which sizes. That is a
question about a pure policy function, so it is answerable here without a device.

It matters to THIS campaign for a reason bigger than the lever: every published cell the ALLM
charter measures is **512 aa**, and this lever is **inert at 512**. A campaign that measures one
size is structurally blind to it.

The mechanism, from `_tri_att_q_chunks`: only q_chunks that DIVIDE the padded sequence are offered.
When the production chunk does not divide it, the caller lands on a padding q_chunk, which sets
`use_padded_mask` -- and that is a precondition of the fused path, so it **declines the fused
kernel outright** and falls back to the stock op. The lever offers the dividing chunks below the
production pick instead.

No device, no `tt_bio` import: the two policy functions are exec'd from the tree's own source.
"""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "tt_bio" / "tenstorrent.py"


def load():
    """`_padded_sdpa_len` and `_capped_sdpa_chunk_size`, exec'd from source, plus their constants."""
    src = SRC.read_text()
    tree = ast.parse(src)
    ns = {}
    for name in ("SDPA_CHUNK_TILE", "SDPA_CHUNK_MAX"):
        node = next((n for n in tree.body if isinstance(n, ast.Assign)
                     and any(getattr(t, "id", "") == name for t in n.targets)), None)
        if node is None:
            raise SystemExit(f"{name} is gone from tenstorrent.py -- re-read the policy")
        ns[name] = ast.literal_eval(node.value)
    for name in ("_padded_sdpa_len", "_capped_sdpa_chunk_size"):
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if fn is None:
            raise SystemExit(f"{name} is gone from tenstorrent.py -- re-read the policy")
        body = ast.get_source_segment(src, fn)
        body = "\n".join(l for l in body.splitlines()
                         if not l.strip().startswith("@"))       # drop lru_cache
        exec(compile(body, f"<{name}>", "exec"), ns)
    return ns, src


def main() -> int:
    ns, src = load()
    padded_len, capped = ns["_padded_sdpa_len"], ns["_capped_sdpa_chunk_size"]
    TILE = ns["SDPA_CHUNK_TILE"]

    # `_sdpa_chunks_shipped`'s q branch is the one thing here transcribed rather than exec'd
    # (its k branch reaches helpers this script does not need), so ASSERT the transcription against
    # the source: if the band moves, this refuses instead of quietly answering for the old policy.
    import re
    shipped = next((ast.get_source_segment(src, n) for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.FunctionDef) and n.name == "_sdpa_chunks_shipped"), None)
    if shipped is None:
        raise SystemExit("`_sdpa_chunks_shipped` is gone -- re-read the policy")
    body = "\n".join(l for l in shipped.splitlines() if not l.strip().startswith("#"))
    if not re.search(r"if 256 < q_len <= 384 and 256 < k_len <= 384:", body):
        raise SystemExit("the 256<len<=384 band in `_sdpa_chunks_shipped` changed shape -- this "
                         "script's q transcription no longer matches it, re-read it")
    if not re.search(r"return \(64, 64\)", body):
        raise SystemExit("the band no longer returns q=64 -- re-read `_sdpa_chunks_shipped`")
    if not re.search(r"return \(_capped_sdpa_chunk_size\(q_len\)", body):
        raise SystemExit("the non-band q pick is no longer `_capped_sdpa_chunk_size` -- re-read it")

    def prod_q(n):
        """`_sdpa_chunks_shipped(n, n)[0]` for triangle attention, where q_len == k_len."""
        if 256 < n <= 384:
            return 64
        return capped(n)

    def fires(n):
        """The lever changes the offered tuple iff the production chunk does not divide padded."""
        return padded_len(n) % prod_q(n) != 0

    # the 15 tile-aligned lengths the lever's own comment counts
    band = [n for n in range(640, 1089, TILE)]
    hit = [n for n in band if fires(n)]
    print(f"THE COMMENT'S OWN CLAIM, recomputed from the policy functions")
    print(f"  tile-aligned lengths 640..1088: {len(band)}")
    print(f"  of those, the lever changes the offered chunks at: {len(hit)}")
    print(f"  -> the comment says 13 of 15. This says {len(hit)} of {len(band)}: "
          f"{'AGREES' if (len(hit), len(band)) == (13, 15) else '*** DISAGREES ***'}")
    print(f"  the {len(band) - len(hit)} that do NOT fire: "
          f"{[n for n in band if not fires(n)]}  (every multiple of 256)")

    print("\nTHE CAMPAIGN'S OWN CELLS")
    for n in (256, 512):
        print(f"  {n:4} aa: padded {padded_len(n):4}, production q_chunk {prod_q(n):3} -> "
              f"{'FIRES' if fires(n) else 'INERT (chunk divides padded)'}")
    print("  Every published cell in this campaign is 512 aa, so the whole seven-model census,")
    print("  every transfer ratio and every fold A/B is blind to this lever by construction.")

    print("\nFULL FIRING MAP, tile-aligned, 256..1408")
    row = []
    for n in range(256, 1409, TILE):
        row.append(f"{n}{'*' if fires(n) else ' '}")
        if len(row) == 12:
            print("  " + "  ".join(row)); row = []
    if row:
        print("  " + "  ".join(row))
    allhit = [n for n in range(256, 1409, TILE) if fires(n)]
    print(f"  (* = lever changes the offered chunks)   {len(allhit)} of "
          f"{len(range(256, 1409, TILE))} lengths fire")

    print("\nWHERE A BLACKHOLE A/B SHOULD GO, which is what `land-standing` asked")
    print("  The lever's 16.7 s was measured at 896 aa on a Galaxy Wormhole 8x9. Its own comment")
    print("  says the ceiling is L1 and 'the per-core budget moves with the core count (110 on")
    print("  this part, 130 on a 13x10 one)', so the Wormhole number does not transfer and the")
    print("  A/B needs sizes that differ in the quantity that MOVES, not just more sizes:")
    print("    896  -- reproduces the measured cell on the new part")
    print("    1088 -- fires, and refuses MORE configs, so it stresses the L1 term hardest")
    print("    768  -- a NEGATIVE CONTROL: it does not fire, so the arm must read 1.00x there.")
    print("            An A/B that moves at 768 is measuring something other than this lever.")
    print("  That third leg is the one a size sweep usually omits, and it is the only one that")
    print("  can falsify the instrument rather than the lever.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

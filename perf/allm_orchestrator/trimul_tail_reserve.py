#!/usr/bin/env python3
"""`trimul_tail.OUT_L1` is ON and fires on NOTHING. Is its reserve the reason, and by how much?

Counted by `allm-gates`: **0 of 1048 admitted on protenix-v2 and 0 of 1084 on esmfold2**, the only
two models that reach the site. The clause is not a model predicate --
`_l1_fits(bytes, 1.0, _PAIR_L1_CONSUMER_RESERVE)` at `tenstorrent.py:7016`, where the reserve is a
global 640 KB/core fitted at S=704 for the **pair-weighted-averaging softmax's** circular buffers
and read here by a **different consumer with different buffers**
(`one-size-tuning-is-a-standing-defect-class`).

`allm-gates` recorded that this "owes the per-core arithmetic on a p300c before it is called
incidental". That arithmetic is here. Everything is read from the tree; the ONE number that needs a
device is `ttnn.get_max_worker_l1_unreserved_size()`, so it is a parameter and the script prints the
answer across a band rather than guessing it.

No `tt_bio` import, no device.
"""
import ast
import re
import sys
from pathlib import Path

TT = Path(__file__).resolve().parents[2] / "tt_bio"
TILE_B = 32 * 32 * 2          # one bf16 tile


def literal(src: str, name: str):
    for n in ast.parse(src).body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise SystemExit(f"no literal assignment for {name}")


def padded_bytes_from_tree(src: str):
    """The tree's OWN `_padded_bytes`, exec'd from source. Never a transcription of it."""
    fn = next((n for n in ast.parse(src).body
               if isinstance(n, ast.FunctionDef) and n.name == "_padded_bytes"), None)
    if fn is None:
        raise SystemExit("no `_padded_bytes` in tenstorrent.py -- it was renamed or moved, and "
                         "this script will not fall back to a transcription of it")
    ns = {}
    exec(compile(ast.get_source_segment(src, fn), "<padded_bytes>", "exec"), ns)
    return ns["_padded_bytes"]


def main() -> int:
    tt_src = (TT / "tenstorrent.py").read_text()
    tail_src = (TT / "trimul_tail.py").read_text()

    # The reserve, and the consumer it was actually fitted to, read from the source.
    def must(pat, what):
        """Every number here is read from the tree. A read that fails REFUSES rather than
        defaulting, because a silently-missing constant would turn this into arithmetic on
        numbers I typed."""
        m = re.search(pat, tt_src)
        if m is None:
            raise SystemExit(f"could not read {what} from tenstorrent.py -- it moved or changed "
                             f"shape; re-read the source rather than trusting this script")
        return m

    reserve = int(must(r"_PAIR_L1_CONSUMER_RESERVE = (\d+) \* 1024", "the reserve").group(1)) * 1024
    fitted = must(r"softmax's static circular\n# buffers needed (\d+) B/core", "the PWA CB figure")
    need = must(r"Need there is (\d+) B/core", "the stated need")
    have = must(r"past the (\d+) B the part has", "the p150a per-core size")
    print(f"THE RESERVE AS SHIPPED: {reserve:,} B/core ({reserve // 1024} KB)")
    print(f"  fitted at S=704 to the PWA softmax: its CBs {int(fitted.group(1)):,} B/core, "
          f"stated need {int(need.group(1)):,} B/core")
    print(f"  the part it was measured on had {int(have.group(1)):,} B/core unreserved (p150a)")
    p150a_per_core = int(have.group(1))

    # What THIS consumer's own circular buffers cost. `_build` adds exactly three, and their sizes
    # are `out_block * 2`, `out_block * 2` and `2` tiles with `out_block = block[0] * block[2]`.
    keys = literal(tail_src, "F1_BLOCK_KEYS")
    mm = literal(tt_src, "_MM_BLOCK")
    print(f"\nTHE CONSUMER THAT ACTUALLY RUNS -- `trimul_tail.fused_tail`, allow-list {sorted(keys)}:")
    for key in sorted(keys):
        blk = mm.get(tuple(key))
        if blk is None:
            print(f"  {tuple(key)}: not in _MM_BLOCK today, skipped")
            continue
        out_block = blk[0] * blk[2]
        cbs = (out_block * 2 + out_block * 2 + 2) * TILE_B
        print(f"  {tuple(key)} -> block {blk}, out_block {out_block}: "
              f"c_4+c_5+c_6 = {cbs:,} B/core ({cbs / 1024:.0f} KB)")
    print("  ^ these are the buffers THIS site adds. The base matmul CBs from `MG.build` sit under")
    print("    them and are not enumerated here, so treat this as a FLOOR on the consumer's need,")
    print("    not as the reserve to ship.")

    # The admission test, solved. `headroom * nbytes <= max(per_core - reserve, 0) * cores`.
    print("\nTHE ADMISSION TEST, SOLVED for the call that is being declined")
    print("  protenix-v2, 512 aa, c_z=256: x[1,512,512] x out_p_weight[...,256], elem 2")
    padded_bytes = padded_bytes_from_tree(tt_src)
    nbytes = padded_bytes([1, 512, 512, 256], 2)
    print(f"  nbytes = {nbytes:,} B ({nbytes / 2**20:.0f} MiB)")
    for cores in (110, 130):
        avail_now = max(p150a_per_core - reserve, 0) * cores
        verdict = "ADMITS" if nbytes <= avail_now else "DECLINES"
        # largest reserve that still admits
        rmax = p150a_per_core - -(-nbytes // cores)      # ceil-div, so the bound is not optimistic
        print(f"\n  at {cores} cores, per_core {p150a_per_core:,} B (p150a, from the comment):")
        margin = (f"short by {nbytes - avail_now:,} B" if nbytes > avail_now
                  else f"spare {avail_now - nbytes:,} B")
        print(f"    available now = ({p150a_per_core:,} - {reserve:,}) x {cores} "
              f"= {avail_now:,} B  ->  {verdict}  ({margin})")
        print(f"    reserve would have to be <= {rmax:,} B/core ({rmax / 1024:.0f} KB) to admit")
        print(f"    with reserve 0: {p150a_per_core * cores:,} B -> "
              f"{'ADMITS' if nbytes <= p150a_per_core * cores else 'DECLINES'}")

    print("""
WHAT THIS SETTLES AND WHAT IT DOES NOT
  Settles: the decline is the RESERVE, not the shape and not a model predicate. The shipped
  640 KB/core is 1.16x a need measured for a DIFFERENT consumer at a different shape, while the
  consumer that actually runs here adds tens of KB, not hundreds. That is
  `one-size-tuning-is-a-standing-defect-class` in its usual form.

  Does NOT settle: the p300c number. Everything above uses the p150a per-core figure that the
  tree's own comment records. `allm-gates` owes ONE device query on its granted card --
  `ttnn.get_max_worker_l1_unreserved_size()` -- and the threshold line above then reads directly.
  Nor does it settle what the reserve SHOULD be: the floor printed here omits `MG.build`'s base
  matmul CBs, and the repair the campaign wants is a reserve DERIVED from the consumer that will
  run, not a second hand-fitted constant to replace the first.

  And the lever still owes a fold A/B with an A/A floor on every model it then fires for. An
  admission counter going 0 -> 1048 is firing, not a win.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

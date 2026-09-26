"""What the device actually sees, for every binder length BindCraft 2 can draw.

The drawn length is not the token count. Two paddings sit between them, and both are run here
rather than reimplemented: pad_design_chains rounds the BINDER chain up to the bucket, then
_predict_complex rounds binder-plus-target up to the bucket again. CPU only, no device.
"""
import argparse, json, sys
import jax
import jax.numpy as jnp
from bindcraft.af2 import pad_design_chains, padded_prediction_length
from bindcraft.protein import Protein
from bindcraft.settings import read_settings, build_design_settings
from bindcraft.campaign import prepare_targets


def tokens(target_length: int, binder_length: int, bucket: int = 32) -> int:
    """BindCraft 2s own two paddings, on real Protein objects."""
    target = Protein.empty(target_length, jax.random.PRNGKey(0)).replace(flags=jnp.zeros((target_length,), dtype=jnp.uint8))
    binder = Protein.empty(binder_length, jax.random.PRNGKey(1))
    padded = pad_design_chains({"s": {"binder": binder, "target": target}}, bucket, 0)["s"]
    return padded_prediction_length(sum(len(p) for p in padded.values()), bucket)


def longest_binder(target_length: int, ceiling: int = 352, bucket: int = 32) -> int:
    """Largest draw whose token count still fits, by search over the real function."""
    fits = [b for b in range(1, ceiling + 1) if tokens(target_length, b, bucket) <= ceiling]
    return max(fits) if fits else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=None, help="read the target length from a settings file")
    parser.add_argument("--target-length", type=int, default=None)
    parser.add_argument("--ceiling", type=int, default=352)
    parser.add_argument("--draws", default="60:180")
    args = parser.parse_args()

    target_length = args.target_length
    if args.settings:
        design_settings = build_design_settings(read_settings(args.settings))
        lengths = {name: len(p) for name, p in prepare_targets(design_settings).items()}
        target_length = max(lengths.values())
        print(f"settings {args.settings}: targets {lengths}")

    low, high = (int(x) for x in args.draws.split(":"))
    buckets: dict[int, list[int]] = {}
    for binder in range(low, high + 1):
        buckets.setdefault(tokens(target_length, binder), []).append(binder)

    print(f"target {target_length} aa, draws {low}..{high}, ceiling {args.ceiling}")
    for token_count in sorted(buckets):
        draws = buckets[token_count]
        mark = "" if token_count <= args.ceiling else "   OOM"
        print(f"  {token_count:4d} tokens  <- draws {draws[0]}..{draws[-1]} ({len(draws)}){mark}")
    print(f"longest draw that fits: {longest_binder(target_length, args.ceiling)}")

    print("\nrule of thumb, checked against the same function:")
    print("  L      longest binder   tokens at that binder")
    for L in (100, 129, 150, 160, 172, 192, 200, 224, 256, 288, 292, 320, 321, 352):
        b = longest_binder(L, args.ceiling)
        print(f"  {L:4d}   {b:4d}             {tokens(L, b) if b else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

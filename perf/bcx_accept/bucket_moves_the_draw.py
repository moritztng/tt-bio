"""Does passing --bucket 32 change the trajectory draw, given BindCraft 2's own value is 32?

Both arm stamps record an effective length_bucket_size of 32, so the orchestrator concluded the
flag was a no-op. That is a claim about the FOLD. The draw is a different question: the trajectory
name is design_hash(settings, ...) and length_bucket_size is not in EXCLUDED_SETTING_NAMES, so it
changes the name if and only if it changes the settings DICT.

Card-free. No weights.
"""
import os
import sys

sys.path.insert(0, "/home/ttuser/bcx_accept_art/armtree/perf/bcx_predictor")
import bc2_state as B  # noqa: E402

from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings  # noqa: E402
from bindcraft.design_identity import design_setting_values, canonical_text  # noqa: E402

SETTINGS = os.path.join(B.BC2, "examples", "pdl1.json")


def build(bucket):
    overrides = ["campaign_seed=0", "max_trajectories=5",
                 "project_folder=/tmp/bucket_probe"]
    if bucket:
        overrides.append("length_bucket_size=%d" % bucket)
    return cleaned_campaign_settings(
        read_settings(SETTINGS, parse_setting_overrides(overrides)))


def main():
    with_bucket = build(32)
    without = build(0)

    print("length_bucket_size in settings:")
    print("  --bucket 32 :", repr(with_bucket.get("length_bucket_size", "<ABSENT>")))
    print("  --bucket 0  :", repr(without.get("length_bucket_size", "<ABSENT>")))
    print()

    hb = design_setting_values(with_bucket)
    hn = design_setting_values(without)
    print("hash basis key count: with=%d without=%d" % (len(hb), len(hn)))
    only_b = sorted(set(hb) - set(hn))
    only_n = sorted(set(hn) - set(hb))
    differing = sorted(k for k in set(hb) & set(hn) if hb[k] != hn[k])
    print("keys only when --bucket passed :", only_b)
    print("keys only when --bucket omitted:", only_n)
    print("keys present in both but differing:", differing)
    for k in differing:
        print("   %s: %r -> %r" % (k, hn[k], hb[k]))
    print()
    same = canonical_text(hb) == canonical_text(hn)
    print("canonical hash text identical:", same)
    print()
    if same:
        print("VERDICT: --bucket 32 does NOT move the draw. The two seed-0 arms drew different")
        print("binders for some other reason, and the matched-pair recipe must be derived from")
        print("whatever that is, not from the bucket flag.")
    else:
        print("VERDICT: --bucket 32 DOES move the draw. A matched pair must pass the SAME")
        print("--bucket argument on both halves, even though the effective bucket is 32 either")
        print("way, because the settings dict feeds the trajectory hash.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

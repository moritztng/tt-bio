#!/usr/bin/env python3
"""A43: an ANGLE may only be read from a triple that satisfies rel^2 = 1 + r^2 - 2 r cos.

`of3t-angle` found the campaign carries two definitions of the same three aggregates.
`of3t-trunkg043/score.py`'s `mass_weighted_rel_l2` is a mass-weighted QUADRATIC mean of the
per-tensor rel, while `mass_weighted_norm_ratio` and `mass_weighted_cos` are ARITHMETIC means
of the per-tensor ratio and cosine -- so the identity does NOT hold on them. On R149's own
shipped arm the residual is 25 %. An angle read off that `cos` is the average of some cosines,
not an angle.

The rule is narrow on purpose, so it can be enforced without arguing about every `cos` in the
tree: a `cos` may be published as a summary statistic, but the moment an artifact publishes an
ANGLE -- or a best-rescaling number, which is sin(angle) and is the same claim -- it is making a
direction claim and owes the residual that says its triple can carry one.

Scoped PER ARTIFACT, not per object, and that is a deliberate weakening I want on the record.
The first version demanded a residual beside every angle and fired on exactly one thing:
`BF16_SPLIT.json`'s `angle_required_degrees` -- the cos 0.9003 that WOULD suffice. That is a
derived target, not a measured split, and every measured triple in that same artifact carries
its residual. Demanding a residual beside a derived number would have taught rows to copy one
next to it, which is worse than not asking.

So the rule enforced here is the one A43 actually needs: an artifact that publishes ANY
direction claim must demonstrate the identity SOMEWHERE in itself, at float64 noise. That
catches the real failure -- a split read off `score.py`'s mass-weighted triple, which can never
show a residual at noise because the identity misses by 25 % -- while leaving derived numbers
alone. An artifact publishing angles with no residual anywhere is the thing this refuses.
"""
import json, math, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
NOISE = 1e-12          # float64 aggregate noise; every real split measures 1e-16..2e-15
ANGLE_KEYS = ("angle_degrees", "angle_deg", "angle")
RESCALE_KEYS = ("residual_after_the_best_possible_rescaling",)


def _residual(obj):
    """The identity residual an object publishes, from itself or a nested IDENTITY block."""
    for k, v in obj.items():
        if k.lower() in ("rel_difference", "identity_residual"):
            if isinstance(v, (int, float)):
                return float(v)
        if k.upper() == "IDENTITY" and isinstance(v, dict):
            r = _residual(v)
            if r is not None:
                return r
    return None


def _claims_direction(obj):
    for k, v in obj.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        kl = k.lower()
        if kl in ANGLE_KEYS or kl in RESCALE_KEYS:
            return k
        # "the angle   35.13 degrees" style keys
        if "angle" in kl and "degree" in kl:
            return k
    return None


def walk(obj, path, out):
    if isinstance(obj, dict):
        claim = _claims_direction(obj)
        if claim is not None:
            res = _residual(obj)
            out.append((path, claim, res))
        for k, v in obj.items():
            walk(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]", out)


def _all_residuals(obj, out):
    if isinstance(obj, dict):
        r = _residual(obj)
        if r is not None:
            out.append(r)
        for v in obj.values():
            _all_residuals(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _all_residuals(v, out)


def scan(files):
    good, bad = [], []
    for f in files:
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        found = []
        walk(j, "", found)
        if not found:
            continue
        res = []
        _all_residuals(j, res)
        clean = [r for r in res if math.isfinite(r) and abs(r) <= NOISE]
        name = Path(f).name
        if not res:
            bad.append((f"{name} ({len(found)} direction claim(s), e.g. {found[0][1]})",
                        "the artifact publishes a direction claim and never demonstrates "
                        "rel^2 = 1 + r^2 - 2 r cos"))
        elif not clean:
            bad.append((f"{name} ({len(found)} direction claim(s))",
                        f"every identity residual in it exceeds {NOISE:g}; worst "
                        f"{max(abs(r) for r in res):g} -- the triple cannot carry an angle"))
        else:
            good.append((f"{name} ({len(found)} direction claim(s))", max(clean)))
    return good, bad


def break_control():
    """A synthetic artifact publishing an angle with no residual MUST be caught, and the same
    one with a residual must not be. A check that cannot fail has not been tested."""
    import tempfile, os
    d = tempfile.mkdtemp()
    naked = Path(d) / "naked.json"
    naked.write_text(json.dumps({"SPLIT": {"cos": 0.77, "norm_ratio": 1.03,
                                           "angle_degrees": 39.4}}))
    dressed = Path(d) / "dressed.json"
    dressed.write_text(json.dumps({"SPLIT": {"cos": 0.6976277066165968,
                                             "norm_ratio": 0.9978645292850583,
                                             "angle_degrees": 45.763016914559415,
                                             "IDENTITY": {"rel_difference": 7.1e-16}}}))
    _, bad_naked = scan([naked])
    good_dressed, bad_dressed = scan([dressed])
    os.remove(naked); os.remove(dressed); os.rmdir(d)
    return len(bad_naked) == 1 and len(bad_dressed) == 0 and len(good_dressed) == 1


def main():
    if not break_control():
        print("A43 GUARD IS DEAD: the break control did not fire on an angle with no residual")
        return 1
    files = sorted(str(p) for p in ROOT.glob("perf/of3t_*/**/*.json"))
    good, bad = scan(files)
    if bad:
        print(f"A43 VIOLATION: {len(bad)} direction claim(s) published without a usable identity")
        for where, why in bad[:20]:
            print(f"  {where} -- {why}")
        return 1
    worst = max((r for _, r in good), default=0.0)
    print(f"ok    A43: {len(good)} artifact(s) publishing direction claims each demonstrate "
          f"rel^2 = 1 + r^2 - 2 r cos, worst residual {worst:g} across {len(files)} of3t "
          f"artifact(s) (break control fires on an angle with no identity anywhere)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

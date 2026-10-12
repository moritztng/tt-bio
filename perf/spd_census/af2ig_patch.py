#!/usr/bin/env python3
"""Open spd-census's device-profiler harness to AF2-IG.

spd-census (on the Galaxy at ~/spd/spd-census) carries the only Tracy-enabled tt-metal build and the
only harness that profiles a real tt-bio fold from inside the device. Its census.py stops at the four
diffusion models. AF2-IG takes neither --diffusion_samples nor the MSA flags, so three edits reach it.

    cp ~/spd/spd-census/tree23/perf/spd_census/{census.py,analyze.py} <tree>/perf/spd_census/
    python3 <tree>/perf/spd_census/af2ig_patch.py <tree>/perf/spd_census/census.py
    MODEL=af2ig TREE=<tree> PY=<serving python> bash ~/spd/spd-census/bin/run.sh \
        $CHIP <outdir> "exact=exact" af2ig_pdl1_l80_s0

The fold's input yaml and its pdb must be in ~/spd-data/inputs (the harness reads from there, not from
the campaign's own data dir).

First result, 2026-10-12, .114 chip 28 at AICLK 1000 on 33/33 in-fold samples, tree 8e053986e, 195 tokens,
30,176 programs with none unattributed: device kernel 4.250 s, device-idle gaps 1.371 s, span 5.620 s;
triangle multiplication 1.57 s, triangle attention 1.33 s, linear/projection 0.61 s, outer product mean
0.33 s; 62 % of kernel time bandwidth-bound, 38 % launch overhead; every bandwidth-bound op at bfp8 LoFi
peak would be 1.41x, bfp4 1.81x.
"""
import sys

SUBS = [
    ('choices=("protenix-v2", "opendde", "openfold3", "boltz2")',
     'choices=("protenix-v2", "opendde", "openfold3", "boltz2", "af2ig")'),
    ('"--model", a.model, "--diffusion_samples", str(a.samples),',
     '"--model", a.model, *([] if a.model == "af2ig" else ["--diffusion_samples", str(a.samples)]),'),
    ('"--msa_dir", str(a.data / "msa"), "--msa_cache_only", ',
     '*([] if a.model == "af2ig" else ["--msa_dir", str(a.data / "msa"), "--msa_cache_only"]), '),
]


def main(path: str) -> None:
    s = open(path).read()
    for old, new in SUBS:
        if new in s:
            continue
        assert s.count(old) == 1, f"census.py has moved on, cannot place: {old}"
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)


if __name__ == "__main__":
    main(sys.argv[1])

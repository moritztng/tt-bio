#!/usr/bin/env python3
"""Which models the Blackhole Transition row-height raise actually speaks for.

The brief's last question: which OTHER models reach this 4-D Transition path with a channel above
`_BH_TRANSITION_L1_ROWS_MAX_C`. Answering it off the guard rather than off a device run is exact,
because the guard is one comparison -- `_TRANSITION_L1_ROWS and _c <= _BH_TRANSITION_L1_ROWS_MAX_C`
-- and a channel above it takes the base height no matter what the flag says.

Channels are the shipped pair widths: openfold3_diffusion_transformer.C_Z, rfd3 model.C_Z,
esmfold2.C_Z, and the two documented in tenstorrent.py itself (protenix-v2 at 256, OpenDDE at 384).
boltz2's two are what the fold prints under TT_BIO_TRANSITION_TRACE at 1024 aa.

    census_models.py
"""
import sys
from pathlib import Path

# sys.path[0] for a script is the script's own directory, so without this the import resolves to
# the INSTALLED tt_bio in the venv and the census scores a package nobody edited. Same pin, and the
# same assert, as hfold.py.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import tt_bio  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

assert Path(tt_bio.__file__).resolve().is_relative_to(REPO), \
    f"imported tt_bio from {tt_bio.__file__}, not this worktree"

MODELS = (("boltz2 pair", 128), ("boltz2 MSA", 64), ("openfold3", 128), ("rfd3", 128),
          ("esmfold2", 256), ("protenix-v2", 256), ("opendde", 384))


def main() -> int:
    maxc = T._BH_TRANSITION_L1_ROWS_MAX_C
    print(f"_BH_TRANSITION_L1_ROWS_MAX_C = {maxc}   TT_BIO_TRANSITION_L1_ROWS default = "
          f"{T._TRANSITION_L1_ROWS}")
    print()
    print("%-12s %-5s %-8s %s" % ("model", "c", "raise", "height source"))
    for name, c in MODELS:
        on = T._TRANSITION_L1_ROWS and c <= maxc
        print("%-12s %-5d %-8s %s" % (
            name, c, "YES" if on else "no",
            "per-shape budget" if on else "base height, lever inert at this channel"))
    print()
    print("So the raise speaks for boltz2, openfold3 and rfd3 only. At 256 and above the guard")
    print("declines it, which means TT_BIO_TRANSITION_L1_ROWS=0 is a NO-OP for esmfold2,")
    print("protenix-v2 and OpenDDE: the branch is not taken at their channel either way. The")
    print("OpenDDE override in the comment above _BH_TRANSITION_L1_ROWS_MAX_C predates that guard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

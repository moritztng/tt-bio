#!/usr/bin/env python3
"""of3t-paedraws: confpfe's devstep.py at a chosen denoise noise seed (PROTOCOL A44).

    devstep.py --seed K <every perf/of3t_confpfe/devstep.py argument>

fullstep64's devstep.py rebuilds trainfwd_run's argv without a seed, so the step always ran at
trainfwd_run's default 20260922. This appends `--seed K` to that argv; trainfwd_run hands it to
`OpenFold3Forward(seed=K)`, which passes it to `denoise_draw`. The rollout still replays `--draws`,
so the denoise noise is the only thing that moves. Nothing under tt_bio changes.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    argv = sys.argv[1:]
    i = argv.index("--seed")
    seed = str(int(argv[i + 1]))
    del argv[i:i + 2]
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trainfwd"))
    import trainfwd_run

    orig_main = trainfwd_run.main

    def main_with_seed():
        sys.argv += ["--seed", seed]
        return orig_main()

    trainfwd_run.main = main_with_seed
    spec = importlib.util.spec_from_file_location(
        "confpfe_devstep", HERE.parent / "of3t_confpfe" / "devstep.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)
    sys.argv = [spec.origin] + argv
    return cf.main()


if __name__ == "__main__":
    raise SystemExit(main())

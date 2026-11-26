"""Generate ``tt_bio/diffdock/score_norm_tables.json``.

    python3 scripts/diffdock_port/gen_score_norms.py [out.json]

Runs in any environment with numpy — it needs neither e3nn nor the upstream clone, because
both curves are computed from their definitions rather than copied from upstream's hidden
``.npy`` caches. Takes a couple of minutes; that is why the result is shipped.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from tt_bio.diffdock import schedules as S  # noqa: E402


def main(out_path):
    so3 = S.compute_so3_score_norms()
    tor = S.compute_torus_score_norms()
    payload = {
        "_so3_note": ("RMS IGSO(3) score on 2000 log-spaced eps in [5e-4, 4], L=2000, in "
                      "upstream's summation order (a matmul reassociation breaks it: the "
                      "series cancels to 1e-14 against a 1e3 peak). Matches upstream's "
                      ".so3_exp_score_norms4.npy to 2.7e-11 max relative over the eps the "
                      "model reaches, [0.03, 1.55]; the 19 non-finite entries are upstream's "
                      "too and sit outside it."),
        "_torus_note": ("Mean square wrapped-normal score on 5001 log-spaced sigma in "
                        "[3e-3, 2] x pi, by quadrature. Upstream Monte-Carlos this with "
                        "10000 UNSEEDED draws at import, so it has no fixed value: three "
                        "upstream processes spread by 0.8 % median, 3.6 % max."),
        "so3_score_norm": [None if not np.isfinite(v) else float(v) for v in so3],
        "torus_score_norm": [float(v) for v in tor],
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh)
        fh.write("\n")
    print(f"wrote so3 {so3.shape} torus {tor.shape} to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tt_bio/diffdock/score_norm_tables.json")

"""Fold one protein with Mol-1 through the Protenix engine and save the coordinates.

Usage: TT_VISIBLE_DEVICES=<card> python perf/mol1_smoke/fold.py CKPT SEQ OUT_PT [N_CYCLES N_STEP]
The child process tests/test_mol1_bucket_hw.py censuses; small cycle/step counts are for tests only.
"""
import sys

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import torch  # noqa: E402

from tt_bio.protenix import Protenix  # noqa: E402
from tt_bio.protenix_data import build_protein_features  # noqa: E402


def main(ckpt, seq, out, n_cycles="20", n_step="200"):
    model = Protenix.load_from_checkpoint(ckpt)
    with torch.no_grad():
        coords = model.fold(build_protein_features(seq), n_step=int(n_step), n_sample=1, seed=101,
                            n_cycles=int(n_cycles))
    torch.save(torch.as_tensor(coords), out)


if __name__ == "__main__":
    main(*sys.argv[1:])

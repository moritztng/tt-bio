"""CLI wrapper for tt_bio.rfd3_design.extract_rfd3_weights (dev/manual use).

    python extract_weights.py /root/work/ckpt/rfd3_latest.ckpt /root/work/capture

Saves, under out_dir:
  token_initializer.real_weights.pt   (model.token_initializer.*  , prefix stripped)
  diffusion_module.real_weights.pt    (model.diffusion_module.*   , prefix stripped)
  *.real_weights.meta.json            (key->shape/dtype + key list)

tt-bio's own CLI (`tt-bio design`) does not need this script — it auto-downloads
the checkpoint and runs the same extraction via tt_bio.main.ensure_rfd3_weights.
"""
import sys

from tt_bio.rfd3_design import extract_rfd3_weights

if __name__ == "__main__":
    extract_rfd3_weights(sys.argv[1], sys.argv[2])

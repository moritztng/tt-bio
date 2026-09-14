# Reproducing the tagged diffusion trace and its ranking

One device, one process per arm. The card grant for this row was pc physical 3, which does not
exist: pc enumerates exactly one board (UMD chip 0, Blackhole p150a). Card 0 is the card that
`pc-card0-512aa-fold-nondeterminism` says must not host bit-exact gating. It is fine here, because
a matmul that returns a slightly wrong value still allocates the same buffers, and this row counts
allocations. No value, digest or wall clock from these runs is used for anything.

    PY=/home/moritz/tt-bio/env/bin/python3

    # today's main
    OMP_NUM_THREADS=8 PYTHONPATH=$PWD \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:roof-diffusion-tagged-capture \
    $PY perf/b2z2_byte_floor/trace_block.py --layer difftx --size 512 --steps 4 --call 3 \
        --out-dir perf/roof_diffusion_tagged/out/default

    # the graph capture's own configuration
    BOLTZ2_TOKEN_DIT_SDPA=0 TT_BIO_ATOM_AXIS_BUCKET=0 OMP_NUM_THREADS=8 PYTHONPATH=$PWD \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:roof-diffusion-tagged-capture \
    $PY perf/b2z2_byte_floor/trace_block.py --layer difftx --size 512 --steps 4 --call 3 \
        --out-dir perf/roof_diffusion_tagged/out/control

    # rank each trace
    for arm in control default; do
      for f in perf/roof_diffusion_tagged/out/$arm/difftx_*.json.gz; do
        $PY perf/roof_orchestrator/fusion_pairs.py "$f" \
            "perf/roof_diffusion_tagged/out/$arm/pairs_$(basename "$f" .json.gz).json"
      done
    done

`fusion_pairs.py` is not committed on this branch. It belongs to `wk/roof-orchestrator` and is taken
from there unchanged, blob `fff4b3e1a0f8255b44b39d91baa40656b69e3f2f`:

    git show 6fa7fd28e:perf/roof_orchestrator/fusion_pairs.py > perf/roof_orchestrator/fusion_pairs.py

`--call 3` records the fourth call of each `DiffusionTransformerLayer` shape, which is the call
`perf/b2x_difflayer/capture_difftx.py` hands to `ttnn.graph`, so the two instruments look at the
same layer of the same step. `--steps 4` keeps the rollout short; the fold columns multiply by the
200-step call counts in `perf/bioir_roofline/flops_bytes_512.json`.

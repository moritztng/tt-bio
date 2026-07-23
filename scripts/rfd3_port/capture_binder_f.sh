#!/bin/bash
# Capture a protein-binder (F1) / motif-scaffolding (F6) reference `f` golden on
# the vast.ai reference instance, for the p11 featurizer parity gate.
#
# Produces, in $CAP_DIR:
#   token_initializer.in_f_<key>.pt   (the reference `f` the ported featurizer
#                                     must reproduce)
#   token_initializer.out_*.pt        (TI outputs, for the device-vs-ref gate)
#   diffusion_module.real_weights.pt, token_initializer.real_weights.pt
#   meta.json                         (input pdb + contig, so parity_compare_f.py
#                                     can replay the SAME input through the ported
#                                     featurizer and compare)
#
# Run on the vast instance (after setup_and_capture_all.sh has installed
# rc-foundry[rfd3] + the ckpt + cloned foundry):
#   bash capture_binder_f.sh
set -e
export PYTHONPATH="/root/work/foundry/src:/root/work/foundry/models/rfd3/src"
export RFD3_CAPTURE_DIR="${RFD3_CAPTURE_DIR:-/root/work/capture_binder}"
mkdir -p "$RFD3_CAPTURE_DIR"

# A protein-only motif-scaffolding case: 1bna is DNA -> use a protein input_pdb.
# 5o4d.pdb is protein (RFD3 demo set). Contig: motif A1-10, scaffold 20, motif A31-40.
PDB="/root/work/foundry/models/rfd3/docs/input_pdbs/5o4d.pdb"
CONTIG="A1-10,20,A31-40"

# Build a one-design demo.json pointing at the protein pdb + the binder contig.
cat > /root/work/run/binder_demo.json <<EOF
{"binder_F1": {"input": "${PDB}", "contig": "${CONTIG}"}}
EOF

# Record the exact input so parity_compare_f.py can replay it through the ported featurizer.
cat > "${RFD3_CAPTURE_DIR}/meta.json" <<EOF
{"input": "${PDB}", "contig": "${CONTIG}", "case": "protein_binder_F1"}
EOF

cd /root/work/foundry/models/rfd3/docs/examples
python /root/work/run/capture_all.py \
    ckpt_path=/root/work/ckpt/rfd3_latest.ckpt \
    out_dir=/root/work/cap_out \
    inputs=/root/work/run/binder_demo.json \
    inference_sampler.num_timesteps=1 \
    diffusion_batch_size=1 n_batches=1 \
    skip_existing=False prevalidate_inputs=True seed=42 \
    json_keys_subset='[binder_F1]' \
    read_sequence_from_sequence_head=False 2>&1 | tail -40

echo "=== capture done ==="
ls -la "$RFD3_CAPTURE_DIR" | head
echo "Replay locally with:"
echo "  .venv/bin/python scripts/rfd3_port/parity_compare_f.py <local_copy_of_$RFD3_CAPTURE_DIR>"

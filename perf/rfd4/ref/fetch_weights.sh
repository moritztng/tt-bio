#!/bin/bash
# Fetch only the RFD4 checkpoints. The HF repo also ships ~90 GB of AF2/OF3/CUDA runtime
# images (apptainer .sif/.sqsh); this row runs on CPU and never uses them.
set -euo pipefail
P=/home/moritz/rfd4-ref-env/venv/bin/python
$P -I -c "
from huggingface_hub import snapshot_download
p = snapshot_download('nvidia/RFD4-Proteina',
    ignore_patterns=['alphafold2/*', 'openfold3/*', 'runtime/*', 'assets/*'])
print('CACHED nvidia/RFD4-Proteina', p)
"
$P -I -c "
from huggingface_hub import snapshot_download
print('CACHED nvidia/RFD4-Proteina-prep', snapshot_download('nvidia/RFD4-Proteina-prep'))
"
echo WEIGHTS_OK

#!/usr/bin/env bash
# Build a rented vast.ai box for the BoltzGen GPU benchmark. Idempotent; safe to re-run.
#
#   bash /work/scripts/gpu_vs_tt/bgg_setup.sh
#
# Image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime, purely for a working python + CUDA
# userspace. The venv is ISOLATED on purpose: boltzgen resolves cuequivariance-ops-torch-cu12
# 0.11.1, which requires torch>=2.11, so pip installs torch 2.13 into the venv. With
# --system-site-packages the image's torchvision (built against 2.7.1) stays visible and ABI-breaks
# the moment anything imports it; three of five models hit exactly that in the 512 aa pass. An
# isolated venv is also what a researcher gets from a plain pip install, so it costs nothing in
# fairness. torch 2.13 ships cu130 wheels, which need driver >= 580 -- pick the offer accordingly.
set -uo pipefail
WORK=/work
LOG=$WORK/results/setup.log
mkdir -p "$WORK/results" "$WORK/out"
cd "$WORK"

say() { echo "== $* : $(date -u +%FT%TZ) ==" | tee -a "$LOG"; }

say "box"
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,power.max_limit \
  --format=csv | tee -a "$LOG"
df -h / | tee -a "$LOG"
nproc | tee -a "$LOG"

say "apt"
# rdkit's Chem.Draw is imported transitively through pdbeccdutils, which boltzgen depends on, and
# dies on libXrender.so.1. A runtime image ships none of these four and the whole import fails.
# build-essential is not optional either: cuEquivariance's fused layer_norm_transpose is a triton
# kernel, and triton JIT-compiles its CUDA driver shim with the system C compiler on first call. On
# a -runtime image that raises "Failed to find C compiler" from inside the first trimul, which reads
# as a model failure and is really a missing gcc.
apt-get update -qq && apt-get install -y -qq libxrender1 libxext6 libsm6 libxi6 curl \
  build-essential 2>&1 | tail -2 | tee -a "$LOG"

say "venv + boltzgen"
if [ ! -x "$WORK/venv-bgg/bin/pip" ]; then
  python3 -m venv "$WORK/venv-bgg"
fi
"$WORK/venv-bgg/bin/pip" install --no-cache-dir --upgrade pip -q
"$WORK/venv-bgg/bin/pip" install --no-cache-dir -q boltzgen==0.3.2 2>&1 | tail -8 | tee -a "$LOG"

say "resolved versions"
"$WORK/venv-bgg/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
import json, torch
from importlib.metadata import version
info = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
    "boltzgen": version("boltzgen"),
    "cuequivariance_torch": version("cuequivariance-torch"),
    "cuequivariance_ops_torch_cu12": version("cuequivariance-ops-torch-cu12"),
    "cuequivariance_ops_cu12": version("cuequivariance-ops-cu12"),
}
print("RESOLVED " + json.dumps(info))
# 0.8.0 and 0.10.0 of cuequivariance-ops-torch-cu12 hang on sm_100 (protenix-v2 and OpenDDE, 30 min
# with no fold, 203 W on a ~1000 W part) while 0.11.1 runs clean. boltzgen pins only >=0.5.0, so the
# resolved version is what decides whether the B200 cell exists. Fail loudly here rather than
# discovering it as a stall.
cueq = tuple(int(x) for x in version("cuequivariance-ops-torch-cu12").split(".")[:3])
cap = torch.cuda.get_device_capability(0)
assert torch.cuda.is_available(), "no CUDA device"
if cap[0] >= 10 and cueq < (0, 11, 1):
    raise SystemExit("REFUSE: cuequivariance-ops-torch-cu12 %s on sm_%d%d is the known hang; "
                     "0.11.1 is the first clean version" % (version("cuequivariance-ops-torch-cu12"),
                                                            cap[0], cap[1]))
PY

say "fixture integrity"
# The pinned fixture, byte-identical on both boxes. The YAML reads the CIF by a relative path
# (../../../examples/ground_truth_structures/9ma0.cif), so the two files must keep this layout.
cat > /tmp/bgg_sha256 <<'EOF'
d08d13832e14b847444e4486d7d6c5d7d149fc71a7f671e82c187f0757e22eee  perf/dsfix/fixtures/bg_R3.yaml
343b9ea97c656fe5ecb15d3fbf773cb71ba373d004762f1c52e9c46785c94b81  perf/dsfix/fixtures/bg_R4.yaml
96bc91c44c36c73819807e2a512e38a93044cfb9fa6102e88c1d68e61e306b39  examples/ground_truth_structures/9ma0.cif
9554895cb4c5e232b10ddad0da1db27f7acb22a4a7b30f1e0320f01817e9c459  examples/ground_truth_structures/9q6y.cif
EOF
sha256sum -c /tmp/bgg_sha256 2>&1 | tee -a "$LOG"
if ! sha256sum -c --status /tmp/bgg_sha256; then
  echo "REFUSE: fixture sha256 mismatch, the transfer is not byte-identical" | tee -a "$LOG"
  exit 1
fi

say "weights"
# Pre-fetch so a 4.3 GB download cannot land inside a timed arm. `--steps design` needs the two
# design checkpoints and the moldir; the folding/affinity/ifold checkpoints are not used here.
"$WORK/venv-bgg/bin/boltzgen" download design-diverse design-adherence moldir 2>&1 \
  | tail -6 | tee -a "$LOG"
"$WORK/venv-bgg/bin/python" - <<'PY' 2>&1 | tee -a "$LOG"
import hashlib, json
from huggingface_hub import hf_hub_download
# Canonical sha256 = the LFS oid of boltzgen/boltzgen-1 at revision
# c1be29e1f82ffcc72264f64b993c43fb4e0d17f0, read from the Hub API on 2026-08-13.
WANT = {
    "boltzgen1_diverse.ckpt":
        ("360af8bd6e59527ff6ec25dd81253967f3bd3567d200053b10680634751f8e3c", 1930847192),
    "boltzgen1_adherence.ckpt":
        ("ac7078b3dc13064c68e0c3fd542e5bc538c33558bf6607f65e499eb336ca5e5d", 1930858014),
}
out = {}
for name, (want_sha, want_size) in WANT.items():
    p = hf_hub_download("boltzgen/boltzgen-1", name, library_name="boltzgen")
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    size = __import__("os").path.getsize(p)
    ok = (h.hexdigest() == want_sha and size == want_size)
    out[name] = {"path": p, "sha256": h.hexdigest(), "size": size, "ok": ok}
    if not ok:
        raise SystemExit("REFUSE: %s is %s / %d, expected %s / %d"
                         % (name, h.hexdigest(), size, want_sha, want_size))
print("CHECKPOINTS " + json.dumps(out))
PY

say "setup done"

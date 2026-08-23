#!/usr/bin/env bash
# Ship the harness and the pinned fixtures to a rented box and prove they arrived byte-identical.
#
#   bash perfpage_xfer.sh root@1.2.3.4 31570
#
# Runs from the repo root on pc. Two things it does that a plain scp does not:
#  * it carries examples/ground_truth_structures/{9ma0,9q6y}.cif. bgg_setup.sh hashes those and
#    refuses to continue without them; the A100 pass lost a stage to leaving them out.
#  * it re-verifies all seven digests ON THE BOX. The BoltzGen integrity gate caught a truncated
#    transfer that way once, and a byte-identical input is the whole reason the new cells are
#    comparable to the published ones.
set -uo pipefail
HOST=${1:?usage: perfpage_xfer.sh user@host [port]}
PORT=${2:-22}
SSH="ssh -o StrictHostKeyChecking=accept-new -p $PORT"

PATHS=(scripts/gpu_vs_tt perf/size512/fixtures perf/dsfix
       examples/ground_truth_structures/9ma0.cif examples/ground_truth_structures/9q6y.cif)
read -r -d '' EXPECT <<'DIG'
24d8b2d8c06e4409995abae024766e316da3175dde7596073b68c7963d2df398  perf/size512/fixtures/cdk2x2_512.yaml
ef2301402e7716e9df368987210456194b108c8ae7e144b7bf09a3dd6f40bf5e  perf/size512/fixtures/cdk2x2_512.a3m
141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d  scripts/gpu_vs_tt/fixtures/prot512.seq
d08d13832e14b847444e4486d7d6c5d7d149fc71a7f671e82c187f0757e22eee  perf/dsfix/fixtures/bg_R3.yaml
647e066a983e66184e16bf7696b6e731f354e4161c6e764b292e1f9a15c00eef  perf/dsfix/fixtures/rfd3_R4_gpu.json
96bc91c44c36c73819807e2a512e38a93044cfb9fa6102e88c1d68e61e306b39  examples/ground_truth_structures/9ma0.cif
9554895cb4c5e232b10ddad0da1db27f7acb22a4a7b30f1e0320f01817e9c459  examples/ground_truth_structures/9q6y.cif
DIG

TAR=/tmp/perfpage_xfer.tar.gz
tar czf "$TAR" "${PATHS[@]}" || exit 1
echo "tarball $(sha256sum "$TAR")"
printf '%s\n' "$EXPECT" > /tmp/perfpage_xfer.sha256
sha256sum -c /tmp/perfpage_xfer.sha256 || { echo "local digests do not match the pins"; exit 1; }

$SSH "$HOST" 'mkdir -p /root/repo /work /root/results' || exit 1
scp -P "$PORT" -q "$TAR" /tmp/perfpage_xfer.sha256 "$HOST:/tmp/" || exit 1
$SSH "$HOST" 'set -e; cd /root/repo && tar xzf /tmp/perfpage_xfer.tar.gz &&
  cp /tmp/perfpage_xfer.sha256 /root/repo/ && cd /root/repo && sha256sum -c perfpage_xfer.sha256 &&
  mkdir -p /work && for d in scripts perf examples; do cp -a /root/repo/$d /work/ 2>/dev/null || true; done &&
  cd /work && sha256sum -c /root/repo/perfpage_xfer.sha256 && echo XFER_VERIFIED_BOTH_ROOTS'

#!/bin/bash
# Rebuild the cleaned target structures from RCSB. Run it in this directory.
#
# The cleaned .pdb files ARE checked in -- they are the campaign's inputs and a rung has to
# be reproducible without the network. The raw downloads are not: they are 7.3 MB, 3V83
# alone is 5.3 MB of six copies in the asymmetric unit, and this script regenerates them.
#
# Chain selections and why: 2P4E keeps the prodomain and the catalytic chain, which is what
# PCSK9 is; 3V83 keeps one copy of the six; 1TNF keeps all three chains of the homotrimer;
# 1AO6 and 3KS3 keep chain A.
set -euo pipefail
for id in 3KS3 1TNF 1AO6 2P4E 3V83; do
    [ -f "$id.pdb" ] || curl -sS -o "$id.pdb" "https://files.rcsb.org/download/$id.pdb"
done
python3 clean_targets.py 3KS3:A:hCA2 1TNF:A,B,C:hTNFa 1AO6:A:hHSA 2P4E:P,A:hPCSK9 3V83:A:hTF

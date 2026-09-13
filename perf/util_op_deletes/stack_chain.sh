#!/usr/bin/env bash
# The whole stack measurement in ONE benchlock. A lock released between rounds lets a co-tenant
# land between them, and then the round-to-round comparison stops meaning anything.
set -u
ART=/home/ttuser/scratch/uod
$ART/ab_stack.sh 0 base patched
$ART/ab_stack.sh 1 patched base
$ART/ab_stack.sh 2 base patched
$ART/ab_stack.sh 3 patched base
echo "=== $(date -u +%H:%M:%S) STACK CHAIN DONE"

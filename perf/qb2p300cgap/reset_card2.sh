#!/usr/bin/env bash
# reset_card2.sh — clear the p300c board carrying card 2, refusing if card 3 is in use.
#
# Two facts this encodes, both measured on qb2 2026-09-02.
#
# 1. A p300c chip on this box wedges on its 4th device open. Three back-to-back
#    esmc-300m-single draws land 9 s apart and the 4th hangs inside ttnn.open_device
#    (futex_do_wait, 100 % CPU, nothing logged past the benchlock line, still hung after
#    6 min). It reproduced twice with the same shape, before and after a reset, and a 20 s
#    settle between draws did not prevent it. The lease is not the cause: it is a bounded
#    120 s flock that raises DeviceInUseError, and the hang is unbounded and past it.
#    Hugepages are not the cause either: FileHugePages sits at 12445696 kB with zero TT
#    processes alive, so that is the driver's static per-card reservation, not a leak.
#    So: reset the board between draws rather than chase an open count.
#
# 2. tt-smi -r 2 is a BOARD reset, not a chip reset. tt-smi -ls reports chips 0 and 1 on
#    board 0000046131934103 and chips 2 and 3 on board 000004613193410d, so resetting
#    card 2 also resets card 3. A worker holding card 3 would be knocked over by it, which
#    is why this refuses to reset while either chip on the board is open.
set -eu
for d in 2 3; do
  holder=$(sudo lsof "/dev/tenstorrent/$d" 2>/dev/null | tail -n +2 | awk '{print $2}' | sort -u | tr '\n' ' ')
  [ -n "$holder" ] && { echo "reset refused: /dev/tenstorrent/$d held by pid(s) $holder" >&2; exit 1; }
done
~/.local/bin/tt-smi -r 2 >/dev/null 2>&1
echo "card2 board reset $(date -u +%FT%TZ)"

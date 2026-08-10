# y-envelope-audit — predictions, recorded before the device was opened

qb1 card 3 (TT_VISIBLE_DEVICES=3 -> /dev/tenstorrent/0), ttnn 0.67.4, base 7224ff348.
Arm A = main as it stands (_PAIR_PROJ_BW = 16, _PAIR_PROJ_L1_BW = 16, NOT bit-exact).
Arm B = control, both constants set to 1 (bit-exact contraction order), then restored.
scripts/full_parity_gate.py in DEFAULT mode (no --legacy-rdx), legs protenix-{prot,ubq,hsa}-msa.

P1  all three legs return PASS on arm A.                  wrong if any leg returns FAIL.
P2  arm A worst-metric ratio (numerator / bound) for
    protenix-prot-msa is in 0.20 - 0.90 (dimensionless).  wrong if outside that band.
P3  arm A worst-metric ratio exceeds arm B's by
    <= 0.25 absolute on every leg (dimensionless).        wrong if any delta > 0.25.
P4  arm B worst-metric ratio >= 0.05 on every leg: the
    device is bf16 against a CPU fp32 reference whatever
    the contraction order (dimensionless).                wrong if any arm B leg < 0.05.
P5  arm A total device-fold wall across the three legs is
    2 - 6 % below arm B (% of arm B wall).                wrong if arm A is slower by > 1 %.

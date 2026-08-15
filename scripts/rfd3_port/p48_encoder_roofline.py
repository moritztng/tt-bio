"""Place the token encoder's matmul half on this chip's MEASURED roofs.

Roofs are qb2 board 007 chip 0, measured 2026-08-09 (perf/ledger_298/roofs_card.py, recorded in
state/perfwar-rfd3-esmfold2-sites.md): bf16 HiFi4 square matmul 102.02 TFLOP/s, DRAM read
390.0 GB/s, DRAM write 269.6 GB/s, machine balance 261.6 FLOP/byte.

Times are measured, from perf/p46/token_encoder_itemise_calibrated.json. FLOPs and bytes are a
shape model from tt_bio/rfd3/model.py (Transition: rms_norm -> silu(x@W1) * (x@W2) -> @W3, every
intermediate a full [I*I, w] tensor in DRAM). The byte model is derived, not measured; that is the
one thing here that is not a measurement.
"""
I = 685
I2 = I * I
BF16 = 2
TFLOP_ROOF = 102.02e12
RD_ROOF = 390.0e9
WR_ROOF = 269.6e9


def transition(c, n, rows, label, ms_per_call, n_inst):
    """Transition(c, n): rms_norm(x) -> a=silu(x@W1), b=x@W2 (both c->n*c), m=a*b, out=m@W3."""
    w = n * c
    flops = n_inst * (2 * rows * c * w * 2 + 2 * rows * w * c)      # W1, W2, W3
    rd = n_inst * BF16 * rows * (c        # rms_norm reads x
                                 + c + c  # W1 and W2 each read the normed x
                                 + w + w  # multiply reads a and b
                                 + w)     # W3 reads m
    wr = n_inst * BF16 * rows * (c        # normed x
                                 + w + w  # a, b
                                 + w      # m
                                 + c)     # out
    return dict(label=label, ms=ms_per_call, flops=flops, rd=rd, wr=wr)


rows = []
# transition_2: two instances of Transition(c_z=128, n=2) over z [1,I,I,128], one row in the probe.
rows.append(transition(128, 2, I2, "transition_2 x2", 29.752, 2))
# each Pairformer block: z_transition = Transition(c_z=128, n=4) over the same [I,I,128] pair
# tensor. s_transition and the attention are O(I) or O(I^2 * 16) and are folded in as measured
# residue, not modelled.
for i, ms in ((0, 25.315), (1, 25.473)):
    rows.append(transition(128, 4, I2, "pairformer[%d].z_transition" % i, ms, 1))

print("%-30s %8s %9s %9s %9s %9s %8s %8s" %
      ("row", "ms/call", "TFLOP/s", "%compute", "GB/s rd", "GB/s wr", "%rd", "%wr"))
tot_ms = tot_f = tot_rd = tot_wr = 0.0
for r in rows:
    s = r["ms"] / 1e3
    tf, grd, gwr = r["flops"] / s, r["rd"] / s, r["wr"] / s
    print("%-30s %8.3f %9.2f %8.1f%% %9.1f %9.1f %7.1f%% %7.1f%%"
          % (r["label"], r["ms"], tf / 1e12, 100 * tf / TFLOP_ROOF, grd / 1e9, gwr / 1e9,
             100 * grd / RD_ROOF, 100 * gwr / WR_ROOF))
    tot_ms += r["ms"]; tot_f += r["flops"]; tot_rd += r["rd"]; tot_wr += r["wr"]

s = tot_ms / 1e3
print("%-30s %8.3f %9.2f %8.1f%% %9.1f %9.1f %7.1f%% %7.1f%%"
      % ("TOTAL per encoder call", tot_ms, tot_f / s / 1e12, 100 * tot_f / s / TFLOP_ROOF,
         tot_rd / s / 1e9, tot_wr / s / 1e9,
         100 * (tot_rd / s) / RD_ROOF, 100 * (tot_wr / s) / WR_ROOF))
print()
print("arithmetic intensity      %.1f FLOP/byte against a machine balance of 261.6" %
      (tot_f / (tot_rd + tot_wr)))
print("time if read+write both ran at roof: %.2f ms/call against %.2f measured"
      % (1e3 * max(tot_rd / RD_ROOF, tot_wr / WR_ROOF), tot_ms))
print("ms/step at 2 calls/step: %.1f measured" % (2 * tot_ms))

fused_rd = tot_rd - BF16 * (2 * I2 * 256 + I2 * 512)   # m no longer read back; a,b never land
fused_wr = tot_wr - BF16 * (2 * I2 * 256 + I2 * 512) * 2
print()
print("if silu(x@W1)*(x@W2) fused so a, b and m never reach DRAM:")
print("  bytes %.2f GB -> %.2f GB, i.e. %.2fx less traffic"
      % ((tot_rd + tot_wr) / 1e9, (fused_rd + fused_wr) / 1e9,
         (tot_rd + tot_wr) / (fused_rd + fused_wr)))
print("  at the SAME measured GB/s that is %.1f ms/call, %.1f ms/step, saving %.1f ms/step"
      % (tot_ms * (fused_rd + fused_wr) / (tot_rd + tot_wr),
         2 * tot_ms * (fused_rd + fused_wr) / (tot_rd + tot_wr),
         2 * tot_ms * (1 - (fused_rd + fused_wr) / (tot_rd + tot_wr))))

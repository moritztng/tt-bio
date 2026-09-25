#!/usr/bin/env python3
"""Compose `MSAModuleBlock`'s residual: main's host-parked depth chunks with of3t-msafwd's fp32 z.

main (MGX, 2026-09-23) added a chunk-list branch, `msa_update_chunks(m, z, ...)`, for alignments
parked on the host; `of3t-msafwd` holds the pair residual in fp32 under a tape
(`z_fp32_residual`, default off) and casts z back to bf16 for the PWA read. Both are kept: the
wide residual first, then either branch, each reading a bf16 z.

Usage: resolve_msa_block_residual.py <file>  ->  0 resolved, 2 not this shape (caller must stop).
"""
import ast
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
m = re.search(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", s, re.S)
if not m:
    sys.exit(2)
head, row = m.group(1), m.group(2)
need_h = ("msa_update_chunks(m, z, self.pwa, self.msa_transition, attn_mask, park=True)",
          "isinstance(m, list)")
need_r = ("wide = self.z_fp32_residual and ops.taping()", "_z_residual(z, z, upd, True)",
          "ttnn.typecast(z, ttnn.bfloat16)")
if not all(k in head for k in need_h) or not all(k in row for k in need_r):
    sys.exit(2)
I = " " * 8
new = (f"{I}# Training only (PairformerLayer's z_fp32_residual): the pair residual held in fp32.\n"
       f"{I}wide = self.z_fp32_residual and ops.taping()\n"
       f"{I}z = _z_residual(z, z, upd, True) if wide else ttnn.add_(upd, z)\n"
       f"{I}if self.has_msa_update and isinstance(m, list):\n"
       f"{I}    zb = ttnn.typecast(z, ttnn.bfloat16) if z.dtype == ttnn.float32 else z\n"
       f"{I}    m = msa_update_chunks(m, zb, self.pwa, self.msa_transition, attn_mask, park=True)\n"
       f"{I}    if zb is not z:\n"
       f"{I}        ttnn.deallocate(zb)\n"
       f"{I}elif self.has_msa_update:\n"
       f"{I}    zc = ttnn.typecast(z, ttnn.bfloat16) if z.dtype == ttnn.float32 else ttnn.clone(z)\n"
       f"{I}    upd = ttnn.reshape(self.pwa(m, zc, attn_mask), tuple(m.shape))\n")
out = s[:m.start()] + new + s[m.end():]
if "<<<<<<< " not in out:
    try:
        ast.parse(out)
    except SyntaxError:
        sys.exit(2)
p.write_text(out)

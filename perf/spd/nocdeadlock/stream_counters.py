"""Compute CB counters from the NoC-overlay stream registers of a hung Blackhole chip (tiles_received = reg 10,
tiles_acked = reg 8 of stream <cb>). TRISC-private cb_interface is not readable on Blackhole (tt-exalens#528).
usage: TT_VISIBLE_DEVICES=<chip> ~/triage_venv/bin/python stream_counters.py <x-y> [<x-y> ...]
"""
import sys
from ttexalens import tt_exalens_lib as L
ctx = L.init_ttexalens()
for core in sys.argv[1:]:
    out = []
    for cb in range(32):
        r = L.read_word_from_device(core, 0xFFB40000 + cb * 0x1000 + 10 * 4, context=ctx) & 0xFFFF
        a = L.read_word_from_device(core, 0xFFB40000 + cb * 0x1000 + 8 * 4, context=ctx) & 0xFFFF
        if r or a:
            out.append(f"cb{cb} recv={r} acked={a} diff={(r - a) & 0xFFFF}")
    print(core, " | ".join(out), flush=True)

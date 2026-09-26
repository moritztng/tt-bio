#!/usr/bin/env python3
"""Reproduce issue #23 (DGX Spark GB10) against whatever BindCraft 2 tree is on PYTHONPATH.

nvidia-smi on a GB10 exits 0 and prints [N/A] for the memory fields; the stub on PATH does the
same. No accelerator is opened: design_devices() is stubbed, exactly as in fix_check.py.
"""
import os, sys
from bindcraft import design_workers as dw

class Device:
    def __init__(self, i, kind='cuda', total_gb=119.0):
        self.id, self.kind, self.total_gb = i, kind, total_gb
    def __str__(self): return f'{self.kind}:{self.id}'
    platform = property(lambda self: self.kind)
    def memory_stats(self):
        return {'bytes_limit': int(self.total_gb * 1024 ** 3), 'bytes_in_use': 0}

if hasattr(dw, 'design_devices'):
    dw.design_devices = lambda: [Device(0)]
for v in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
    os.environ.pop(v, None)

print('nvidia-smi on PATH:', os.popen('nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits').read().strip())
for label, call in (('design_gpu_memory_gb()', dw.design_gpu_memory_gb),
                    ('plan_design_workers({})', lambda: dw.plan_design_workers({}, residue_count=201))):
    try:
        print(f'  {label:26s} -> {call()}')
    except Exception as e:
        print(f'  {label:26s} -> RAISES {type(e).__name__}: {e}')

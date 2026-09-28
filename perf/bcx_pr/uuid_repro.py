#!/usr/bin/env python3
"""Reproduce issue #24 (a UUID in CUDA_VISIBLE_DEVICES) against whatever BindCraft 2 tree is on PYTHONPATH.

Two boards on an nvidia-smi stub: index 0 with 24 GB, index 1 with 96 GB. The job selects board 1,
first by index and then by UUID. No accelerator is opened: design_devices() is stubbed at the one
device the process would see, exactly as in gb10_repro.py.

Run:  PYTHONPATH=<tree> JAX_PLATFORMS=cpu python3 uuid_repro.py
"""
import os, stat, sys, tempfile

BOARDS = [('0', 'GPU-0a1b2c3d-0000-0000-0000-000000000000', 24576, 24576),
          ('1', 'GPU-9f8e7d6c-1111-1111-1111-111111111111', 98304, 98304)]
FIELDS = {'index': 0, 'uuid': 1, 'memory.free': 2, 'memory.total': 3}

stub_directory = tempfile.mkdtemp()
stub = os.path.join(stub_directory, 'nvidia-smi')
with open(stub, 'w') as handle:
    handle.write(f'''#!{sys.executable}
import sys
boards, fields = {BOARDS!r}, {FIELDS!r}
query = next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--query-gpu='))
for board in boards:
    print(', '.join(str(board[fields[name]]) for name in query.split(',')))
''')
os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
os.environ['PATH'] = stub_directory + os.pathsep + os.environ['PATH']

from bindcraft import design_workers as dw

class Device:
    id, platform = 0, 'cuda'
    def __str__(self): return 'cuda:0'
    def memory_stats(self): return {}

if hasattr(dw, 'design_devices'):
    dw.design_devices = lambda: [Device()]

for selection in (BOARDS[1][0], BOARDS[1][1]):
    os.environ['CUDA_VISIBLE_DEVICES'] = selection
    plan = dw.plan_design_workers({}, residue_count=360)
    print(f'CUDA_VISIBLE_DEVICES={selection}')
    print(f'  memory keys   -> {sorted(dw.design_gpu_memory_gb())}')
    print(f'  workers       -> {len(plan)} on {sorted({w["gpu"] for w in plan})}, memory_fraction {plan[0]["memory_fraction"]}')

#!/usr/bin/env python3
"""Seven behaviours of BindCraft 2's design-worker fan-out, checked against whatever tree is on
PYTHONPATH. Run it on PR #19 head 68b853dde for the control and on the same tree with the fix
applied for the result.

`design_devices()` is stubbed because this machine has no accelerator; everything else is the
module as written. Case E is the one that matters most: it is the regression the fix first
offered in the PR comment would have introduced, and the reason that fix was withdrawn.

Run:  PYTHONPATH=<tree> JAX_PLATFORMS=cpu python3 fix_check.py
"""
import os
import stat
import sys
import tempfile
from bindcraft import design_workers as dw


GB10_NVIDIA_SMI = """#!/bin/sh
# A DGX Spark GB10 shares one memory pool with the host, so nvidia-smi cannot measure a free and
# a total for the board: it prints [N/A] for both and exits 0. Issue #23.
case "$*" in
  *memory.free*) echo "0, [N/A], [N/A]" ;;
  *index*)       echo "0" ;;
esac
exit 0
"""


def nvidia_smi_printing_not_available():
    """Put a GB10's nvidia-smi first on PATH. Returns the directory so the caller can undo it."""
    directory = tempfile.mkdtemp(prefix='gb10-')
    stub = os.path.join(directory, 'nvidia-smi')
    with open(stub, 'w') as handle:
        handle.write(GB10_NVIDIA_SMI)
    os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
    os.environ['PATH'] = directory + os.pathsep + os.environ['PATH']
    return directory


class Device:
    def __init__(self, i, kind, total_gb=80.0):
        self.id, self.kind, self.total_gb = i, kind, total_gb
    def __str__(self):
        return f'{self.kind}:{self.id}'
    platform = property(lambda self: self.kind)
    def memory_stats(self):
        return {'bytes_limit': int(self.total_gb * 1024 ** 3), 'bytes_in_use': 0}


def devices(kind, count):
    dw.design_devices = lambda: [Device(i, kind) for i in range(count)]


def only(*names):
    for variable in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
        os.environ.pop(variable, None)
    for name, value in zip(names[::2], names[1::2]):
        os.environ[name] = value


def pins(plan):
    """What launch_design_workers would put in each worker's environment, without spawning one."""
    resolve = getattr(dw, 'design_visibility_variable', lambda: None)  # absent at 301efdd and 3e3563894
    variable = resolve() or 'CUDA_VISIBLE_DEVICES'
    return variable, [worker['gpu'] for worker in plan]


def attempt(call):
    try:
        return call(), None
    except Exception as error:
        return None, f'{type(error).__name__}: {error}'


def main() -> int:
    checks = []

    # A -- a platform that is neither cuda nor rocm designs on one worker, which is what the
    # comment under the guard in plan_design_workers promises.
    devices('tpu', 2)
    only()
    plan, error = attempt(lambda: dw.plan_design_workers({}))
    checks.append(('A  third platform, 2 devices -> a 1-worker plan, not a raise',
                   error is None and plan is not None and len({w['gpu'] for w in plan}) == 1,
                   error or f'{len(plan)} worker(s) on {sorted({w["gpu"] for w in plan})}'))

    # B -- the process is given part of the box. jax renumbers from 0; the pins must not.
    devices('cuda', 2)
    only('CUDA_VISIBLE_DEVICES', '2,3')
    gpus, error = attempt(dw.selected_design_gpus)
    checks.append(('B  CUDA_VISIBLE_DEVICES=2,3 -> the cards the job was given',
                   gpus == ['2', '3'], error or str(gpus)))
    plan, error = attempt(lambda: dw.plan_design_workers({'workers_per_gpu': 1}))
    variable, pinned = pins(plan) if plan is not None else ('?', [])
    checks.append(('B  workers pinned to 2 and 3, not 0 and 1',
                   pinned == ['2', '3'], error or f'{variable}={pinned}'))

    # B2 -- the packing decision must read the memory of the card the worker runs on. With
    # nvidia-smi absent the jax reading is used, and its keys have to match the pin names.
    memory, error = attempt(dw.design_gpu_memory_gb)
    checks.append(('B  free memory keyed by the pin name, not the jax id',
                   memory is not None and sorted(memory) == ['2', '3'],
                   error or str(sorted(memory or {}))))

    # C -- the case the PR exists for: no variable set, no nvidia-smi, fan out over the whole box.
    devices('rocm', 4)
    only()
    gpus, error = attempt(dw.selected_design_gpus)
    checks.append(('C  rocm, no variable set -> the whole box',
                   gpus == ['0', '1', '2', '3'], error or str(gpus)))

    # D -- same as B on the other vendor's variable. Two entries, two devices: a process given
    # part of a box sees only the part.
    devices('rocm', 2)
    only('HIP_VISIBLE_DEVICES', '2,3')
    gpus, error = attempt(dw.selected_design_gpus)
    checks.append(('D  HIP_VISIBLE_DEVICES=2,3 -> the cards the job was given',
                   gpus == ['2', '3'], error or str(gpus)))

    # D2 -- the entries do not describe these devices, which CUDA allows: it stops at the first
    # unparseable entry, so the variable can name more cards than the process ends up with. Fall
    # back to jax ids wholesale rather than pinning some workers by name and some by id.
    devices('cuda', 2)
    only('CUDA_VISIBLE_DEVICES', '2,3,bogus,7')
    gpus, error = attempt(dw.selected_design_gpus)
    checks.append(('D  variable does not describe the devices -> jax ids, no mixed list',
                   gpus == ['0', '1'], error or str(gpus)))

    # E -- a box with no accelerator plans nothing and the campaign runs in one process, which is
    # what both 301efdd and 68b853dde do. A fix that resolves the variable eagerly raises here.
    devices('cpu-only', 0)
    only()
    gpus, error = attempt(dw.selected_design_gpus)
    checks.append(('E  no accelerator -> no fan-out, no raise',
                   gpus == [], error or str(gpus)))
    plan, error = attempt(lambda: dw.plan_design_workers({}))
    checks.append(('E  no accelerator -> an empty plan, no raise',
                   error is None and plan == [], error or str(plan)))

    # F -- issue #23, reported against 3e3563894 by a third party on a DGX Spark GB10. nvidia-smi
    # is present and exits 0, and prints [N/A] where the memory numbers go, so the float() in
    # nvidia_smi_memory_gb raises out through plan_design_workers and the campaign refuses. The
    # guard drops the unmeasurable rows, which lets the plugin's own reading answer instead.
    devices('cuda', 1)
    only()
    stub_directory = nvidia_smi_printing_not_available()
    memory, error = attempt(dw.design_gpu_memory_gb)
    checks.append(('F  nvidia-smi reports [N/A] memory -> a reading, not a raise',
                   error is None and memory == {'0': (80.0, 80.0)}, error or str(memory)))
    plan, error = attempt(lambda: dw.plan_design_workers({}, residue_count=201))
    checks.append(('F  nvidia-smi reports [N/A] memory -> a plan, not a refused campaign',
                   error is None and plan is not None and len(plan) >= 1, error or f'{len(plan)} worker(s)'))
    os.environ['PATH'] = os.environ['PATH'].split(os.pathsep, 1)[1]

    for label, ok, detail in checks:
        print(f"  {'pass' if ok else 'FAIL'}  {label}\n              {detail}")
    failed = [label for label, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

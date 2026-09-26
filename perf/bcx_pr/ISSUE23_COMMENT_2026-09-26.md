That error is in `bindcraft/design_workers.py`, in `design_gpu_memory_gb`:

```python
return {index: (float(free_mib) / 1024, float(total_mib) / 1024) for index, free_mib, total_mib in fields}
```

The GB10 shares one memory pool with the host, so `nvidia-smi --query-gpu=memory.free,memory.total` has no board figure to give: it prints `[N/A]` in both fields and exits 0. `check=True` does not trip on a zero exit and the `except (OSError, CalledProcessError)` does not catch a `ValueError`, so it raises out through `plan_design_workers`, and `cli.py` turns it into `campaign refused` before the first trajectory.

Reproduced at `3e3563894` with an `nvidia-smi` stub that prints what yours does:

```
design_gpu_memory_gb()  -> ValueError: could not convert string to float: '[N/A]'
plan_design_workers({}) -> ValueError: could not convert string to float: '[N/A]'
```

Your patch is the right shape, and with it the same repro gives `{}` and a single-process campaign rather than a refusal.

#19 now carries the same guard, written to drop only the rows that do not parse, so a board that reports numbers is read exactly as before and one that cannot falls through to the PJRT plugin's own `memory_stats()`. Whether the CUDA plugin reports a `bytes_limit` for the GB10's unified pool I cannot test, having no such box: if it does you get memory-aware worker packing, if it does not you get the one worker your patch gives. Either way the campaign runs.

# Phase 0 rental in flight — resume instructions

instance id: 48125033  (vast.ai, offer 45315037, machine 98731, H200 NVL 143771 MiB, $3.71/hr)
label: freebindcraft-feasibility
ssh: `ssh -p 15032 root@ssh9.vast.ai`   (StrictHostKeyChecking=no)
rented 2026-08-19 15:29Z. Credit before: 25.38484706055459.

## State at 2026-08-19 16:13Z

Setup is DONE and verified: BindCraft env built, AF2 params on disk (15 models, 5.4 GB),
`jax.devices()` = `[CudaDevice(id=0)]`, timing shim applied to both modules and both re-parsed.

The timed run is RUNNING, launched 15:48Z, detached under `nohup bash /root/fbc/runonly.sh`:
  /work/run-driver.log   driver stdout
  /work/out/run.log      the --verbose run log the parser reads
  /work/out/stages.jsonl the shim records
  /work/RUN_FINISHED     touched when the run + parse + hash have finished

At 16:13Z: 3 of 6 relaxed trajectory PDBs, 7 trajectories started, 2 accepted designs.
`max_trajectories` counts relaxed PDBs in Trajectory/Relaxed, not started trajectories, so
trajectories killed by the pLDDT gate do not count toward the 6. ~8 min per relaxed trajectory.

## What the resuming pass must do

1. `ssh -p 15032 root@ssh9.vast.ai 'ls /work/RUN_FINISHED; tail -20 /work/out/run.log'`
   If RUN_FINISHED is absent and no python is running, re-run:
   `nohup bash /root/fbc/runonly.sh > /work/run-driver.log 2>&1 </dev/null &`
2. `gpu_fbc_run.sh` already runs parse_fbc_run.py and writes split.json + sha256sums.txt itself.
   Copy `/work/out` back, verify the hashes on the receiving side, THEN destroy.
3. `export VAST_API_KEY="$(tr -d '[:space:]' < ~/vast.txt)"`
   `~/.vast-venv/bin/vastai destroy instance 48125033 -y`
   then `~/.vast-venv/bin/vastai show instances` until 48125033 is gone. Re-destroy until clean.
   Re-read credit after a 10-20 min delay before quoting a final spend figure.
4. Write the measured split into section 1 of the state doc and flip the header to
   `VERDICT: NO-GO — ...`.

## Do not touch

instance 48122246 (machine 146954) belongs to the concurrent `pxdesign-gpu-reference-vast` worker.

## Instrument caveat found this pass

`gpumem.csv` peaks at 143771 MiB, exactly the card's total, because XLA preallocates. So
`peak_gpu_mem_mib` in split.json is NOT the workload footprint and must not be quoted as one. A
real footprint needs `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which is not worth a second rental;
upstream's own ">= 32 GB" recommendation stands as the memory requirement instead.

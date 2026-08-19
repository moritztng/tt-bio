# Phase 0 rental in flight

instance id: 48125033   (vast.ai, offer 45315037, machine 98731, H200 NVL 141GB, $3.607/hr)
label: freebindcraft-feasibility
rented: 2026-08-19 ~17:35Z by the execution pass
credit before: 25.38484706055459

If you are a relaunch and this file is still here with no `perf/freebindcraft/split.json` committed:
  export VAST_API_KEY="$(tr -d '[:space:]' < ~/vast.txt)"
  ~/.vast-venv/bin/vastai show instances          # is 48125033 still up?
  ~/.vast-venv/bin/vastai show instance 48125033  # ssh addr + port
Reuse it (do not rent a second one), collect /work/out, then destroy it and verify with
`vastai show instances` until the list is clean.

NOTE: instance 48122246 (machine 146954) belongs to the concurrent `pxdesign-gpu-reference-vast`
worker. Do NOT destroy it.

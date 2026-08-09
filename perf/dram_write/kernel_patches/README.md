# Writer-kernel experiments

Throwaway patchers used to test what limits the matmul's DRAM writeback. They rewrite the two
matmul writer kernels in a tt-metal source checkout from a `.dws_backup` of the pristine file, so
each run starts from stock. **Both kernels were restored to pristine after every experiment; nothing
here is applied to any checkout.**

- `patch_recv.py` — adds `DeviceZoneScopedN` around the writer's per-subblock `cb_out.wait_front`
  and `noc_async_write_barrier`. This is what produced the retirement curve in `results_exec.json`.
  Note that only the *in1 sender* core runs `..._in1_sender_writer_padding.cpp`; the other 127 cores
  run the receiver variant, so both have to be instrumented or you see one core.
- `patch_depth.py D` — one write barrier per D out-subblocks, waiting for all D up front.
- `patch_pipe.py D` — one write barrier per D out-subblocks, waiting per subblock so the writes
  still start as early as the stock kernel does.
- `an_zone.py` — reads the zones back out of `profile_log_device.csv`.

Both depth variants are correct but pointless: see the verdicts in `results_exec.json`.

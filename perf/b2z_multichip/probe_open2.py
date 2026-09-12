"""Second attempt at bringing fabric up on this WH Galaxy: sweep fabric mode x reliability mode.

The box reports one chip (UMD 1) with ETH_STATUS 0x1012222/0x10010 where all 31 others read
0x22222222, i.e. a partially untrained link. RELAXED_INIT exists exactly for that case, so it is
the first thing to try before anything that touches other workers' chips.
"""
import os, sys, time
import ttnn

FAB = {n: getattr(ttnn.FabricConfig, n) for n in
       ("FABRIC_1D", "FABRIC_1D_NEIGHBOR_EXCHANGE", "FABRIC_1D_RING", "FABRIC_2D")}
REL = {n: getattr(ttnn.FabricReliabilityMode, n) for n in
       ("RELAXED_INIT", "DYNAMIC_RECONFIG", "STRICT_INIT")}

shape = tuple(int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2").split(","))

for fname, f in FAB.items():
    for rname, r in REL.items():
        tag = f"{shape}/{fname}/{rname}"
        t0 = time.time()
        try:
            try:
                ttnn.set_fabric_config(f, r)
            except TypeError:
                ttnn.set_fabric_config(f, reliability_mode=r)
            m = ttnn.open_mesh_device(ttnn.MeshShape(*shape),
                                      dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER))
            n = m.get_num_devices()
            ttnn.close_mesh_device(m)
            print(f"OK   {tag:52s} n={n} {time.time()-t0:.1f}s", flush=True)
        except Exception as e:                                    # noqa: BLE001
            msg = [l for l in str(e).splitlines() if l.strip() and not l.startswith(" ---")]
            print(f"FAIL {tag:52s} {time.time()-t0:.1f}s {msg[-1][:150] if msg else type(e).__name__}", flush=True)
        finally:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
print("SWEEP-DONE", flush=True)

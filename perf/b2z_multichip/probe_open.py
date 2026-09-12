"""Find a (mesh shape, fabric, dispatch) triple that actually opens on this Galaxy."""
import sys, time, traceback
import ttnn

CASES = []
for shape in ((1, 2), (2, 1), (1, 4), (8, 4), (1, 8)):
    for fab in ("1d", "2d", "off"):
        CASES.append((shape, fab, "worker"))

FAB = {"1d": ttnn.FabricConfig.FABRIC_1D, "2d": ttnn.FabricConfig.FABRIC_2D,
       "off": ttnn.FabricConfig.DISABLED}

only = sys.argv[1] if len(sys.argv) > 1 else None
for shape, fab, disp in CASES:
    tag = f"{shape}/{fab}/{disp}"
    if only and only not in tag:
        continue
    t0 = time.time()
    try:
        ttnn.set_fabric_config(FAB[fab])
        m = ttnn.open_mesh_device(ttnn.MeshShape(*shape),
                                  dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER))
        n = m.get_num_devices()
        ttnn.close_mesh_device(m)
        print(f"OK   {tag:24s} n={n} {time.time()-t0:.1f}s", flush=True)
    except Exception as e:                                        # noqa: BLE001
        print(f"FAIL {tag:24s} {time.time()-t0:.1f}s {type(e).__name__}: {str(e).splitlines()[-1][:160]}", flush=True)
    finally:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass

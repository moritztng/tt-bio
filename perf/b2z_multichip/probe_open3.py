"""Open a 2-chip fabric mesh PINNED to a free adjacent pair, so the 10 sibling workers on this
box keep their chips.

Unpinned, tt-metal's UMD opens all 32 ASICs and then blocks on `CHIP_IN_USE_<n>_PCIe` held by a
sibling. TT_VISIBLE_DEVICES is UMD logical numbering and must be set before ttnn is imported.
"""
import os, sys, time

vis = sys.argv[1] if len(sys.argv) > 1 else "24,28"
os.environ["TT_VISIBLE_DEVICES"] = vis
import ttnn                                                      # noqa: E402

print("pinned to", vis, "-> num devices seen:", ttnn.GetNumAvailableDevices(), flush=True)
try:
    print(ttnn.visualize_system_mesh(), flush=True)
except Exception as e:
    print("visualize failed:", e, flush=True)

FAB = ["FABRIC_1D", "FABRIC_1D_NEIGHBOR_EXCHANGE", "FABRIC_1D_RING", "FABRIC_2D"]
REL = ["RELAXED_INIT", "STRICT_INIT"]
for fn in FAB:
    for rn in REL:
        t0 = time.time()
        try:
            try:
                ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fn), getattr(ttnn.FabricReliabilityMode, rn))
            except TypeError:
                ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fn))
            m = ttnn.open_mesh_device(ttnn.MeshShape(1, 2),
                                      dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER))
            n = m.get_num_devices()
            ttnn.close_mesh_device(m)
            print(f"OK   {fn}/{rn} n={n} {time.time()-t0:.1f}s", flush=True)
        except Exception as e:                                   # noqa: BLE001
            msg = [l for l in str(e).splitlines() if l.strip() and not l.startswith(" ---")]
            print(f"FAIL {fn}/{rn} {time.time()-t0:.1f}s {msg[-1][:160] if msg else type(e).__name__}", flush=True)
        finally:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
print("SWEEP-DONE", flush=True)

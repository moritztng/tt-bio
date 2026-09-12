"""Is there any chip-to-chip ethernet on this box at all?"""
import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
m = ttnn.open_mesh_device(ttnn.MeshShape(8, 4),
                          dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER))
devs = m.get_devices()
print("n devices:", len(devs))
d0 = devs[0]
print("device attrs:", sorted(n for n in dir(d0) if "eth" in n.lower() or "core" in n.lower()))
ids = [d.id() for d in devs]
print("ids:", ids)
total = 0
for d in devs:
    for other in ids:
        if other == d.id():
            continue
        try:
            s = d.get_ethernet_sockets(other)
        except Exception as e:
            s = None
        if s:
            total += len(s)
            print(f"  {d.id()} -> {other}: {len(s)} eth sockets")
print("TOTAL eth links found:", total)
ttnn.close_mesh_device(m)

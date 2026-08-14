import os, sys, time
import ttnn

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
shape = ttnn.MeshShape(1, n)
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    print("fabric_config: FABRIC_1D set")
except Exception as e:
    print("fabric_config failed:", e)
t0 = time.time()
md = ttnn.open_mesh_device(shape)
print("opened mesh", n, "in %.2f s" % (time.time() - t0))
print("num_devices", md.get_num_devices())
print("shape", md.shape)
ttnn.close_mesh_device(md)
print("closed OK")

import sys, time
import ttnn
r, c = int(sys.argv[1]), int(sys.argv[2])
fab = sys.argv[3] if len(sys.argv) > 3 else "1d"
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if fab == "1d" else ttnn.FabricConfig.DISABLED)
t0 = time.time()
md = ttnn.open_mesh_device(ttnn.MeshShape(r, c))
print("OPENED %dx%d in %.2f s, num_devices=%d, shape=%s" % (r, c, time.time() - t0, md.get_num_devices(), md.shape))
ttnn.close_mesh_device(md)
print("CLOSED OK")

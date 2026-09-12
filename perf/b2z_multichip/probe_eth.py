"""What does the cluster actually report about chip-to-chip ethernet on this box?"""
import glob, json, os, subprocess
import ttnn

print("=== arch / cluster ===")
print("arch:", ttnn.get_arch_name(), "num_pcie:", ttnn.GetNumPCIeDevices(), "num_dev:", ttnn.GetNumAvailableDevices())
print("=== cluster descriptor files ===")
for pat in ("/tmp/*cluster_descriptor*.yaml", "/tmp/**/cluster_descriptor*.yaml", "/tmp/umd/**/*.yaml"):
    for f in glob.glob(pat, recursive=True):
        print(" ", f, os.path.getsize(f))
print("=== fabric helpers ===")
for n in ("get_fabric_config", "get_tt_fabric_max_payload_size_bytes", "get_tt_fabric_packet_header_size_bytes"):
    try:
        print(n, getattr(ttnn, n)())
    except Exception as e:
        print(n, "ERR", e)

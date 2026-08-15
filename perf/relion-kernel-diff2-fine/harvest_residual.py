import struct, sys
EXE = "/home/ttuser/relion-scratch/relion/build-fine/bin/relion_refine_mpi"
# nm virtual addresses (PIE, relative to load base)
OFF = {"bucket": 0x49f7e0, "exact": 0x49f830, "n": 0x49f838, "max": 0x49f840}

def base_of(pid):
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            if line.rstrip().endswith(EXE):
                return int(line.split("-")[0], 16)
    raise SystemExit(f"exe mapping not found for {pid}")

tot_n = tot_exact = 0
tot_bucket = [0]*10
tot_max = 0.0
for pid in sys.argv[1:]:
    b = base_of(pid)
    with open(f"/proc/{pid}/mem", "rb") as m:
        m.seek(b + OFF["bucket"]); bucket = list(struct.unpack("<10q", m.read(80)))
        m.seek(b + OFF["exact"]);  exact  = struct.unpack("<q", m.read(8))[0]
        m.seek(b + OFF["n"]);      n      = struct.unpack("<q", m.read(8))[0]
        m.seek(b + OFF["max"]);    mx     = struct.unpack("<d", m.read(8))[0]
    print(f"pid {pid}: n={n} bit_identical={exact} max_rel={mx:.3e} bucket={bucket}")
    tot_n += n; tot_exact += exact; tot_max = max(tot_max, mx)
    tot_bucket = [a+c for a, c in zip(tot_bucket, bucket)]
print(f"TOTAL n={tot_n} bit_identical={tot_exact} ({100.0*tot_exact/max(tot_n,1):.3f}%) max_rel={tot_max:.6e}")
print("by -log10(rel):", " ".join(f"{k}:{v}" for k, v in enumerate(tot_bucket)))

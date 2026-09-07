"""Turn one rung's raw log into a JSONL record: capacity, both memory axes, and on a
failure the allocator's own byte counts.

The requested-vs-available numbers are the point of the whole exercise: the Wormhole
failures the brief cites name an 8790736896 B request against a 12 GiB card, so a
Blackhole record that only said "ok" would not be comparable to them.
"""
import glob, json, re, sys, os, pathlib

model, rung, recyc, samp, probe, rc, wall, out = sys.argv[1:9]
out = pathlib.Path(out)
log = (out / "fold.log").read_text(errors="replace") if (out / "fold.log").exists() else ""

# Status as the engine itself recorded it, not as the exit code implies: a nonzero rc with
# an ok results.json (or the reverse) is a harness fact, and both are kept.
status = "NORESULT"
if int(rc) == 124:
    status = "TIMEOUT"
for g in glob.glob(str(out / "*" / "results.json")) + glob.glob(str(out / "results.json")):
    try:
        status = json.load(open(g))[0]["status"]
        break
    except Exception:
        pass

# The allocator's refusal, verbatim in numbers. Every OOM in the cited job logs is this shape.
oom = None
m = re.search(r"Not enough space to allocate (\d+) B (\w+) buffer across (\d+) banks, where each "
              r"bank needs to store (\d+) B, but bank size is (\d+) B "
              r"\(allocated: (\d+) B, free: (\d+) B, largest free block: (\d+) B\)", log)
if m:
    oom = dict(requested_B=int(m[1]), space=m[2], banks=int(m[3]), per_bank_needed_B=int(m[4]),
               bank_size_B=int(m[5]), allocated_B=int(m[6]), free_B=int(m[7]),
               largest_free_block_B=int(m[8]))
elif re.search(r"Not enough space to allocate (\d+) B", log):
    oom = dict(requested_B=int(re.search(r"Not enough space to allocate (\d+) B", log)[1]))

# Host RAM: the floor of MemAvailable and the ceiling of summed python3 RSS. A host OOM is
# a different verdict from a device ceiling, so it has to be visible in the record.
min_avail_kb, peak_rss_kb = None, None
rl = out / "hostram.log"
if rl.exists():
    for line in rl.read_text(errors="replace").splitlines():
        p = line.split()
        if len(p) == 2 and p[0] == "RSS":
            peak_rss_kb = max(peak_rss_kb or 0, int(p[1]))
        elif len(p) == 2 and p[1].isdigit():
            min_avail_kb = int(p[1]) if min_avail_kb is None else min(min_avail_kb, int(p[1]))

peak_dram_B, dram_lines = None, 0
pk = out / "dram_peak.log"
if pk.exists():
    gib = [float(x) for x in re.findall(r"\[DRAM\] .*?: ([\d.]+) GiB used", pk.read_text(errors="replace"))]
    dram_lines = len(gib)
    if gib:
        peak_dram_B = int(max(gib) * 2**30)

# Where a stalled rung was when its budget ran out. A rung that does not finish is only
# useful if you can say which op it died in: pass 1's OF3 1024 sat 11+ min inside one
# ttnn.to_torch of a [1024] token mask, which is a wedge, not slow compute -- and that
# distinction is the difference between a size wall and a sick card.
stall_frame, stall_tau = None, None
st = out / "stack.log"
if st.exists():
    txt = st.read_text(errors="replace")
    frames = re.findall(r"^ +(\S+ \(tt_bio/\S+:\d+\))", txt, re.M)
    if frames:
        stall_frame = frames[-1]
    taus = re.findall(r"^ +tau: (\d+)", txt, re.M)
    if taus:
        stall_tau = int(taus[-1])

host_oom = bool(re.search(r"MemoryError|Killed|Cannot allocate memory|std::bad_alloc|"
                          r"DefaultCPUAllocator: can't allocate", log))

conclusive = (status in ("ok", "TIMEOUT")) or (oom is not None) or host_oom

rec = dict(model=model, rung=rung, tokens=int(re.search(r"(\d+)$", rung)[1]), msa_rows=14190,
           recycling_steps=int(recyc), sampling_steps=int(samp), probe=probe,
           rc=int(rc), status=status, wall_s=int(wall),
           peak_host_rss_kb=peak_rss_kb, min_mem_available_kb=min_avail_kb,
           peak_device_dram_B=peak_dram_B, dram_probe_samples=dram_lines,
           host_oom_signature=host_oom, oom=oom,
           stall_frame=stall_frame, stall_diffusion_step=stall_tau,
           conclusive=conclusive, arch="blackhole", card="pc physical 0", banks=8, bank_size_B=4278190016)
p = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "results.jsonl"
with open(p, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(json.dumps({k: v for k, v in rec.items() if v not in (None, False)}, indent=1))

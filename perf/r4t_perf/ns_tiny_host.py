"""r4t-perf: price host bf16 Newton-Schulz (upstream order) for the 23 RFD4 matrices with min dim < 64 (B1 double-prime option iii). qb2 8 cores: 1.5 ms."""
import sys, time, torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from b1prime import ns_ref
sh = [((12,384),14),((3,768),1),((44,32),1),((6,32),1),((1473,32),1),((11,16),1),((128,16),1),((7,32),1),((4,16),1),((32,768),1)]
ws = [torch.randn(n, *s) for s, n in sh]
ws = [w.transpose(-1,-2).contiguous() if w.shape[-2] > w.shape[-1] else w for w in ws]
for _ in range(3): [ns_ref(w, torch.bfloat16) for w in ws]
t=[]
for _ in range(20):
    t0=time.perf_counter(); [ns_ref(w, torch.bfloat16) for w in ws]; t.append(time.perf_counter()-t0)
print("host bf16 NS, 23 matrices with min dim < 64: min %.4f s median %.4f s, threads %d" % (min(t), sorted(t)[10], torch.get_num_threads()))

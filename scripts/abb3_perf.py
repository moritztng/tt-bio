"""Warm single-card throughput for the hybrid ABodyBuilder3 port.

Warms up (2 runs incl compile), then times W warm runs on the 6yio H0-L0 Fv (N=229)
and reports per-structure wall time + throughput. The on-device pieces (embeddings,
IPA projections + linear_out, LayerNorm, Transition, BackboneUpdate, AngleResnet
linears, pLDDT head) run in bf16; the IPA attention + quaternion compose + atom14
run host fp32 (the documented ceiling). Reference CPU fp32 forward = 225 ms."""
import os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from tt_bio.abodybuilder3_weights import ABB3_CONFIG, ensure_abb3_weights
from tt_bio.abodybuilder3 import abb3_compute_kernel_config, StructureModuleTT
from abodybuilder3_reference import string_to_input, EXAMPLE_HEAVY, EXAMPLE_LIGHT

cache = os.environ.get("TT_BIO_CACHE", "/tmp/abb3_cache")
W = int(os.environ.get("ABB3_WARM_ITERS", 5))
inp = string_to_input(EXAMPLE_HEAVY, EXAMPLE_LIGHT, "cpu")
single, pair, aatype = inp["single"], inp["pair"], inp["aatype"]
mask = torch.ones(single.shape[:-1], dtype=single.dtype)
sd = torch.load(ensure_abb3_weights(cache), map_location="cpu", weights_only=True)
ck = abb3_compute_kernel_config()
model = StructureModuleTT(sd, ck, ABB3_CONFIG)
with torch.no_grad():
    for _ in range(2):
        model(single, pair, aatype, mask)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(W):
        model(single, pair, aatype, mask)
    dt = (time.time() - t0) / W
print(f"hybrid ABodyBuilder3 warm: {dt*1000:.1f} ms/structure  ({1/dt:.2f} structures/s)  N=229, {W} warm iters")

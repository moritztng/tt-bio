"""Frame/alignment probe for the per-block sub-op taps at one block."""
import os, sys
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
from pathlib import Path
import torch, ttnn
from scripts.opendde_real_seam_parity import pcc_ratio
from scripts.opendde_pairformer_block_trace import (
    _load_reference, _dev_setup, ref_sub, dev_sub, ref_block_z, dev_block_z, SUB_OPS)
torch.set_grad_enabled(False)

cache = Path("_run/pf_trace")
BLK = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ref = torch.load(cache / f"ref_z_{BLK:02d}.pt")
src = torch.load(cache / "ref_z_msa.pt") if BLK == 0 else torch.load(cache / f"ref_z_{BLK-1:02d}.pt")
n = src.shape[0]
print(f"block {BLK}: z_in rms={float(src.pow(2).mean().sqrt()):.4f} "
      f"max={float(src.abs().max()):.2f} | z_out rms={float(ref.pow(2).mean().sqrt()):.4f}")

model, p, sd = _dev_setup()
refm, _ = _load_reference(sd)
rb = refm.pairformer_stack.blocks[BLK]
db = p.trunk.PF.blocks[BLK]

ref_taps = {}
ref_out = ref_block_z(rb, src, taps=ref_taps)
print("ref z-only vs cached:", pcc_ratio(ref_out, ref)[0])

dev_taps = {}
zt = p.trunk._up(src.reshape(1, n, n, -1))
dev_out = dev_block_z(db, zt, taps=dev_taps, to_host=lambda t, s=None: p._to_host(t, s))
host_dev_out = p._to_host(dev_out, tuple(ref.shape))
ttnn.deallocate(dev_out)
print("dev block out vs ref:", pcc_ratio(host_dev_out, ref)[0])

for name in SUB_OPS:
    r = ref_taps[name].float()
    g = dev_taps[name].reshape(r.shape).float()
    direct = pcc_ratio(g, r)
    tr = pcc_ratio(g, r.transpose(0, 1).contiguous())
    print(f"{name:16s} direct PCC={direct[0]:8.6f} nr={direct[1]:.4f} | "
          f"vs ref^T PCC={tr[0]:8.6f} nr={tr[1]:.4f} | "
          f"rms_ref={float(r.pow(2).mean().sqrt()):.4e}")

# reassemble the block from the DEVICE taps in the reference's order: if this lands on
# the device block output, the taps are the real sub-op results and only the comparison
# frame was wrong.
z = src.clone()
z = z + dev_taps["tri_mul_out"].reshape(z.shape).float()
z = z + dev_taps["tri_mul_in"].reshape(z.shape).float()
z = z + dev_taps["tri_att_start"].reshape(z.shape).float()
z = z + dev_taps["tri_att_end"].reshape(z.shape).float()
z = z + dev_taps["pair_transition"].reshape(z.shape).float()
print("host-reassembled dev taps vs dev block out:", pcc_ratio(z, host_dev_out)[0])
print("host-reassembled dev taps vs ref:", pcc_ratio(z, ref)[0])

"""E2E before/after on the real cached target (NT=38): default host-heads path
vs TT_PROTENIX_CONF_DEVICE=1. At NT=38 the device path is gated off (NT<128), so
this confirms the default is unchanged and reports the honest e2e split."""
import os, sys, time
os.environ.setdefault("TT_VISIBLE_DEVICES", "0"); os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix
from protenix_confidence_device_parity import load_feats

CKPT = "/home/moritz/.boltz/protenix-v2.pt"


def run(label, n_step=10, n_sample=1):
    feats, _ = load_feats()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)
    m = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
    ttnn.synchronize_device(dev); t0 = time.time()
    coords, conf = m.fold(feats, n_step=n_step, n_sample=n_sample, seed=0, return_confidence=True)
    ttnn.synchronize_device(dev); dt = time.time() - t0
    print(f"[{label}] e2e {dt*1000:.1f} ms  plddt={conf['plddt']:.4f} ptm={conf['ptm']} iptm={conf['iptm']}", flush=True)
    return dt, conf


if __name__ == "__main__":
    run("default host-heads (flag off)")
    os.environ["TT_PROTENIX_CONF_DEVICE"] = "1"
    run("flag on (NT=38 -> host fallback)")
    print("E2E_DONE", flush=True)

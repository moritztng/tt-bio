# ODesign on-device denoiser parity (deterministic, no RNG matching needed):
# replay the reference diffusion trajectory step-by-step using the production
# tt_bio.odesign.ODesign (which reuses tt_bio.protenix.DiffusionModule loaded with
# ODesign weights). For each of the N_step denoiser calls, run
# ODesign.denoise_step(x_noisy_i, t_hat_i, cond) with the fixed trunk conditioning
# (from odesign_denoiser_pre.pkl) and compare to the reference denoised coords
# (odesign_traj.pkl). Validates the ODesign diffusion leg across the full sigma range.
#
# Pass-3 scope: diffusion/denoiser leg only. The trunk (Pairformer + MSA + the
# constraint/hotspot distogram embedder) is NOT exercised -- trunk conditioning is
# read from the captured golden intermediates, so this parity number is independent
# of the (unported) trunk. See tt_bio/odesign.py for what is built vs deferred.
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES', '0')
os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, pickle, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.odesign import ODesign

GOLDEN = '/home/moritz/.coworker/scratch/odesign-ref/golden'
CKPT = '/home/moritz/.coworker/scratch/odesign-ref/ckpt/odesign_base_prot_flex.pt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--pre', default=os.path.join(GOLDEN, 'odesign_denoiser_pre.pkl'))
    ap.add_argument('--traj', default=os.path.join(GOLDEN, 'odesign_traj.pkl'))
    ap.add_argument('--n-steps', type=int, default=None, help='cap replay to N steps (default all 200)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    pre = pickle.load(open(args.pre, 'rb'))
    traj = pickle.load(open(args.traj, 'rb'))

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    model = ODesign.load_from_checkpoint(args.ckpt, compute_kernel_config=ckc, device=dev)
    print('ODesign loaded: %d state-dict keys; N_token=%d N_atom=%d'
          % (len(model._w), pre['N_token'], pre['N_atom']), flush=True)

    pccs, maxerrs = model.replay_trajectory(pre, traj, n_steps=args.n_steps, verbose=not args.quiet)
    if pccs:
        print('\nRESULT denoiser PCC: min %.5f  mean %.5f  max %.5f  (n=%d steps)'
              % (min(pccs), sum(pccs) / len(pccs), max(pccs), len(pccs)), flush=True)
        print('RESULT denoiser maxerr: max %.3e  mean %.3e'
              % (max(maxerrs), sum(maxerrs) / len(maxerrs)), flush=True)


if __name__ == '__main__':
    main()

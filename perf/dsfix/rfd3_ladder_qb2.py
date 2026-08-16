"""RFD3 TT ladder on qb2 at shipped defaults, batch 1, every rung.

Same instrument as rfd3_ladder.py (per-step wall by step-count differential). The qb1
ladder in results/rfd3_tt.jsonl is ttnn 0.67.4 and predates matmul calibration, so it
overstates the ratio to the GPU ladder by up to ~1.8x at R4 and must not be published.
This copy runs qb2 / ttnn 0.68.0 and applies the shipped calibration rule per rung:
sampler.sample is called directly here, which bypasses run_design where the rule lives.

Load-once: weights and per-rung featurisation are paid once for the whole sweep, so the
only thing being timed is sampler.sample. Each (rung, batch) does one discarded warm-up
then n=3 rounds of an N1-leg and an N2-leg, alternating leg order between rounds.

    per_step = (t(N2) - t(N1)) / (N2 - N1)

Results append to perf/dsfix/results/rfd3_tt.jsonl one record per (rung, batch); a rerun
skips any (rung, batch) already present, so a relaunch never redoes a finished point.
"""
import json, os, pathlib, statistics, sys, time
import torch

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3.design import build_token_initializer, build_diffusion_module
from tt_bio.rfd3.sampler import RFD3Sampler
from tt_bio.rfd3.input import InputSpecification
from tt_bio.rfd3.featurize import featurize
from tt_bio.rfd3.model import set_tune_matmul_for_atoms

N1, N2, NREP = 8, 40, 3
# Machine facts from the environment: this ladder has to run on the Wormhole Galaxy too, and the
# tune_matmul arm is per-row, so a row that does not name its host and its arm cannot be compared.
CKPT = pathlib.Path(os.environ.get("RFD3_CKPT", "/home/ttuser/.boltz/rfd3/weights"))
OUT = pathlib.Path(os.environ.get("RFD3_OUT", "perf/dsfix/results/rfd3_tt_qb2.jsonl"))
OUT.parent.mkdir(parents=True, exist_ok=True)
HOST = os.environ.get("RFD3_HOST", "qb2")
CARD = os.environ.get("RFD3_CARD", "0")
TTNN = os.environ.get("RFD3_TTNN", "0.68.0")
LADDER = [(r, [1]) for r in os.environ.get("RFD3_RUNGS", "R0,R1,R2,R3,R4").split(",")]

done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["rung"], r["batch"], r.get("tune_matmul")))
print("[ladder] already done: %s" % sorted(done), flush=True)

dm = build_diffusion_module(torch.load(CKPT / "diffusion_module.real_weights.pt",
                                       map_location="cpu", weights_only=True))
ti = build_token_initializer(torch.load(CKPT / "token_initializer.real_weights.pt",
                                        map_location="cpu", weights_only=True))
print("[ladder] weights loaded", flush=True)


def timed(sampler, D, L, coord0, f, init, is_motif, seed0):
    gens = [torch.Generator().manual_seed(seed0 + i) for i in range(D)]
    t0 = time.perf_counter()
    with torch.no_grad():
        X, _ = sampler.sample(dm, D, L, coord0, f, init, is_motif, generator=gens)
    return time.perf_counter() - t0, X


for rung, batches in LADDER:
    # The arm is not known until the rung is featurised (set_tune_matmul_for_atoms needs the atom
    # count), so the rung-level skip is gone: the per-batch check below carries it.
    pass
    specs = json.loads(pathlib.Path("perf/dsfix/fixtures/rfd3_%s.json" % rung).read_text())
    sid, sdict = next(iter(specs.items()))
    spec = InputSpecification.from_dict(sdict)
    t_feat = time.perf_counter()
    f = featurize(spec.input, spec)
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    tuned = set_tune_matmul_for_atoms(L)
    is_motif = f["is_motif_atom_with_fixed_coord"]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    print("[ladder] %s featurised L=%d atoms in %.1fs, tune_matmul=%s"
          % (rung, L, time.perf_counter() - t_feat, tuned), flush=True)
    for b in batches:
        if (rung, b, tuned) in done:
            print("[ladder] %s b=%d tune=%s cached" % (rung, b, tuned), flush=True)
            continue
        s1, s2 = RFD3Sampler(num_timesteps=N1), RFD3Sampler(num_timesteps=N2)
        wt, _ = timed(s1, b, L, coord0, f, init, is_motif, 1000)   # discarded cold rep
        print("[ladder] %s b=%d warmup %.2fs" % (rung, b, wt), flush=True)
        per_step, legs, bad = [], [], []
        for rep in range(NREP):
            order = [(s1, N1), (s2, N2)] if rep % 2 == 0 else [(s2, N2), (s1, N1)]
            leg = {}
            for smp, n in order:
                t, X = timed(smp, b, L, coord0, f, init, is_motif, 7 + 100 * rep + n)
                leg[n] = t
                if int(X.shape[1]) != L:
                    bad.append("rep%d N%d atoms %d != %d" % (rep, n, int(X.shape[1]), L))
                if not torch.isfinite(X).all():
                    bad.append("rep%d N%d non-finite coords" % (rep, n))
                if int(X.shape[0]) != b:
                    bad.append("rep%d N%d batch %d != %d" % (rep, n, int(X.shape[0]), b))
            legs.append({"t_N1": round(leg[N1], 4), "t_N2": round(leg[N2], 4)})
            per_step.append((leg[N2] - leg[N1]) / (N2 - N1))
            print("[ladder] %s b=%d rep%d N%d=%.2fs N%d=%.2fs per_step=%.1fms"
                  % (rung, b, rep, N1, leg[N1], N2, leg[N2], per_step[-1] * 1000), flush=True)
        med = statistics.median(per_step)
        rec = {"rung": rung, "batch": b, "atoms": L,
               "target_res": int(spec.contig.split(",")[0].split("-")[1]),
               "per_step_s_median": round(med, 5), "per_step_s_min": round(min(per_step), 5),
               "per_step_s_max": round(max(per_step), 5), "n": NREP, "legs": legs,
               "designs_per_s_at_200": round(b / (200 * med), 5),
               "N1": N1, "N2": N2, "warmup_s": round(wt, 3),
               "sanity_ok": not bad, "sanity_fail": bad,
               "loadavg": float(open("/proc/loadavg").read().split()[0]),
               "tune_matmul_env": os.environ.get("RFD3_TUNE_MATMUL"),
               "host": HOST, "card": CARD, "ttnn": TTNN, "tune_matmul": tuned}
        with OUT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("[ladder] DONE %s b=%d: %.1f ms/step, T=%s designs/s, sanity=%s"
              % (rung, b, med * 1000, rec["designs_per_s_at_200"], not bad), flush=True)
print("[ladder] ALL DONE", flush=True)

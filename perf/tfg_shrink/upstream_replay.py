"""Same recorded x0 -> guided x0: tt-bio's port (tt_bio.tfg) vs upstream OpenDDE v1.2.0 (opendde.tfg), CPU.

For each x0-hook step (100, 133, 157 by default) the recorded denoiser x0 [S, N_atom, 3] of a traced guided fold is fed
to both x0 passes: ours (tt_bio.tfg.epitope.guide_x0, as tt_bio.tfg.engine calls it) and upstream's
(opendde.tfg.epitope_guidance.guide_x0, as opendde.tfg.engine.TFGEngine.step calls it after _project), with upstream
forced onto the dense torch path (OPENDDE_RIGID_CORE=off, no Triton). Both passes are deterministic (refine: fixed-step
rigid descent; search: fixed 25 axes x 12 turns x 5 radii grid), so no RNG is involved. Prints per sample the max |dx|
between the two guided outputs and each side's Kabsch rotation of the movable group; with a difference, each stage
(refine, search, refine) is also replayed on the SAME input (upstream's previous-stage output) to find where they split.
QUICK=1: our default core only, no stage-isolated replay.
usage: python upstream_replay.py DIR/<n> [steps=100,133,157] [threads=8]
"""
import math
import os
import sys

os.environ["OPENDDE_RIGID_CORE"] = "off"  # dense path upstream
for k in ("OPENDDE_RIGID_CONTACT", "OPENDDE_RIGID_X0_START", "OPENDDE_RIGID_X0_EVERY", "OPENDDE_RIGID_X0_LAST"):
    os.environ.pop(k, None)  # upstream defaults: auto, 100, 3, 189 (= RigidSchedule defaults)
UP = os.environ.get("OPENDDE_SRC", "/home/moritz/tfg-src/OpenDDE")
sys.path.insert(0, UP)

import torch  # noqa: E402

from opendde.tfg import epitope_guidance as up_epi  # noqa: E402
from opendde.tfg import rigid_contact as up_rc  # noqa: E402
from tt_bio.tfg import epitope as our_epi  # noqa: E402
from tt_bio.tfg import rigid as our_rc  # noqa: E402
from tt_bio.tfg.rigid import RigidSchedule  # noqa: E402

p = sys.argv[1]
steps = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "100,133,157").split(",")]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 8)
d = torch.load(p + "_traj.pt", weights_only=False)
f = torch.load(p + "_feats.pt", weights_only=False)
x0s = d["x0"]
fixed, moving, _, _ = our_rc.contact_groups(x0s[0], f)
ufixed, umoving, _, _ = up_rc.contact_groups(x0s[0], f)
assert torch.equal(fixed, ufixed) and torch.equal(moving, umoving), "group split differs"
assert up_epi.rigid_mode(f) == our_epi.rigid_mode(f, True, RigidSchedule()) == "on"
print(f"upstream {UP}; dense core; movable atoms {len(moving)}, fixed {len(fixed)}, contacts "
      f"{f['user_distance_restraint_index'].shape[1]}; ours core={our_rc.resolve_core(None)}")


def angle(a, b):
    """Rotation angle (deg) of the best rigid fit of a onto b, both [n, 3]."""
    a = a.double() - a.double().mean(0)
    b = b.double() - b.double().mean(0)
    u, _, vh = torch.linalg.svd(a.T @ b)
    s = torch.ones(3, dtype=torch.float64)
    s[2] = torch.sign(torch.det(u @ vh))
    R = u @ torch.diag(s) @ vh
    return math.degrees(math.acos(max(-1.0, min(1.0, (torch.trace(R).item() - 1) / 2))))


def mdx(a, b):
    return (a - b).abs().flatten(1).amax(1)


up_stages = [lambda y: up_rc.refine_rigid_contact(y, f, iterations=40), lambda y: up_rc.search_rigid_contact(y, f),
             lambda y: up_rc.refine_rigid_contact(y, f, iterations=40)]


def our_stages(core):
    return [lambda y: our_rc.refine_rigid_contact(y, f, iterations=40, core=core),
            lambda y: our_rc.search_rigid_contact(y, f, core=core),
            lambda y: our_rc.refine_rigid_contact(y, f, iterations=40, core=core)]


names = ["refine", "search", "refine2"]
verdicts = []
for k in steps:
    if not (up_epi.x0_step_active(k) and our_rc.x0_step_active(k, RigidSchedule())):
        k = min((j for j in range(x0s.shape[0]) if up_epi.x0_step_active(j)), key=lambda j: abs(j - k))
    x0 = x0s[k].clone()
    gu = up_epi.guide_x0(x0.clone(), f, k)
    for core in (("auto",) if os.environ.get("QUICK") else ("auto", "off")):
        go = our_epi.guide_x0(x0.clone(), f, k, RigidSchedule(core=core))
        dx = mdx(go, gu)
        print(f"\nstep {k} t_hat={d['t_hat'][k]:.3f} ours core={core}: max|dx| all samples {dx.max().item():.2e} A")
        print(" s | max|dx| A | rot ours deg | rot upstream deg | moved ours/up")
        for s in range(x0.shape[0]):
            print(f" {s} | {dx[s].item():9.2e} | {angle(x0[s, moving], go[s, moving]):12.3f} | "
                  f"{angle(x0[s, moving], gu[s, moving]):16.3f} | "
                  f"{(go[s, moving] - x0[s, moving]).abs().max().item():6.2f}/{(gu[s, moving] - x0[s, moving]).abs().max().item():6.2f} A")
        if os.environ.get("QUICK"):
            verdicts.append((k, core, dx.max().item(), "not localised (QUICK)"))
            continue
        # per stage, both sides on the same input (upstream's previous-stage output)
        y = x0.float().clone()
        first = None
        cells = []
        for name, us, os_ in zip(names, up_stages, our_stages(core)):
            yu, yo = us(y.clone()), os_(y.clone())
            e = mdx(yo, yu)
            cells.append(f"{name} {e.max().item():.2e} (per sample {[round(v, 4) for v in e.tolist()]})")
            if first is None and e.max().item() >= 1e-3:
                first = name
            y = yu
        print("  stage-isolated max|dx| (same input to both):\n   " + "\n   ".join(cells))
        verdicts.append((k, core, dx.max().item(), first))

print("\nsummary (step, our core, max|dx| A, first diverging stage):")
for v in verdicts:
    print(" ", v)
bad = [v for v in verdicts if v[2] >= 1e-3]
print("VERDICT:", "EQUAL (max dx < 1e-3 A)" if not bad else
      "DIFFERENT; first diverges at " + ", ".join(f"step {v[0]} core={v[1]}: {v[3]}" for v in bad))

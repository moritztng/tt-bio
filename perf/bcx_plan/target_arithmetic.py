"""What the block's floor is worth: per step, per trajectory, per design campaign, per fleet.

Every input is either measured (floor screen, b2z2 roofs) or an upstream-published count. Nothing
here is a projection dressed as a target: each scenario names the lever that would produce it and
the arithmetic is the same in every row.

    python3 perf/bcx_plan/target_arithmetic.py
"""
import json

# --- measured, `perf/hallgrad/p2_floor_screen.json`, qb2 card 0, BH p300c, AICLK 1337-1350 MHz
A, B = 0.03967, 0.23204          # fwd+bwd, n=256, transition on
A_F, B_F = 0.00565, 0.08542      # forward only
BLOCKS = 52                      # 4 extra-MSA + 48 Evoformer pair blocks
STEPS = 125                      # BC2 gradient rounds per trajectory, settings.py:604
SEMIGREEDY = 15                  # forward-only mutate rounds
H200_STEP = 0.511                # s per gradient iteration, one H200 NVL, envelope 0.43-0.74
GO_BAR = 3.0                     # the previous campaign's bar: within 3x of the reference

# --- from `perf/bcx_plan/block_cost_model.py`, floors at the two measured roofs
FLOOR = json.load(open("perf/bcx_plan/block_cost_model.json"))
F_TOT = FLOOR["total_floor_s"]
F_FWD = FLOOR["by_phase"]["fwd"]["floor"]
F_MM = FLOOR["by_class"]["matmul"]["floor"]
F_MM_FWD = sum(r["floor_s"] for r in FLOOR["rows"]
               if r["class"] == "matmul" and r["op"].startswith("fwd"))

# --- the shape-invariant term, split out of the two measured points by the previous campaign
INVARIANT = 0.01847              # s per block; 7.96 % of b at n=256


def step(b, bf, a=A, af=A_F):
    """One BC2 gradient round: one stop-gradient recycle forward, one fwd+bwd (C2)."""
    grad = a + BLOCKS * b
    fwd = af + BLOCKS * bf
    return grad + fwd, grad, fwd


SCEN = [
    ("today, measured", B, B_F,
     "the floor screen as it stands"),
    ("AF2's own ReLU transition", B - 0.0070, B_F - 0.0029,
     "the harness runs a SwiGLU (24 N^2 c^2); AF2's pair transition is LN/linear/ReLU/linear "
     "(16 N^2 c^2). Scaled from the measured transition-on minus transition-off delta 0.02098 s "
     "by the 2/3 FLOPs ratio and one fewer big eltwise"),
    ("trace capture alone", B - INVARIANT, B_F - INVARIANT * (B_F / B),
     "removes ALL of the shape-invariant term. Capped by construction at 1.09x"),
    ("every op at its own measured roof", F_TOT, F_FWD,
     "each op at max(FLOPs/18.65 TFLOP/s, bytes/390.7 GB/s), intermediates still in DRAM"),
    ("+ fusion: only matmul bytes survive", F_MM, F_MM_FWD,
     "the non-matmul classes fused into their producers; matmul FLOPs are irreducible"),
]

print(f"{'scenario':38s} {'b (ms)':>8s} {'grad s':>8s} {'step s':>8s} {'traj min':>9s} "
      f"{'x H200':>7s} {'x bar':>6s}")
out = []
for name, b, bf, why in SCEN:
    t, grad, fwd = step(b, bf)
    traj = STEPS * t / 60.0
    print(f"{name:38s} {b*1e3:8.2f} {grad:8.3f} {t:8.3f} {traj:9.2f} "
          f"{t/H200_STEP:7.1f} {t/(GO_BAR*H200_STEP):6.2f}")
    out.append({"scenario": name, "b": b, "t_step": t, "t_grad": grad, "t_fwd": fwd,
                "traj_min": traj, "x_h200": t / H200_STEP, "why": why})

print(f"\nthe bar: {GO_BAR}x of {H200_STEP} s = {GO_BAR*H200_STEP:.3f} s per step")
print(f"today is {step(B, B_F)[0]/H200_STEP:.1f}x the reference; "
      f"the all-roofs floor is {step(F_TOT, F_FWD)[0]/H200_STEP:.1f}x; "
      f"the fused floor is {step(F_MM, F_MM_FWD)[0]/H200_STEP:.1f}x")

# --- the throughput axis. Trajectories are independent, so chips multiply cleanly.
BH_CHIPS = 8            # two QuietBoxes, four Blackhole processors each
WH_CHIPS = 128          # four Osaka Galaxies, 32 Wormhole chips each
WH_PENALTY = (1.72, 3.42)   # measured BH:WH ratios, DRAM roof 390.7:227.5 and tile pass 20.82:71.3
ACCEPT = 101 / 91       # upstream's published B200 PDL1 run: 101 accepted from 91 trajectories
DESIGNS_NEEDED = 18     # designs per problem a Track 1 team can put in the wet lab

print("\nthroughput, trajectories being independent")
for name, b, bf, _ in SCEN[:1] + SCEN[-2:]:
    t = step(b, bf)[0]
    traj_s = STEPS * t
    bh = BH_CHIPS * 3600 / traj_s
    print(f"  {name:38s} {traj_s/60:7.2f} min/traj   {bh:7.2f} traj/h on {BH_CHIPS} BH "
          f"({bh*ACCEPT:6.2f} designs/h)")
    for p in WH_PENALTY:
        wh = WH_CHIPS * 3600 / (traj_s * p)
        print(f"{'':40s} {'':7s}         {wh:7.2f} traj/h on {WH_CHIPS} WH at {p:.2f}x "
              f"({wh*ACCEPT:6.2f} designs/h)")
    need_h = DESIGNS_NEEDED / (bh * ACCEPT)
    print(f"{'':40s} {DESIGNS_NEEDED} designs on the {BH_CHIPS} BH chips: {need_h:.2f} h")

json.dump({"scenarios": out, "h200_step": H200_STEP, "go_bar_s": GO_BAR * H200_STEP,
           "bh_chips": BH_CHIPS, "wh_chips": WH_CHIPS, "accept_per_traj": ACCEPT},
          open("perf/bcx_plan/target_arithmetic.json", "w"), indent=1)

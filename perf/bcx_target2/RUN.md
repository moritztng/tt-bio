# The run

    cd /home/ttuser/.coworker/wt/bcx-target2
    TT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 /home/ttuser/bcx_e2e_venv/bin/python3 -u \
      perf/bcx_exact/traj_arm.py --exact off --seed 1 --trajectories 1 --resident 1 \
      --params /home/ttuser/bcx_e2e/af2_params \
      --settings perf/bcx_target2/lysozyme.json \
      --out /home/ttuser/bcx_target2_art/lyso_s1

qb1 card 2 (PCI 0000:42:00.0, class node 3), launched 2026-09-26T01:53:18Z at commit 12c39e510,
detached as pid 187751.

BindCraft 2 took the target on the first try. The banner reads
`target HEWL (target) | 129 residues | hotspots 35,52,62,101`, it drew a 111 aa binder, and the
design loop ran gradient steps on card at **AICLK 1350** sampled during, with 0 tracebacks.

The binder chain is bucketed before the complex is, so the card sees `ceil32(111) + 129 = 257`
rounded to **288 tokens**, not the 256 that adding the raw lengths gives. See `CEILING.md`. `traj_live.json` reads
`arm_held: true` with 0 exact-path entries, so the OFF arm is the OFF arm.

Host loadavg ran 27.9 to 39.1 on 32 cores carrying three other rows arms, so no per-round timing
from this run is a perf number. Whether it fires is load-insensitive, which is the question here.

## Reading the ending

    grep -a "design stage" /home/ttuser/bcx_target2_art/lyso_s1.log
    cat "/home/ttuser/bcx_target2_art/lyso_s1/1_Trajectories/!_Trajectories.csv"
    cat /home/ttuser/bcx_target2_art/lyso_s1/traj_stamp.json   # wall_seconds, aiclk_med, failed

The `terminated` column of the CSV names the stage a trajectory was rejected at, and an empty one
means it went the whole way. The comparable PD-L1 trajectory on this box spent 2523.8 s of design
before its anneal rejection, at the smaller bucket 224.

The card carries a standing block that predates this row and still stands; see
`/home/ttuser/.coworker/state/cardblock-qb1-2` and the `.bcx-target2` marker beside it. Do not
reset it: the BDF neighbour 0000:41 carries another rows live arm. A single SIGINT to 187751
stops this arm, never a pkill.

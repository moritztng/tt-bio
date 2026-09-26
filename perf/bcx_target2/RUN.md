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
rounded to **288 tokens**, not the 256 that adding the raw lengths gives. See `CEILING.md`.

The card was released when the holder exited; the standing block on it is untouched and stands.

## The ending

**Rejected at the anneal design stage on i_pTM, after 6119.7 s.** A verdict, not a traceback.

    rejected at anneal design stage  i_pTM=0.19  pLDDT=0.82  due to [i_pTM]
    campaign stopped: 1 trajectories ran and none were accepted, so the settings rather than
    the budget are what to change
    campaign done: 0 accepted design(s) after 1 trajectory, ranked by i_pDAE

`!_Trajectories.csv` carries `terminated=anneal`, `.campaign_state.json` carries
`{"terminated": {"anneal": 1}}`, and `traj_stamp.json` carries `failed: null`. The filter it
missed is `i_pTM >= 0.7`; it reached 0.19. One trajectory against a lysozyme active-site cleft
was never likely to bind, and the task was whether the loop runs, not whether it binds.

| stage | rounds | i_pTM | pLDDT | outcome |
|---|---|---|---|---|
| screen | 50 | 0.15 | 0.82 | passed |
| refine | 25 | 0.19 | 0.79 | passed |
| anneal | 45 | 0.19 | 0.82 | rejected, i_pTM |

`device_calls` reads `taped: 240, backward: 120, primal: 0`. 120 backward calls is exactly
50 + 25 + 45, one per round, so all three stages ran their full default round counts and the
rejection came at the end of anneal rather than early. Nothing was skipped.

All five shipped models were drawn: `model_1` 27, `model_2` 29, `model_3` 21, `model_4` 24,
`model_5` 20, 121 selections over 120 rounds.

**Clock.** AICLK median **1350 MHz** over **1220** samples taken during the run, max 1350, min
800. The 800 is an idle dip between stages, not a throttled step. `arm_held: true` with 0
exact-path entries in both counters, so the OFF arm was the OFF arm the whole way.

**No timing here is a perf number.** Host loadavg ran 27.9 to 48.0 on 32 cores carrying three
other rows arms, and the 6103.69 s design time in the CSV is that box, not this card. Whether the
loop fires is load-insensitive, which is what this row asked.

## The teardown abort, which is not a failure

At interpreter shutdown the process raised SIGABRT inside `tt::umd::Cluster::close_device()`:

    pthread_mutex_unlock failed for mutex CHIP_IN_USE_3_PCIe errno: 1
     --- tt::umd::RobustMutex::unlock()
     --- tt::umd::LocalChip::close_device()

This happens **after** everything is written. `finished_utc` is 03:35:18Z and `summary.csv`,
`!_Trajectories.csv`, `.campaign_state.json` and the per-round losses CSV are all complete and
consistent. The card came back clean: nothing holds `/dev/tenstorrent/3`, `tt_heartbeat` advanced
153514 to 153534 in 2 s, and `tt_aiclk` returned to the 800 idle reading.

Worth knowing because the backtrace looks alarming and the tempting response is a reset. On qb1
that is the dangerous act: the BDF neighbour 0000:41 is `TT_VISIBLE_DEVICES=1` and carries
another rows live arm. A SIGABRT in `close_device` at exit is a teardown abort, not a wedged
card. Read the heartbeat and the idle clock before touching anything.

## Reading it back

    grep -a "design stage\|campaign done" /home/ttuser/bcx_target2_art/lyso_s1.log
    cat "/home/ttuser/bcx_target2_art/lyso_s1/1_Trajectories/!_Trajectories.csv"
    cat /home/ttuser/bcx_target2_art/lyso_s1/summary.csv
    cat /home/ttuser/bcx_target2_art/lyso_s1/traj_stamp.json

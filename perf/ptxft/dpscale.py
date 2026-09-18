#!/usr/bin/env python3
"""Phase 5.3: data-parallel training across qb1's chips, measured rather than projected.

DATA PARALLELISM, NEVER TENSOR PARALLELISM -- standing direction, and here it is also the
only thing the shapes want: one 48-block step at 128 tokens is 13.3 s of device work
against a 24.1 MB gradient exchange, so the communication is 0.4 % of the step even done
naively over the host.

Every rank holds the whole adapter and differs only in which targets it sees. Ranks start
from the same seed, so their weights are identical at step 0; the all-reduce averages the
gradients before the step, so they stay identical afterwards. That invariant is ASSERTED,
not assumed: `--check-sync` compares every rank's masters at the end and requires them
bit-identical. Without it a DP run that silently diverged would still produce a
respectable-looking throughput number, which is the failure this arm exists to refuse.

The all-reduce is host-side through /dev/shm rather than through the device: the masters
and the Adam moments already live on the host (`moreh_adamw` cannot hold an fp32 master,
so the optimizer is host-side anyway), the gradient has to cross PCIe to reach them
whatever happens, and an on-device collective would move it to the device and back for
nothing.

  driver:  python3 perf/ptxft/dpscale.py --cards 1,3 --steps 12
  rank:    the driver spawns one child per card with RANK/WORLD/TT_VISIBLE_DEVICES set.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from perf.ptxft import dgdata as D           # noqa: E402

ART = os.path.expanduser("~/ptxft-art")
SHM = "/dev/shm/ptxft-dp"
TRAIN28 = ("1UBQ,1VII,2GB1,1PGB,1CRN,2CI2,1ENH,1BDD,1SHF,1CSP,1MJC,1TIT,2PTL,1BTA,1IGD,"
           "1E0L,5PTI,2TRX,1ARR,1ROP,1FKB,3CHY,1POH,1RIS,1TEN,1OPD,2ACY,1IMQ")


# ------------------------------------------------------------------ the collective

def allreduce(vec, rank, world, step, *, timeout=300.0):
    """Mean of `vec` across ranks. Returns (mean, seconds spent waiting + copying)."""
    if world == 1:
        return vec, 0.0
    t0 = time.perf_counter()
    tmp = os.path.join(SHM, f"s{step}_r{rank}.npy.part")
    done = os.path.join(SHM, f"s{step}_r{rank}.npy")
    # np.save appends .npy unless the handle is passed, which would defeat the
    # atomic rename below by writing to a third name.
    with open(tmp, "wb") as fh:
        np.save(fh, vec)
    os.replace(tmp, done)            # atomic, so a reader never sees a half file
    peers = [os.path.join(SHM, f"s{step}_r{r}.npy") for r in range(world)]
    while True:
        if all(os.path.exists(p) for p in peers):
            break
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"rank {rank} waited {timeout}s at step {step}")
        time.sleep(0.002)
    acc = np.zeros_like(vec)
    for p in peers:
        acc += np.load(p)
    acc /= world
    return acc, time.perf_counter() - t0


def flatten(params, opt):
    """Gradients as one fp32 vector in a fixed order, plus the order itself."""
    from tt_bio import finetune as ft
    names = sorted(params)
    parts = []
    for n in names:
        g = params[n].grad
        shape = opt.master[n].shape
        parts.append(np.zeros(shape, np.float32).ravel() if g is None
                     else ft.to_host(g).astype(np.float32).reshape(shape).ravel())
    return np.concatenate(parts), names


def unflatten(vec, names, params, opt):
    from tt_bio import finetune as ft
    i = 0
    for n in names:
        shape = opt.master[n].shape
        k = int(np.prod(shape))
        arr = vec[i:i + k].reshape(shape)
        i += k
        params[n].grad = ft.to_device(arr, params[n].value.device(),
                                      dtype=params[n].value.dtype)


# ------------------------------------------------------------------ one rank

def run_rank(a):
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import finetune as ft
    from perf.ptxft import tape_block as TB
    from perf.ptxft.train_distogram import build_stack, forward_loss, load_target
    from perf.clocksample import during

    rank, world = a.dp_rank, a.world
    names = [t for t in a.train.split(",") if t]
    shard = names[rank::world]               # disjoint, deterministic
    with during() as clk:
        device = get_device()
        blocks, head, params, _l = build_stack(a, device, adapt_all=True)
        opt = ft.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
        tg = {p: load_target(p) for p in shard}
        rows = []
        for step in range(1, a.steps + 1):
            p = shard[(step - 1) % len(shard)]
            t = tg[p]
            n = int(t["target"].shape[0])
            t0 = time.perf_counter()
            loss, out, _lg, seed, _pr = forward_loss(
                blocks, head, t["z"], t["target"], t["pair_mask"], device,
                checkpointed=True, n_real=n, want_grad=True)
            full = np.zeros((out.value.shape[-3], out.value.shape[-2], D.N_BINS), np.float32)
            full[:n, :n, :] = TB.symmetrize_bins(seed)
            opt.zero_grad()
            out.backward(seed=ft.to_device(full, device, dtype=ttnn.bfloat16))
            t_bwd = time.perf_counter()
            vec, order = flatten(params, opt)
            vec, t_comm = allreduce(vec, rank, world, step)
            unflatten(vec, order, params, opt)
            opt.step()
            dt = time.perf_counter() - t0
            rows.append({"step": step, "target": p, "tokens": n, "loss": loss,
                         "s": dt, "compute_s": t_bwd - t0, "comm_s": t_comm})
            print(f"r{rank} step {step:>3} {p:<6} n={n:<4} loss {loss:>8.5f} "
                  f"{dt:>6.2f} s (comm {t_comm:>5.3f})", flush=True)
        payload = {"rank": rank, "world": world, "card": os.environ.get(
            "TT_VISIBLE_DEVICES"), "rows": rows, "bytes": int(vec.nbytes),
            "clock": clk.summary()}
        # The sync invariant, checked rather than trusted.
        m = np.concatenate([opt.master[n].ravel() for n in sorted(params)])
        payload["master_sha"] = __import__("hashlib").sha256(m.tobytes()).hexdigest()
    with open(os.path.join(SHM, f"result_r{rank}.json"), "w") as fh:
        json.dump(payload, fh)
    return 0


# ------------------------------------------------------------------ the driver

def drive(a):
    cards = [c for c in a.cards.split(",") if c]
    world = len(cards)
    os.makedirs(SHM, exist_ok=True)
    for f in os.listdir(SHM):
        os.remove(os.path.join(SHM, f))
    env0 = dict(os.environ)
    env0["TT_BIO_LEASE_CARDS"] = ",".join(cards)
    env0["TT_BIO_LEASE_HOLDER"] = "worker:ptxft-build"
    procs = []
    t0 = time.perf_counter()
    for r, c in enumerate(cards):
        env = dict(env0)
        env["TT_VISIBLE_DEVICES"] = c
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--dp-rank", str(r),
               "--world", str(world), "--steps", str(a.steps), "--lr", str(a.lr),
               "--rank-mode", "--train", a.train, "--depth", str(a.depth),
               "--rank", str(a.rank), "--alpha", str(a.alpha), "--seed", str(a.seed)]
        with open(os.path.join(SHM, f"rank{r}.log"), "w") as lg:
            procs.append(subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    if any(codes):
        for r, c in enumerate(codes):
            if c:
                print(f"rank {r} exited {c}; tail of its log:")
                print(open(os.path.join(SHM, f"rank{r}.log")).read()[-2000:])
        return 1
    res = [json.load(open(os.path.join(SHM, f"result_r{r}.json"))) for r in range(world)]
    return report(res, wall, a)


def report(res, wall, a):
    world = len(res)
    # Step 1 carries the kernel compile; throughput is a steady-state number.
    per_rank = []
    for r in res:
        rows = r["rows"][1:] if len(r["rows"]) > 1 else r["rows"]
        tok = sum(x["tokens"] for x in rows)
        sec = sum(x["s"] for x in rows)
        comm = sum(x["comm_s"] for x in rows)
        per_rank.append({"rank": r["rank"], "card": r["card"], "steps": len(rows),
                         "tokens": tok, "s": sec, "comm_s": comm,
                         "tok_s": tok / sec, "median_s": float(np.median(
                             [x["s"] for x in rows])),
                         "clock": r["clock"].get(0) or r["clock"].get("0")})
    print()
    print(f"# --dp: {world} rank(s), {a.steps} steps each (step 1 dropped, it carries the "
          f"kernel compile), {res[0]['bytes'] / 1e6:.1f} MB all-reduced per rank per step")
    print(f"{'rank':>4} {'card':>5} {'steps':>6} {'tokens':>8} {'s':>8} {'comm s':>8} "
          f"{'tok/s':>8} {'median s':>9}  clock")
    for p in per_rank:
        c = p["clock"] or {}
        print(f"{p['rank']:>4} {str(p['card']):>5} {p['steps']:>6} {p['tokens']:>8} "
              f"{p['s']:>8.2f} {p['comm_s']:>8.2f} {p['tok_s']:>8.2f} {p['median_s']:>9.2f}"
              f"  {c.get('median', '?')} MHz over {c.get('n', 0)}")
    agg = sum(p["tok_s"] for p in per_rank)
    comm_frac = sum(p["comm_s"] for p in per_rank) / sum(p["s"] for p in per_rank)
    print()
    print(f"AGGREGATE: {agg:.2f} tokens/s over {world} chip(s), wall {wall:.1f} s, "
          f"all-reduce {100 * comm_frac:.2f} % of step time")
    shas = {r["master_sha"] for r in res}
    ok = len(shas) == 1
    print(f"SYNC: {'PASS' if ok else 'FAIL'} -- {len(shas)} distinct master-weight hashes "
          f"across {world} rank(s); data parallelism requires exactly 1")
    out = {"per_rank": per_rank, "aggregate_tok_s": agg, "world": world,
           "wall_s": wall, "comm_frac": comm_frac, "sync_ok": ok,
           "bytes": res[0]["bytes"]}
    with open(os.path.join(ART, f"dp_world{world}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/.boltz/protenix-v2.pt"))
    ap.add_argument("--cards", default="1,3")
    ap.add_argument("--train", default=TRAIN28)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--depth", type=int, default=48)
    ap.add_argument("--dp-rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--rank-mode", action="store_true")
    # build_stack's knobs, named as train_distogram names them so one configuration
    # description covers both arms. --rank is LoRA's rank, --dp-rank is the DP rank.
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--q-chunk", type=int, default=None)
    ap.add_argument("--freeze-head", action="store_true")
    a = ap.parse_args()
    return run_rank(a) if a.rank_mode else drive(a)


if __name__ == "__main__":
    sys.exit(main())

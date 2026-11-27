#!/usr/bin/env python3
"""r4t-bwd B1 grade across all 9 batches (batch00..08) on Blackhole:
  1. D37 loss-parity against upstream ref + committed bf16 floor (`perf/r4t/floor/up_bf16.json`)
     for: baseline (`serving`+`tape`), `b1_suspend` (`serve`+`suspend`), `b1_reuse_serve` (`serve`+`reuse`),
     `b1_reuse_train` (`serve_train`+`reuse`), and `b1_broken` (`serve`+`reuse`+negated cycle-0 state, MUST FAIL).
  2. Winning-loss and final-cycle input match vs baseline (<= 1e-2).
  3. Inference grad-off A/A byte-identical check on `Trunk.__call__` (default `prev=None, return_penultimate=False`).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "perf" / "r4t")]


def sample_aiclk(card: int, stop: threading.Event, out: list):
    path = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"
    while not stop.is_set():
        try:
            out.append(int(open(path).read().split()[0]))
        except Exception:
            pass
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", required=True, help="comma-separated batch00..08.npz paths")
    ap.add_argument("--floor", type=Path, default=REPO / "perf/r4t/floor/up_bf16.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import numpy as np
    import torch
    import ttnn
    from loss_parity_reference import load_batch
    from tt_bio import autograd as ag, tenstorrent as T
    from tt_bio.rfd4 import engine as E, network as N, trunk as TR
    from tt_bio.train import parity as PR, rfd4_objective as RO
    from tt_bio.train.folds import Folds

    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    sha = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], capture_output=True, text=True).stdout.strip()
    clk, stop = [], threading.Event()
    thr = threading.Thread(target=sample_aiclk, args=(card, stop, clk), daemon=True)
    thr.start()

    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config)
    batches = [(os.path.basename(f), f) + load_batch(f) for f in a.batches.split(",")]

    def put(t, dtype=ttnn.float32):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def flat(o):
        v = lambda m: next(iter(o[m].values()))  # noqa: E731
        out = {"v_bb_ca": v("bb_ca"), "v_local_latents": v("local_latents")}
        if "sequence" in o:
            out["sequence_logits"] = o["sequence"]["logits"]
        if "insertion" in o:
            out["insertion_logits"] = o["insertion"]["logits"]
        return out

    def npy(d):
        return {k: npy(v) for k, v in d.items()} if isinstance(d, dict) else (
            d.detach().numpy() if torch.is_tensor(d) else d)

    def take(d, i, K):
        if isinstance(d, dict):
            return {k: take(v, i, K) for k, v in d.items()}
        return d[i:i + 1] if torch.is_tensor(d) and d.dim() and d.shape[0] == K else d

    def conds_for(b, depth):
        cs = [glue.conditioning(b, None)]
        for cyc in range(1, depth):
            cs.append(glue.conditioning(b, depth - cyc - 1) if glue.condition_on_recycle_remaining else cs[0])
        return cs

    # ---- Inference grad-off A/A byte-identical check ----
    trunk_inf = E.load_trunk(glue, state, cfg, training=False)
    nm0, _, bt0, m0, _, _ = batches[0]
    bw0 = take(bt0, 0, int(bt0["padding_mask"].shape[0]))
    with torch.no_grad():
        s0_i, z0_i = glue.initial_states(bw0)
        cs0_i = conds_for(bw0, max(1, int(m0["depth"])))
    mk0 = TR.trunk_masks(bw0["padding_mask"].float(), dev, dtype=trunk_inf.dtype, residual_dtype=trunk_inf.residual_dtype)
    si0, zi0 = put(s0_i), put(z0_i, trunk_inf.pair_dtype(mk0))
    cs0_tt = [put(c, trunk_inf.dtype) for c in cs0_i]
    s_aa1, z_aa1 = trunk_inf(si0, zi0, cs0_tt, mk0)
    s_aa2, z_aa2 = trunk_inf(si0, zi0, cs0_tt, mk0, prev=None, return_penultimate=False)
    t1 = ttnn.to_torch(s_aa1).float().numpy()
    t2 = ttnn.to_torch(s_aa2).float().numpy()
    inf_aa = {
        "bit_exact": bool(np.array_equal(t1, t2)),
        "digest1": hashlib.sha256(t1.tobytes()).hexdigest()[:16],
        "digest2": hashlib.sha256(t2.tobytes()).hexdigest()[:16],
    }
    for t in (si0, zi0, *cs0_tt, s_aa1, z_aa1, s_aa2, z_aa2):
        ttnn.deallocate(t)
    del trunk_inf
    print("INFERENCE A/A:", inf_aa, flush=True)

    # ---- Training trunk (attn_fp32=on, matching production PERF/G3 recipe) ----
    ag.install()
    folds = Folds(state)
    with folds.recording() as sd:
        trunk = E.load_trunk(glue, sd, cfg, training=True, attn_fp32=True)
    for _p, _o, _k, t in T.walk_device_weights(trunk):
        ag.parameter(t)
    levers = trunk._levers

    @contextlib.contextmanager
    def cand_scope(mode):
        if mode in ("training", "serve_train"):
            yield
            return
        mods = [trunk] + list(trunk.pair_updates.values())
        saved = [(m, m.pair_residual_dtype if m is trunk else m.residual_dtype) for m in mods]
        saved_levers = trunk._levers
        cand_levs = levers | {"pair_bf16z", "dit_sigf32", "tok_recycle_mm"}
        trunk.pair_residual_dtype = ttnn.bfloat16
        for m in mods[1:]:
            m.residual_dtype = ttnn.bfloat16
        if mode == "serve":
            trunk._levers = cand_levs
        try:
            with T.lever_scope(cand_levs):
                yield
        finally:
            trunk._levers = saved_levers
            for m, d in saved:
                setattr(m, "pair_residual_dtype" if m is trunk else "residual_dtype", d)

    def slice_win(t, win):
        sh = [int(d) for d in t.shape]
        return ttnn.slice(t, [win] + [0] * (len(sh) - 1), [win + 1] + sh[1:])

    def eval_arm(cand_mode: str, win_pre: str, brk: bool = False):
        ours_rows, ref_rows, pairs, extras = [], [], [], []
        for nm, fpath, bt, meta, _rec, rl in batches:
            K = int(bt["padding_mask"].shape[0])
            depth = max(1, int(meta["depth"]))
            mk = TR.trunk_masks(bt["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
            with torch.no_grad():
                ck = conds_for(bt, depth)
                s0, z0 = glue.initial_states(bt)
            want_penult = win_pre == "reuse" and depth > 1
            penult_k = (None, None)
            if cand_mode in ("serve", "serve_train"):
                with ag.suspended(), cand_scope(cand_mode):
                    s0_t, z0_t = put(s0), put(z0, trunk.pair_dtype(mk))
                    ck_t = [put(c, trunk.dtype) for c in ck]
                    if want_penult:
                        sk, zk, penult_k = trunk(s0_t, z0_t, ck_t, mk, return_penultimate=True)
                    else:
                        sk, zk = trunk(s0_t, z0_t, ck_t, mk)
                    sh = ttnn.to_torch(sk).float().reshape(s0.shape)
                    for t in (s0_t, z0_t, *ck_t, sk, zk):
                        ttnn.deallocate(t)
            else:
                with T.lever_scope(levers), ag.tape(), ag.no_grad(), cand_scope(cand_mode):
                    s0_t, z0_t = put(s0), put(z0, trunk.pair_dtype(mk))
                    ck_t = [put(c, trunk.dtype) for c in ck]
                    if want_penult:
                        sk, zk, penult_k = trunk(s0_t, z0_t, ck_t, mk, return_penultimate=True)
                    else:
                        sk, zk = trunk(s0_t, z0_t, ck_t, mk)
                    sh = ttnn.to_torch(getattr(sk, "value", sk)).float().reshape(s0.shape)
                    for t in (s0_t, z0_t, *ck_t, sk, zk):
                        ttnn.deallocate(getattr(t, "value", t))
                penult_k = tuple(getattr(t, "value", t) if t is not None else None for t in penult_k)

            with torch.no_grad():
                ok = {k: v.double().numpy() for k, v in flat(glue.heads(sh, bt)).items()}
            _t, bd, _s = RO.rfd4_loss({**npy(bt), "exploration_k": K}, ok, RO.WEIGHTS)
            win = int(bd["winners"][0])
            bw = take(bt, win, K)
            m1 = TR.trunk_masks(bw["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
            with torch.no_grad():
                cs = conds_for(bw, depth)
                s_i, z_i = glue.initial_states(bw)

            prev = None
            if want_penult and penult_k[0] is not None:
                s_prev, z_prev = slice_win(penult_k[0], win), slice_win(penult_k[1], win)
                ttnn.deallocate(penult_k[0])
                ttnn.deallocate(penult_k[1])
                if brk:
                    bad = ttnn.multiply(s_prev, -1.0)
                    ttnn.deallocate(s_prev)
                    s_prev = bad
                prev = (s_prev, z_prev)
            elif win_pre == "suspend" and depth > 1:
                with ag.suspended(), cand_scope(cand_mode):
                    sp_in, zp_in = put(s_i), put(z_i, trunk.pair_dtype(m1))
                    cp_in = [put(c, trunk.dtype) for c in cs[:-1]]
                    s_prev, z_prev = trunk(sp_in, zp_in, cp_in, m1)
                    for t in (sp_in, zp_in, *cp_in):
                        ttnn.deallocate(t)
                if brk:
                    bad = ttnn.multiply(s_prev, -1.0)
                    ttnn.deallocate(s_prev)
                    s_prev = bad
                prev = (s_prev, z_prev)

            # Final cycle (taped, identical to training win_trunk)
            with T.lever_scope(levers), ag.tape(), ag.no_grad():
                si_t, zi_t = put(s_i), put(z_i, trunk.pair_dtype(m1))
                if prev is not None:
                    c_last = put(cs[-1], trunk.dtype)
                    s_out, z_out = trunk(si_t, zi_t, [c_last], m1, prev=prev)
                    c_used = [c_last]
                else:
                    c_used = [put(c, trunk.dtype) for c in cs]
                    s_out, z_out = trunk(si_t, zi_t, c_used, m1)
                sh_w = ttnn.to_torch(getattr(s_out, "value", s_out)).float().reshape(s_i.shape)
                for t in (si_t, zi_t, *c_used, s_out, z_out):
                    ttnn.deallocate(getattr(t, "value", t))
            if prev is not None:
                for t in prev:
                    ttnn.deallocate(t)

            with torch.no_grad():
                ow = flat(glue.heads(sh_w, bw))
                total, bdw, _ = RO.rfd4_loss({**npy(bw), "exploration_k": 1},
                                             {k: v.detach().double().numpy() for k, v in ow.items()}, RO.WEIGHTS)
            terms = {k: float(v["value"]) for k, v in bdw.items() if isinstance(v, dict) and "value" in v}
            terms["total"] = sum(terms.values())
            ref_row = {k: float(rl[k][win]) for k in terms if k in rl and win < len(rl[k])}
            ref_row["total"] = sum(ref_row.values())
            ours_rows.append(terms)
            ref_rows.append(ref_row)
            pairs.append({"batch": nm, "winner": win, "upstream_winner": int(meta["winners"][0]), "depth": depth})
            extras.append({"batch": nm, "winner": win, "total": terms["total"], "bb_ca": terms.get("bb_ca"),
                           "local_latents": terms.get("local_latents")})

        # Grade against committed D37 floor
        fdoc = json.loads(a.floor.read_text())
        by_b = {Path(str(r["batch"])).stem: r for r in fdoc["rows"]}
        floor_col = [by_b[Path(p["batch"]).stem]["up_bf16"] for p in pairs]
        # Align ref_rows to floor file's exact ref (for winner=upstream_winner batches)
        rep = PR.loss_parity(ours_rows, [by_b[Path(p["batch"]).stem]["ref"] for p in pairs], "bfloat16", floor_col)
        return {
            "cand_mode": cand_mode, "win_pre": win_pre, "break": brk,
            "passed": bool(rep.passed), "report": str(rep),
            "winners": [p["winner"] for p in pairs],
            "upstream_winners": [p["upstream_winner"] for p in pairs],
            "batches": extras,
        }

    arms = {}
    for name, cm, wp, brk in (
        ("baseline", "serving", "tape", False),
        ("b1_suspend", "serve", "suspend", False),
        ("b1_reuse_serve", "serve", "reuse", False),
        ("b1_reuse_train", "serve_train", "reuse", False),
        ("b1_broken", "serve", "reuse", True),
    ):
        t0 = time.time()
        r = eval_arm(cm, wp, brk)
        r["wall_s"] = round(time.time() - t0, 2)
        arms[name] = r
        print(f"ARM {name:16s}: passed={r['passed']} winners={r['winners']} wall={r['wall_s']}s\n  {r['report']}", flush=True)

    # Compare each B1 arm's winning total loss vs baseline
    base_tot = [b["total"] for b in arms["baseline"]["batches"]]
    for name, r in arms.items():
        if name == "baseline":
            continue
        diffs = [abs(b["total"] - bt) / max(abs(bt), 1e-12) for b, bt in zip(r["batches"], base_tot)]
        r["max_rel_vs_baseline_total"] = round(max(diffs), 6)
        r["winner_match_vs_baseline"] = f"{sum(a == b for a, b in zip(r['winners'], arms['baseline']['winners']))}/{len(base_tot)}"
        print(f"  vs baseline {name:16s}: max_rel_total={r['max_rel_vs_baseline_total']:.6f} winners={r['winner_match_vs_baseline']}", flush=True)

    # Direct final-cycle input (s_prev, z_prev) match vs taped baseline on batch00
    nm0, _, bt0, m0, _, _ = batches[0]
    K0 = int(bt0["padding_mask"].shape[0])
    win0 = arms["baseline"]["winners"][0]
    bw0 = take(bt0, win0, K0)
    m1_0 = TR.trunk_masks(bw0["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
    with torch.no_grad():
        cs0 = conds_for(bw0, max(1, int(m0["depth"])))
        si0_h, zi0_h = glue.initial_states(bw0)
    with T.lever_scope(levers), ag.tape(), ag.no_grad():
        si0_t, zi0_t = put(si0_h), put(zi0_h, trunk.pair_dtype(m1_0))
        c0_t = [put(cs0[0], trunk.dtype)]
        sb0, zb0 = trunk(si0_t, zi0_t, c0_t, m1_0)
        s_ref = ttnn.to_torch(getattr(sb0, "value", sb0)).double().reshape(si0_h.shape)
        z_ref = ttnn.to_torch(getattr(zb0, "value", zb0)).double().reshape(zi0_h.shape)
        for t in (si0_t, zi0_t, *c0_t, sb0, zb0):
            ttnn.deallocate(getattr(t, "value", t))
    prev_cmp = {}
    for nm_c, cm in (("serve", "serve"), ("serve_train", "serve_train")):
        mk_0 = TR.trunk_masks(bt0["padding_mask"].float(), dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
        with torch.no_grad():
            ck0 = conds_for(bt0, max(1, int(m0["depth"])))
            s0k, z0k = glue.initial_states(bt0)
        with ag.suspended(), cand_scope(cm):
            sk_in, zk_in = put(s0k), put(z0k, trunk.pair_dtype(mk_0))
            ck_in = [put(c, trunk.dtype) for c in ck0]
            sk_o, zk_o, pk = trunk(sk_in, zk_in, ck_in, mk_0, return_penultimate=True)
            sp = slice_win(pk[0], win0)
            zp = slice_win(pk[1], win0)
            s_got = ttnn.to_torch(sp).double().reshape(si0_h.shape)
            z_got = ttnn.to_torch(zp).double().reshape(zi0_h.shape)
            for t in (sk_in, zk_in, *ck_in, sk_o, zk_o, pk[0], pk[1], sp, zp):
                ttnn.deallocate(t)
        prev_cmp[nm_c] = {
            "s_rel_l2": round(float((s_got - s_ref).norm() / s_ref.norm()), 6),
            "z_rel_l2": round(float((z_got - z_ref).norm() / z_ref.norm()), 6),
        }
        print(f"FINAL_CYCLE_INPUT {nm_c}: {prev_cmp[nm_c]}", flush=True)

    stop.set()
    thr.join(timeout=2)
    out = {
        "host": os.uname().nodename,
        "chip": card,
        "sha": sha,
        "aiclk": [min(clk or [0]), max(clk or [0])],
        "aiclk_med": sorted(clk)[len(clk) // 2] if clk else 0,
        "inference_aa": inf_aa,
        "final_cycle_input": prev_cmp,
        "arms": arms,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"DONE grade_b1 aiclk={out['aiclk']} med={out['aiclk_med']}", flush=True)


if __name__ == "__main__":
    main()

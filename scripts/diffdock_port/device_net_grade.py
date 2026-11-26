"""Grade the score network with every convolution on the card, on a real complex.

    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEASE_CARDS=<chip> python3 \\
        scripts/diffdock_port/device_net_grade.py --fixture 8G43_ZU6.pt --out net.json

The fixture is ``score_model_grade.py --save-fixture``: the complex graph, the ESM-2 block, the
step inputs replayed from ``dfd-ref``'s captured trajectory and upstream's own scores. All 15
convolutions (3 receptor, 3 ligand, 9 main, the centre head's and the torsion head's) run on
the card through :func:`tt_bio.diffdock.device.convs_on_device`; the embedders, the graph
builders and the head MLPs stay on the host for this grade. Three comparisons per step:

* card vs the same network on the host in float32 (the device arithmetic, nothing else);
* card vs upstream's captured scores, beside host-fp32 vs upstream (the floor the port has);
* every convolution vs the float64 host layer on the inputs it actually got.

AICLK is sampled during the run and reported with the timing.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import torch

CKPT = Path(os.environ.get("DFD_CKPT", str(Path.home() / "scratch/dfd/ckpt/extracted")))


def grade(got, ref):
    got, ref = got.double().flatten(), ref.double().flatten()
    if ref.numel() == 0:
        return {}
    return dict(pcc=float(torch.corrcoef(torch.stack([got, ref]))[0, 1]) if ref.numel() > 1
                else None,
                max_rel_err=float((got - ref).abs().max() / ref.abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import time
    from tt_bio.diffdock import config as C
    from tt_bio.diffdock import device as ddev
    from tt_bio.diffdock.capture import clock_summary, sample_aiclk
    from tt_bio.diffdock.score_model import HostScoreNetwork, load_released
    from tt_bio.tenstorrent import get_device

    fx = torch.load(args.fixture, weights_only=False)
    cfg = C.load(str(CKPT / "score_model"), old_model=False)
    sd = torch.load(CKPT / "score_model/best_ema_inference_epoch_model.pt", map_location="cpu",
                    weights_only=False)
    host = HostScoreNetwork(load_released(sd, cfg), lm_embeddings=fx["emb"])
    card_model = load_released(sd, cfg)

    dev = get_device()
    convs = ddev.convs_on_device(card_model, dev, grade=True)
    card = HostScoreNetwork(card_model, lm_embeddings=fx["emb"])
    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample_aiclk, daemon=True,
                          args=(stop, samples, os.environ.get("TT_VISIBLE_DEVICES", "0")))
    th.start()

    h_score, c_score = host.score_fn(fx["graph"]), card.score_fn(fx["graph"])
    out = {"complex": fx["complex"], "convs_on_device": len(convs), "steps": {}}
    for k, step in sorted(fx["steps"].items()):
        with torch.no_grad():
            h = dict(zip(("tr", "rot", "tor"), h_score(step)))
            for m in convs.values():
                m.log.clear()
            t0 = time.perf_counter()
            c = dict(zip(("tr", "rot", "tor"), c_score(step)))
            secs = time.perf_counter() - t0
        want = fx["want"][k]
        row = {"seconds_untraced_with_grading": secs, "poses": int(step.lig_pos.shape[0])}
        for n in ("tr", "rot", "tor"):
            w = want[n].reshape(-1)
            row[n] = {"card_vs_host_fp32": grade(c[n].reshape(-1), h[n].reshape(-1)),
                      "card_vs_upstream": grade(c[n].reshape(-1), w),
                      "host_fp32_vs_upstream": grade(h[n].reshape(-1), w)}
        worst = {}
        for name, m in convs.items():
            pc = [e["pcc"] for e in m.log if e["pcc"] is not None]
            worst[name] = dict(calls=len(m.log), min_pcc=min(pc) if pc else None,
                               max_rel_err=max(e["max_rel_err"] for e in m.log) if m.log
                               else None)
        row["convs"] = worst
        out["steps"][k] = row
        print(f"step {k}: " + "  ".join(
            f"{n} card/host pcc {row[n]['card_vs_host_fp32'].get('pcc')} "
            f"card/up pcc {row[n]['card_vs_upstream'].get('pcc')}"
            for n in ("tr", "rot", "tor")), flush=True)
    stop.set()
    th.join(timeout=30)
    out.update(chip=os.environ.get("TT_VISIBLE_DEVICES"), **clock_summary(samples))
    print(json.dumps({k: v for k, v in out.items() if k != "steps"}))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

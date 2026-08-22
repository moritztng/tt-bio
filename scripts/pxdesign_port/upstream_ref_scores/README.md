# Upstream reference scores, PD-L1 anchor

Produced by `upstream_ref.py --stage traj` (the real upstream `sample_diffusion`, CPU, protenix
0.5.5, the release `pxdesign_v0.1.0` checkpoint) on the same captured input the device path uses,
then scored by `design_e2e.py --score_coords` so the reference and the port go through one
implementation of each metric.

| file | sampler settings | target reproduction RMSD | closest binder-target atom | contacts < 5 A |
|---|---|---|---|---|
| `pdl1_s0.json` | eta piecewise_65 1.0->2.5, gamma 1.0 / 0.01 (`configs_base.py`) | 8.34 / 0.69 A | 42.6 / 31.1 A | 0 / 0 |
| `pdl1_eta25.json` | eta const 2.5, gamma 1.0 / 0.01 (**what the CLI runs**) | 0.62 / 0.63 A | 41.8 / 22.3 A | 0 / 0 |
| `pdl1_eta25_gv2.json` | eta const 2.5, gamma 0.8 / 1.0 (Protenix-v2's) | 0.45 / 0.49 A | 23.3 / 25.2 A | 0 / 0 |

`parity_vs_upstream.json` is `upstream_parity.py --stage denoise`: the device port against the CPU
reference on identical inputs. `z_trunk` bit-exact, `s_inputs` PCC 0.99999826, the denoise net PCC
0.9991-0.9997 at five noise levels off the 400-step schedule.

Two things follow. The eta the port shipped was the one `configs_base.py` declares and not the one
a run uses, and fixing it removes the 8.34 A outlier from the first row. And **upstream itself does
not dock the binder on this anchor** in any of the three settings, so the undocked binder is not a
port defect. The per-step frames show where it goes: the binder sits on the target surface (closest
atom 1-6 A) until sigma ~25 around step 250, then is ejected to 20-60 A by step 325, in every cell.

The trajectories themselves are ~45 MB each and are not committed; regenerate with

    ~/protenix_ref_venv/bin/python scripts/pxdesign_port/upstream_ref.py --stage traj \
        --n_step 400 --n_sample 2 --seed 0 --eta_type const --eta_min 2.5 --eta_max 2.5 \
        --out /tmp/ref_traj_eta25.pt

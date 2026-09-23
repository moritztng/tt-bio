#!/usr/bin/env python3
"""Write the lane plans (plans/*.jsonl) and the screen shards. Re-run after changing a plan.

  size_nesso1   CEILING + LIGANDS for Nesso-1: 512/1024/1536 with the small drug, sirolimus and
                cobalamin at 512 and 1536, then 1664/1792/2048 to find what fails past the floor.
  size_boltz2   the same grid through `predict --model boltz2` at the shipped affinity settings
                (200 steps, 5 samples, mps 1).
  samples       --diffusion_samples_affinity 5/10/25 at 512 + small drug, timed at 200 steps,
                then a DRAM census of each at 20 steps (the census drains the pipeline, so its
                runs are memory-only).
  fix_nesso1    the size_nesso1 grid again after 29b285fb7 (trunk off the cross-chain mask path),
                the refused 1536 ligand rungs first, then 2048 with each ligand and
                2560/3072 to find the new wall; tags nf_*.
  screen_*      the 100-ligand screens. Nesso-1 YSK4 in two shards on two chips, Nesso-1 LCK on
                one, Boltz-2 LCK as one `predict` over the directory on two chips, which is how
                the shipped CLI spreads a screen across cards.
"""
import json
import pathlib
import shutil

HERE = pathlib.Path(__file__).resolve().parent
IN = "perf/mgx_affinity/inputs"
P = HERE / "plans"


def size(n, lig):
    return f"{IN}/size/cdk2_{n}_{lig}.yaml"


GRID = [(512, "small"), (1024, "small"), (1536, "small"), (1536, "rap"), (1536, "b12"),
        (512, "rap"), (512, "b12"), (1664, "small"), (1792, "small"), (2048, "small")]


def write(name, jobs):
    P.mkdir(exist_ok=True)
    (P / f"{name}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))


def main():
    write("size_nesso1", [{"tag": f"n_{n}_{lig}", "surface": "nesso1", "input": size(n, lig)}
                          for n, lig in GRID]
          + [{"tag": "n_screen_lck", "surface": "nesso1", "input": f"{IN}/screen/lck"}])
    write("size_boltz2", [{"tag": f"b_{n}_{lig}", "surface": "boltz2", "input": size(n, lig)}
                          for n, lig in GRID if n <= 1792])
    write("samples",
          [{"tag": f"s_512_small_a{k}", "surface": "boltz2", "input": size(512, "small"),
            "args": ["--diffusion_samples_affinity", str(k)]} for k in (10, 25)]
          + [{"tag": f"m_512_small_a{k}", "surface": "boltz2", "input": size(512, "small"),
              "census": True, "args": ["--diffusion_samples_affinity", str(k),
                                       "--sampling_steps", "20", "--sampling_steps_affinity", "20"]}
             for k in (5, 10, 25)]
          + [{"tag": "m_1536_b12_a5", "surface": "boltz2", "input": size(1536, "b12"), "census": True,
              "args": ["--sampling_steps", "20", "--sampling_steps_affinity", "20"]},
             {"tag": "mn_1536_b12", "surface": "nesso1", "input": size(1536, "b12"), "census": True}])
    fix = [(1536, "rap"), (1536, "b12"), (512, "small"), (1536, "small"), (1024, "small"),
           (1664, "small"), (1792, "small"), (2048, "small"), (512, "rap"), (512, "b12")]
    fix += [(2048, "rap"), (2048, "b12"), (2560, "small"), (3072, "small")]
    write("fix_nesso1", [{"tag": f"nf_{n}_{lig}", "surface": "nesso1", "input": size(n, lig)}
                         for n, lig in fix])
    ysk4 = sorted((HERE / "inputs/screen/ysk4").glob("*.yaml"))
    for k in range(2):
        d = HERE / f"out/shards/ysk4_{k}"
        if not d.exists():  # a running screen reads it, so never rewrite it
            d.mkdir(parents=True)
            for y in ysk4[k::2]:
                shutil.copy(y, d / y.name)
        write(f"screen_nesso1_ysk4_{k}", [{"tag": f"n_screen_ysk4_{k}", "surface": "nesso1",
                                           "input": f"perf/mgx_affinity/out/shards/ysk4_{k}"}])
    write("screen_boltz2_lck", [{"tag": "b_screen_lck", "surface": "boltz2", "input": f"{IN}/screen/lck"}])
    # The four DAVIS salts the first LCK screen lost to standardize(), re-run after the fix.
    salts = HERE / "out/shards/lck_salts"
    if not salts.exists():
        salts.mkdir(parents=True)
        for rid in ("d30_9926054", "d54_11984591", "d63_25127112", "d66_44150621"):
            shutil.copy(HERE / f"inputs/screen/lck/{rid}.yaml", salts / f"{rid}.yaml")
    write("salts_boltz2_lck", [{"tag": "b_lck_salts", "surface": "boltz2",
                                "input": "perf/mgx_affinity/out/shards/lck_salts"}])


if __name__ == "__main__":
    main()

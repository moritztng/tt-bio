"""perf/kitcmp/parse.py on a synthetic session tree: every arm kind lands on its BOARD cell id with the right warm list."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("kitcmp_parse", Path(__file__).parents[1] / "perf/kitcmp/parse.py")
parse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse)

T0 = 1_760_000_000.0


def stamp(d, a, b):
    from datetime import datetime, timezone
    f = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    (d / "t_start").write_text(f(a))
    (d / "t_end").write_text(f(b))
    (d / "rc").write_text("0")


def smi(path, a, b, mhz=1410):
    from datetime import datetime, timezone
    rows = []
    for i in range(int((b - a) * 2)):
        t = datetime.fromtimestamp(a + i / 2, timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        w = 50 if i == 0 else 380  # the idle sample is dropped by the > 100 W filter
        rows.append(f"{t}, {mhz} MHz, 1593 MHz, {w}.00 W, 400.00 W, 60, 99 %, 0x0, 40960 MiB")
    path.write_text("\n".join(rows) + "\n")


def build(r):
    d = r / "boltz2/exact"; d.mkdir(parents=True)
    (d / "times.jsonl").write_text("".join(json.dumps({"s": s, "fn": "boltz.model.models.boltz2.Boltz2.predict_step",
                                                       "t": T0 + i}) + "\n" for i, s in enumerate([30.0, 9.0, 8.9, 8.8, 9.1])))
    (d / "stdout.log").write_text("ACTIVE mode=exact levers=...\n")
    stamp(d, T0 - 1, T0 + 60); smi(r / "boltz2/smi.csv", T0 - 1, T0 + 60)

    d = r / "esmc/exact"; d.mkdir(parents=True)
    (d / "esmc-300m.jsonl").write_text(json.dumps(dict(variant="300m", cold_s=1.5, warm_s=[0.021, 0.020, 0.022, 0.021],
                                                       pooled_digest="ab", batch=7)) + "\n")
    stamp(d, T0, T0 + 5)

    d = r / "rosettafold3-no-prefetch-awrite/exact"; d.mkdir(parents=True)
    stamp(d, T0, T0 + 300)
    (d / "times.jsonl").write_text("".join(json.dumps({"s": 0.1, "fn": "rf3.x.RF3Output.dump", "t": T0 + t}) + "\n"
                                           for t in (100, 140, 181, 221, 262)))

    d = r / "rfdiffusion3/off"; d.mkdir(parents=True)
    stamp(d, T0, T0 + 300)
    (d / "stderr.log").write_text("".join(f"Finished inference batch in {s} seconds\n" for s in (60, 45.1, 45.5, 45.3, 45.9)))

    d = r / "stock-boltzgen"; d.mkdir(parents=True)
    stamp(d, T0, T0 + 400); smi(d / "smi.csv", T0, T0 + 400, mhz=1395)
    (d / "bgg_A100.jsonl").write_text(json.dumps(dict(arm="headline", batch=1, ok=True, cold_loop_s=40.0,
                                                      s_per_design_all=[21.1, 21.2, 21.0, 21.3])) + "\n")

    d = r / "stock-saprot"; d.mkdir(parents=True)
    stamp(d, T0, T0 + 30)
    (d / "gpu_embed_saprot-650m_prot512_a100.json").write_text(json.dumps(
        {"bf16": dict(cold_s=0.5, warm_times_s=[0.03] * 15, attn_engaged=True)}))

    d = r / "stock-nesso1"; d.mkdir(parents=True)
    stamp(d, T0, T0 + 100)
    for leg, reps in (("A", [11.0, 7.5, 7.6, 7.2]), ("B", [10.0, 7.4, 7.3, 7.5])):
        (d / f"nesso1_ladder_aa512_cueq_a100_{leg}.json").write_text(json.dumps(dict(
            label="ladder_aa512_cueq", rep_s=reps, ok=True, affinity={"sha256_of_values": "f" * 64})))



def test_every_arm_kind(tmp_path):
    build(tmp_path)
    got = {(x["cell"], x["mode"]): x for x in parse.arms(tmp_path)}
    b = got[("boltz2", "exact")]
    assert (b["n"], b["median"], b["cold"], b["fn"]) == (4, 8.95, 30.0, "Boltz2.predict_step")
    assert b["sm_mhz"] == 1410 and b["watts"] == 380 and b["engaged"].startswith("ACTIVE mode=exact")
    assert got[("esmc-300m", "exact")]["median"] == 0.021
    rf3 = got[("rf3", "exact")]
    assert rf3["cold"] == 100.0 and rf3["warm"] == [40.0, 41.0, 40.0, 41.0] and rf3["arm"] == "rosettafold3-no-prefetch-awrite"
    assert got[("rfd3", "off")]["warm"] == [45.1, 45.5, 45.3, 45.9]
    bg = got[("boltzgen", "stock")]
    assert (bg["n"], bg["median"], bg["sm_mhz"]) == (4, 21.15, 1395)
    assert got[("saprot-650m", "stock")]["n"] == 15
    ne = got[("nesso1", "stock")]
    assert ne["n"] == 6 and ne["cold"] == 11.0 and ne["median"] == 7.45


def test_board_rows(tmp_path):
    import subprocess, sys
    build(tmp_path / "results")
    (tmp_path / "kit-commit").write_text("f4f62fa6592ae4938d49b1757bea0cfeff9f468e\n")
    s = tmp_path / "results/setup-boltz2"; s.mkdir()
    (s / "pip-freeze.txt").write_text("torch==2.13.0+cu130\nboltz @ file:///root/kit/boltz2/stock/boltz-2.2.1-py3-none-any.whl\n")
    (s / "gpu.csv").write_text("name, driver_version, power.limit [W], clocks.max.sm [MHz]\nNVIDIA A100-SXM4-80GB, 580.173.02, 400.00 W, 1410 MHz\n")
    out = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "perf/kitcmp/board.py"), str(tmp_path), "--gpu", "a100",
                          "--row", "bmk-a100"], capture_output=True, text=True, check=True).stdout.splitlines()
    b = next(l for l in out if l.startswith("| boltz2 | a100 | kit-exact |"))
    c = [x.strip() for x in b.strip("|").split("|")]
    assert c[3] == "8.95" and c[4].startswith("n=4") and "SM 1410 MHz" in c[5]
    assert "kit f4f62fa65" in c[6] and "torch 2.13.0+cu130, boltz 2.2.1" in c[6] and "driver 580.173.02" in c[6] and "--mode exact" in c[8]
    assert any(l.startswith("| nesso1 | a100 | stock | 7.45 |") for l in out)


def test_board_freeze_versions(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("kc_board", Path(__file__).parents[1] / "perf/kitcmp/board.py")
    board = importlib.util.module_from_spec(spec); spec.loader.exec_module(board)
    f = tmp_path / "pip-freeze.txt"
    f.write_text("torch @ https://download.pytorch.org/whl/cu126/torch-2.7.1%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=e1a8\n"
                 "esm @ git+https://github.com/Biohub/esm.git@43ccece2ad485f27db46afdb67da2a9601e8f106\n"
                 "-e git+https://github.com/bytedance/PXDesign.git@f788441313c84c3074fe9596ac2433f96b15c763#egg=pxdesign\n"
                 "rc-foundry @ file:///root/kit/rfdiffusion3/stock/foundry-4010e3e.tar.gz#sha256=2fe8cc01\n")
    assert board.freeze(f) == {"torch": "2.7.1+cu126", "esm": "43ccece2a", "pxdesign": "f78844131", "rc-foundry": "4010e3e"}


def test_esmfold2_variants_read_the_parent_clock(tmp_path):
    """box_arms.sh runs both ESMFold2 variants in one call; the nvidia-smi trace is written to esmfold2/, not the variant dirs."""
    (tmp_path / "esmfold2").mkdir()
    smi(tmp_path / "esmfold2/smi.csv", T0, T0 + 100, mhz=1395)
    d = tmp_path / "esmfold2-fast/off"; d.mkdir(parents=True)
    stamp(d, T0, T0 + 50)
    (d / "times.jsonl").write_text("".join(json.dumps({"s": s, "fn": "esm.x.fold", "t": T0 + i}) + "\n"
                                           for i, s in enumerate([9.0, 4.7, 4.7, 4.8, 4.7])))
    (r,) = [a for a in parse.arms(tmp_path) if a["cell"] == "esmfold2-fast"]
    assert r["sm_mhz"] == 1395

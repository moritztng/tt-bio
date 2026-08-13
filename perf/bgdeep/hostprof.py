"""L1 (the plan's §4.1) screen: cProfile a warm BoltzGen design forward and name the host block.

§1 puts 10.69 s/design (21.1 %) on the host side: 5.670 s of undecomposed featurisation/writer
residual, 1.783 s DiffusionConditioning, 1.673 s template_module, 1.125 s of sampler-loop residual,
0.444 s of small e1d modules. Nothing here touches device math, so a correct host refactor is
bit-exact by construction.

PRE-COMMITTED GATE, from the plan and not moved after seeing the number:
    kill  -- if no single named host function exceeds 1.0 s/design, stop.
    GO    -- iff >= 3.0 s/design lands in <= 3 named functions.

Profiling is explicitly fine to overlap with other work on the box (benchlock exists for TIMED
runs), and cProfile inflates everything uniformly, so this is read as a RANKING and a share, never
as a wall clock. Two designs; design 1 is cold and only design 2 is reported.
"""
from __future__ import annotations
import argparse, cProfile, io, os, pstats, shutil, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="perf/dsfix/fixtures/bg_R3.yaml")
    ap.add_argument("--num-designs", type=int, default=2)
    ap.add_argument("--out", default="perf/bgdeep/hostprof_R3.txt")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    from tt_bio.boltzgen.model.models import boltz as BZ

    # Profile only the SECOND Boltz.forward, so model load and the cold design are excluded.
    state = {"n": 0, "pr": None, "wall": None}
    _orig = BZ.Boltz.forward

    def fwd(self, *x, **k):
        state["n"] += 1
        if state["n"] != 2:
            return _orig(self, *x, **k)
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        try:
            return _orig(self, *x, **k)
        finally:
            pr.disable()
            state["wall"] = time.perf_counter() - t0
            state["pr"] = pr

    BZ.Boltz.forward = fwd

    workdir = Path.home() / "bghostprof_out"
    if workdir.exists():
        shutil.rmtree(workdir)
    argv = ["run", a.spec, "--output", str(workdir), "--num_designs", str(a.num_designs),
            "--protocol", "protein-anything", "--steps", "design",
            "--device_ids", os.environ.get("TT_VISIBLE_DEVICES", "0")]
    from tt_bio.main import _run_boltzgen_cli
    try:
        _run_boltzgen_cli("tt-bio design", argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise

    pr = state["pr"]
    if pr is None:
        print("NO PROFILE CAPTURED", flush=True)
        return 1
    prof_path = str(Path(a.out).with_suffix(".prof"))
    pr.dump_stats(prof_path)
    print(f"=== profile dumped to {prof_path} ===", flush=True)
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    print(f"=== profiled Boltz.forward wall (cProfile-inflated): {state['wall']:.3f} s ===")
    buf.write("\n===== BY CUMULATIVE TIME (top 45) =====\n")
    st.sort_stats("cumulative").print_stats(45)
    buf.write("\n===== BY TOTAL (SELF) TIME (top 45) =====\n")
    st.sort_stats("tottime").print_stats(45)
    txt = buf.getvalue()
    Path(a.out).write_text(f"profiled Boltz.forward wall {state['wall']:.3f} s\n" + txt)
    print(txt[:14000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

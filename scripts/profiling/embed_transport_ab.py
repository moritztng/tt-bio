#!/usr/bin/env python3
"""Run the same embed twice against one pool -- results via shared paths, then via base64.

This is the end-to-end half of the result-transport question. The controller leg is already
costed off-hardware (`controller_transport_ab.py`: ~22 s vs ~0.8 s warm through real HTTP), and
the shared path is already proven byte-exact on a single card. What only a real pool can show is
what that does to the wall, because the base64 tail is time every card spends idle.

Both cells run the identical sequences against the identical warm pool, A then B, and every
output file is hashed so a faster transport that quietly changed an embedding cannot pass.

    python3 embed_transport_ab.py <src-dir> <n-seqs> <controller-url>
"""

import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"


def make_seqs(n, seed=7):
    rnd = random.Random(seed)
    return {f"s{i:05d}": "".join(rnd.choice(AA) for _ in range(rnd.randint(150, 450)))
            for i in range(n)}


def digest(out_dir: Path) -> dict[str, str]:
    """name -> sha256 for every result file, so 'faster' can be checked against 'same'."""
    got = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and not p.name.startswith(".tt-bio-share-"):
            got[p.relative_to(out_dir).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return got


def run_cell(src: str, seqs_yaml: Path, out_dir: Path, controller: str, shared: bool):
    """One embed. `shared` False monkeypatches the co-location offer away in the CLIENT, which
    is where it runs, so the workers see no nonce and take the base64 path."""
    driver = out_dir.parent / f"driver_{out_dir.name}.py"
    driver.write_text(
        "import sys\n"
        "def main():\n"
        "    import tt_bio.main as m\n"
        + ("" if shared else "    m._offer_shared_outputs = lambda p, s: None\n") +
        f"    sys.argv = ['tt-bio', 'embed', {str(seqs_yaml)!r}, '--model', 'esmc-600m',\n"
        f"                '--out_dir', {str(out_dir)!r}, '--controller', {controller!r},\n"
        "                '--batch_size', '8']\n"
        "    m.cli()\n"
        # load-bearing: multiprocessing spawn re-imports __main__ in every worker, and without
        # this guard each worker re-enters the CLI and the run wedges with no output at all
        "if __name__ == '__main__':\n"
        "    main()\n")
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, cwd=src)
    return round(time.monotonic() - t0, 2), p.returncode


def main() -> int:
    src, n, controller = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    base = Path("/home/cust-team/mthuening/g32/embedwindow") / f"n{n}"
    base.mkdir(parents=True, exist_ok=True)
    seqs_yaml = base / "seqs.yaml"
    import yaml
    seqs_yaml.write_text(yaml.safe_dump(make_seqs(n)))

    res = {}
    for label, shared in (("shared_path", True), ("base64", False)):
        out = base / label
        wall, rc = run_cell(src, seqs_yaml, out, controller, shared)
        res[label] = {"wall_s": wall, "rc": rc, "files": len(list(out.rglob("*.npz")))}
        res[label]["digest"] = digest(out)

    a, b = res["shared_path"], res["base64"]
    same = a["digest"] == b["digest"]
    print(json.dumps({
        "n": n,
        "shared_path_wall_s": a["wall_s"], "shared_path_rc": a["rc"], "shared_path_npz": a["files"],
        "base64_wall_s": b["wall_s"], "base64_rc": b["rc"], "base64_npz": b["files"],
        "speedup": round(b["wall_s"] / a["wall_s"], 2) if a["wall_s"] else None,
        "outputs_identical": same,
        "n_files_compared": len(a["digest"]),
        "first_mismatch": None if same else next(
            (k for k in sorted(set(a["digest"]) | set(b["digest"]))
             if a["digest"].get(k) != b["digest"].get(k)), None),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

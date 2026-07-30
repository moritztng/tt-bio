"""The acceptance gate decides this whole campaign, and until now its verdict could only be
exercised by finishing a 492-fold run -- so the one path that matters most, "everything is good,
exit 0", had never executed. These drive it as a subprocess against synthetic slabs.

The gate's contract:
  * a pair is accepted iff SOME host has an `ok` record for it with the right artifact counts, the
    right config and stateable provenance;
  * a pair with no such record is outstanding, whatever else exists for it;
  * `progress.jsonl` is append-only, so an earlier failure record must NOT disqualify a pair that
    later succeeded -- this is the measurement error that made 63 failures out of 1 real one;
  * exit status is the verdict.
"""
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "abag_xm_acceptance.py"
MODELS = ("protenix-v2", "opendde-abag", "boltz2")
N = 50


def rec(target, model, **kw):
    r = {"target": target, "model": model, "status": "ok", "n_samples": N, "mps": 5,
         "n_cifs": N, "n_paes": N, "tt_bio_commit": "abc1234", "tt_bio_tree": "d" * 40,
         "msa_sha": "cafebabe0000" + target[:4]}
    r.update(kw)
    return r


def run_gate(tmp_path, per_host, targets):
    """Drive the gate over synthetic hosts. Returns (returncode, stdout)."""
    tf = tmp_path / "targets.txt"
    tf.write_text("\n".join(targets) + "\n")
    args = [sys.executable, str(GATE), "--targets_from", str(tf)]
    for host, rows in per_host.items():
        p = tmp_path / f"{host}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        args += ["--progress", f"{host}={p}"]
    out = subprocess.run(args, capture_output=True, text=True, timeout=120)
    return out.returncode, out.stdout + out.stderr


def test_a_complete_slab_is_accepted_and_exits_zero(tmp_path):
    """The success path. Two hosts, disjoint halves, every pair good."""
    targets = [f"T{i:02d}" for i in range(8)]
    a = [rec(t, m) for t in targets[:4] for m in MODELS]
    b = [rec(t, m) for t in targets[4:] for m in MODELS]
    code, out = run_gate(tmp_path, {"hostA": a, "hostB": b}, targets)
    assert "VERDICT: ACCEPTED" in out, out
    assert f"accepted folds   : {len(targets) * 3} / {len(targets) * 3}" in out, out
    assert "OUTSTANDING      : 0" in out, out
    assert code == 0, out


def test_an_earlier_failure_does_not_disqualify_a_pair_that_later_succeeded(tmp_path):
    """The measurement error at the heart of this task: progress.jsonl keeps every attempt, so a
    pair that failed at 02:00 and succeeded at 06:00 has both records forever. Counting records
    reported 63 failures where exactly 1 pair lacked a good fold."""
    targets = ["T00"]
    rows = []
    for m in MODELS:
        rows.append(rec("T00", m, status="fold_failed", n_cifs=None, n_paes=None,
                        stderr="another process already holding the card (DeviceInUseError)"))
        rows.append(rec("T00", m, status="incomplete", n_cifs=3,
                        stderr="TT_FATAL a_dim == b_dim"))
        rows.append(rec("T00", m))          # and then it worked
    code, out = run_gate(tmp_path, {"hostA": rows}, targets)
    assert "VERDICT: ACCEPTED" in out, out
    assert code == 0, out


def test_one_missing_pair_fails_the_gate(tmp_path):
    targets = ["T00", "T01"]
    rows = [rec(t, m) for t in targets for m in MODELS][:-1]   # drop T01/boltz2
    code, out = run_gate(tmp_path, {"hostA": rows}, targets)
    assert "VERDICT: NOT ACCEPTED" in out, out
    assert "OUTSTANDING      : 1" in out, out
    assert "never attempted" in out, out
    assert code == 1, out


@pytest.mark.parametrize("defect,shown", [
    ({"n_paes": N + 1}, "n_paes"),            # the aggregate-PAE reconstruction
    ({"n_cifs": N - 1}, "n_cifs"),            # dropped samples
    ({"n_samples": 20}, "n_samples"),         # different sampling depth
    ({"tt_bio_commit": "abc1234-dirty", "tt_bio_tree": None}, "provenance"),
    ({"tt_bio_commit": None, "tt_bio_tree": None}, "provenance"),
])
def test_an_ok_record_with_a_defect_leaves_its_pair_outstanding(tmp_path, defect, shown):
    targets = ["T00"]
    rows = [rec("T00", m) for m in MODELS[:-1]] + [rec("T00", "boltz2", **defect)]
    code, out = run_gate(tmp_path, {"hostA": rows}, targets)
    assert "VERDICT: NOT ACCEPTED" in out, out
    assert "OUTSTANDING      : 1" in out, out
    assert shown in out, out
    assert code == 1, out


def test_mps_only_disqualifies_the_model_that_reads_it(tmp_path):
    """protenix and opendde have supports_multiplicity=False, so max_parallel_samples never
    reaches their samplers -- holding them to it would refold ~330 good folds for nothing."""
    targets = ["T00"]
    rows = [rec("T00", m, mps=3) for m in ("protenix-v2", "opendde-abag")] + [rec("T00", "boltz2")]
    code, out = run_gate(tmp_path, {"hostA": rows}, targets)
    assert "VERDICT: ACCEPTED" in out, out
    assert code == 0, out
    rows = [rec("T00", m) for m in ("protenix-v2", "opendde-abag")] + [rec("T00", "boltz2", mps=3)]
    code, out = run_gate(tmp_path, {"hostA": rows}, targets)
    assert "VERDICT: NOT ACCEPTED" in out and "mps" in out, out
    assert code == 1, out


def test_a_pair_folded_on_either_host_counts_once(tmp_path):
    """Duplicates across hosts are waste, not corruption -- one host takes over the other's
    slices while it is down. The gate must not double-count or reject them."""
    targets = ["T00"]
    rows = [rec("T00", m) for m in MODELS]
    code, out = run_gate(tmp_path, {"hostA": rows, "hostB": list(rows)}, targets)
    assert "accepted folds   : 3 / 3" in out, out
    assert code == 0, out


def test_engine_and_msa_homogeneity_are_reported(tmp_path):
    """One engine tree and one MSA hash set per target is what makes the slab one slab."""
    targets = ["T00", "T01"]
    rows = [rec(t, m) for t in targets for m in MODELS]
    rows[0]["tt_bio_tree"] = "e" * 40                 # a second engine tree
    rows[1]["msa_sha"] = "different0000000"           # same target, different MSA bytes
    code, out = run_gate(tmp_path, {"hostA": rows}, targets)
    assert "engine trees     : 2 distinct" in out, out
    assert "1 target(s) folded against DIFFERENT MSA bytes" in out, out


def test_unparseable_lines_are_counted_not_swallowed(tmp_path):
    targets = ["T00"]
    p = tmp_path / "hostA.jsonl"
    p.write_text("".join(json.dumps(rec("T00", m)) + "\n" for m in MODELS) + "{not json\n")
    tf = tmp_path / "t.txt"; tf.write_text("T00\n")
    out = subprocess.run([sys.executable, str(GATE), "--targets_from", str(tf),
                          "--progress", f"hostA={p}"], capture_output=True, text=True, timeout=120)
    assert "unparseable=1" in out.stdout, out.stdout
    assert out.returncode == 0, out.stdout


def test_a_missing_override_file_is_fatal(tmp_path):
    tf = tmp_path / "t.txt"; tf.write_text("T00\n")
    out = subprocess.run([sys.executable, str(GATE), "--targets_from", str(tf),
                          "--progress", f"hostA={tmp_path / 'nope.jsonl'}"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode != 0
    assert "does not exist" in (out.stdout + out.stderr)


def test_the_json_verdict_matches_the_printed_one(tmp_path):
    targets = ["T00", "T01"]
    rows = [rec(t, m) for t in targets for m in MODELS][:-1]
    p = tmp_path / "hostA.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tf = tmp_path / "t.txt"; tf.write_text("\n".join(targets) + "\n")
    vj = tmp_path / "verdict.json"
    out = subprocess.run([sys.executable, str(GATE), "--targets_from", str(tf),
                          "--progress", f"hostA={p}", "--json", str(vj)],
                         capture_output=True, text=True, timeout=120)
    v = json.loads(vj.read_text())
    assert v["verdict"] == "NOT ACCEPTED" and v["accepted"] == 5 and v["total"] == 6
    assert v["outstanding"] == [["T01", "boltz2"]], v["outstanding"]
    assert out.returncode == 1


def test_a_partial_host_set_says_so_loudly(tmp_path):
    """Reading a subset of the slab as the whole thing is the mistake abag_xm_status_xhost.py was
    written for: qb1 alone once reported 44 of 492 done when the truth was 84, with nothing in the
    output hinting at the other host."""
    targets = ["T00"]
    rows = [rec("T00", m) for m in MODELS]
    # a real host name, so HOSTS minus this set is non-empty
    code, out = run_gate(tmp_path, {"tt-quietbox": rows}, targets)
    assert "NOT the full campaign" in out, out
    assert "tt-quietbox2" in out, out


def test_the_full_host_set_does_not_warn(tmp_path):
    targets = ["T00"]
    rows = [rec("T00", m) for m in MODELS]
    code, out = run_gate(tmp_path, {"tt-quietbox": rows, "tt-quietbox2": []}, targets)
    assert "NOT the full campaign" not in out, out
    assert code == 0, out

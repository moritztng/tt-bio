"""--single_sequence must actually skip the MSA, and must change nothing when it is off.

The flag is documented as "Fold single-sequence: skip MSA entirely". The MSA SEARCH honoured
it, but the chain-spec build did not, and `_resolve_a3m_text` has two sources the search never
produces: an a3m the YAML pinned with `msa:`, and one already cached under msa_dir for that
sequence hash. On any target with an MSA from either source the flag was a silent no-op.

Measured on a 117-aa target, protenix-v1, 20 steps, seed 7: flag ignored -> pLDDT 0.764634,
flag honoured -> 0.501251. A user asking for a no-MSA baseline got the MSA answer.

ESMFold2 was the one predict path that never got the fix: it does not go through
`_build_chain_specs`, it calls `resolve_msa` itself, so the flag stayed a no-op there for a
pinned or cached a3m long after the other models were fixed. That arm is pinned below too.

Both halves are pinned here, and the second half is the one that matters for a shipped model:
with the flag OFF the helper must return exactly what the two hand-written comprehensions
returned before they were replaced, for Protenix (protein chains only) and for OpenDDE (every
chain). No device, no weights.
"""
import sys
import types


def _ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    return cond


CHAINS = [("A", "PEPTIDE", "/pinned/a.a3m", "protein", None),
          ("B", "ACGU", None, "rna", None),
          ("C", "OTHER", None, "protein", None)]


def _patched():
    """Stub _resolve_a3m_text so the test observes WHICH chains it is asked about."""
    import tt_bio.main as M

    calls = []

    def fake(spec, seq, msa_dir):
        calls.append((spec, seq))
        return "A3M:" + seq
    orig = M._resolve_a3m_text
    M._resolve_a3m_text = fake
    return M, orig, calls


def check():
    from tt_bio.worker import _build_chain_specs

    M, orig, calls = _patched()
    try:
        ok = True
        # --- flag OFF: byte-for-byte what the old comprehensions produced -------------------
        del calls[:]
        got = _build_chain_specs(CHAINS, "/msa", {}, protein_only=True)
        want = [(c[1], "A3M:" + c[1] if c[3] == "protein" else None, c[3]) for c in CHAINS]
        ok = _ok(got == want, "flag off, protein_only=True reproduces the Protenix build") and ok

        del calls[:]
        got = _build_chain_specs(CHAINS, "/msa", {}, protein_only=False)
        want = [(c[1], "A3M:" + c[1], c[3]) for c in CHAINS]
        ok = _ok(got == want, "flag off, protein_only=False reproduces the OpenDDE build") and ok

        # --- flag ON: no MSA reaches the featurizer, from ANY source ------------------------
        for po in (True, False):
            del calls[:]
            got = _build_chain_specs(CHAINS, "/msa", {"single_sequence": True}, protein_only=po)
            ok = _ok(all(a3m is None for _s, a3m, _m in got),
                     f"flag on, protein_only={po}: every chain folds without an MSA") and ok
            ok = _ok(not calls,
                     f"flag on, protein_only={po}: the a3m resolver is not even consulted, so a "
                     f"PINNED or CACHED a3m cannot leak in") and ok
            ok = _ok([(s, m) for s, _a, m in got] == [(c[1], c[3]) for c in CHAINS],
                     f"flag on, protein_only={po}: sequences and mol_types are untouched") and ok
        return ok
    finally:
        M._resolve_a3m_text = orig


def main():
    r = check()
    print("\n" + ("PASSED" if r else "FAILURES"))
    return 0 if r else 1


if __name__ == "__main__":
    sys.exit(main())


def test_esmfold2_honours_the_flag(monkeypatch, tmp_path):
    """The path that was missed. ESMFold2 resolves its own MSAs instead of going through
    `_build_chain_specs`, so the flag has to be honoured at that call site as well. With the
    flag on, `resolve_msa` must not be consulted at all -- a pinned or cached a3m is exactly
    what the MSA search cannot prevent."""
    import tt_bio.esmfold2_runtime as R
    from tt_bio.worker import _WorkerState

    asked = []

    def fake_resolve(spec, seq, msa_dir, max_sequences=None):
        asked.append(seq)
        return "A3M"

    folded = {}

    def fake_fold_complex(model, chains, **kw):
        folded["chains"] = chains
        s = types.SimpleNamespace(plddt=_Tensor(0.9), ptm=None, complex=None)
        return [s]

    monkeypatch.setattr(R, "resolve_msa", fake_resolve)
    monkeypatch.setattr(R, "fold_complex", fake_fold_complex)
    monkeypatch.setattr("tt_bio.main._write_structure", lambda *a, **k: None)

    q = tmp_path / "q.yaml"
    q.write_text("version: 1\nsequences:\n  - protein:\n      id: A\n"
                 "      sequence: MKTAYIAKQRQISFVKSHFSRQ\n      msa: /pinned/a.a3m\n")

    state = object.__new__(_WorkerState)
    state.model = types.SimpleNamespace(msa_encoder=object())
    state.accelerator = "cpu"
    cfg = {"model": "esmfold2", "msa_dir": str(tmp_path), "struct_dir": str(tmp_path),
           "output_format": "cif", "recycling_steps": 1, "sampling_steps": 1,
           "diffusion_samples": 1, "seed": 0, "max_msa_seqs": 16}

    metrics, _best, _feats = state.predict_one(q, {**cfg, "single_sequence": True})
    assert asked == [], "resolve_msa was consulted with --single_sequence on"
    assert metrics["msa"] is False
    assert [c[2] for c in folded["chains"]] == [None]

    # The control: with the flag off the same pinned a3m IS used, so the arm above is
    # measuring the flag and not a broken stub.
    del asked[:]
    metrics, _best, _feats = state.predict_one(q, {**cfg, "single_sequence": False})
    assert asked and metrics["msa"] is True


class _Tensor(float):
    """Just enough tensor for the metrics row: `.mean()` on a scalar."""

    def mean(self):
        return self

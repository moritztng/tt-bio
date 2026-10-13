"""Releasing the diffusion context frees the DiT's per-layer step caches too.

The pair bias, adaLN chunks and atom conditioning are cached per sampling trajectory and keyed on the
identity of tensors release_cache() frees. They used to stay on the chip until the next fold rebuilt
them, so a warm 1024-token ESMFold2-Fast fold on one Wormhole chip carried 384 MiB of stale pair bias
through its language model and trunk and died with TT_FATAL Out of Memory where the cold fold fit.
The device repro is the pool job in docs/esmfold2.md (cold + 3 warm at the largest Wormhole size).
"""
import types

import pytest

pytest.importorskip("ttnn")


class _T:
    def __init__(self):
        self.alive = True

    def is_allocated(self):
        return self.alive


def test_release_cache_frees_step_caches(monkeypatch):
    from tt_bio import esmfold2 as E

    monkeypatch.setattr(E.ttnn, "deallocate", lambda t: setattr(t, "alive", False))
    ns = types.SimpleNamespace
    attn = [ns(_pb=_T(), _pb_src=object()) for _ in range(3)]
    blocks = [ns(_mod=tuple(_T() for _ in range(6)), _mod_src=object()) for _ in range(4)]
    m = object.__new__(E.DiffusionModuleModel)
    m.token_transformer = ns(attn=attn)
    m.atom_encoder = ns(_c=_T(), _c_src=object(), transformer=ns(blocks=blocks[:2]))
    m.atom_decoder = ns(transformer=ns(blocks=blocks[2:]))
    m._ctx = {}
    held = [a._pb for a in attn] + [t for b in blocks for t in b._mod] + [m.atom_encoder._c]

    m.release_cache()

    assert not any(t.alive for t in held)
    assert all(a._pb is None and a._pb_src is None for a in attn)
    assert all(b._mod is None and b._mod_src is None for b in blocks)
    assert m.atom_encoder._c is None and m.atom_encoder._c_src is None
    m.release_cache()  # a second release (sample() after draw()) is a no-op, not a double free


def test_unbuilt_caches_are_a_noop(monkeypatch):
    from tt_bio import esmfold2 as E

    monkeypatch.setattr(E.ttnn, "deallocate", lambda t: pytest.fail("nothing to free"))
    ns = types.SimpleNamespace
    m = object.__new__(E.DiffusionModuleModel)
    m.token_transformer = ns(attn=[ns(_pb=None, _pb_src=None)])
    m.atom_encoder = ns(_c=None, _c_src=None, transformer=ns(blocks=[ns(_mod=None, _mod_src=None)]))
    m.atom_decoder = ns(transformer=ns(blocks=[]))
    m._ctx = {}
    m.release_cache()

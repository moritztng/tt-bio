"""Every trace region is sized in one place, tenstorrent.TRACE_REGIONS, by capture and arch.

A caller names the capture it will replay (`get_device(trace="diffusion")`) and never passes
bytes. The literal this replaces, `get_device(trace_region_size=1 << 30)` in boltz2 and
boltzgen, asked for more than a whole Wormhole DRAM bank (1073741792 B): the allocator's bank
size underflowed to 2**64 - 32 and the chip's first dispatch never completed (whglx cards 10, 11
and 22, perf/mgx_trace_region). These tests fail if bytes come back at a call site, if a name
matches no entry, if an entry could eat a bank, or if a new capture site appears unsized.

No device is opened here.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = [ROOT / "tt_bio", ROOT / "scripts"]
WORMHOLE_BANK = 1073741792   # bytes per DRAM bank on a Wormhole chip, measured on j10glx02

# Every file that begins a ttnn trace capture, and the TRACE_REGIONS entry that sizes it.
CAPTURE_SITES = {
    "tt_bio/tenstorrent.py": "diffusion",
    "tt_bio/protenix.py": "protenix",
    "tt_bio/esmc.py": "esmc",
}


def _py(dirs):
    for d in dirs:
        yield from sorted(d.rglob("*.py"))


def _calls(path):
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _name(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def test_no_call_passes_trace_region_bytes():
    bad = [f"{p.relative_to(ROOT)}:{c.lineno}"
           for p in _py(SHIPPED) for c in _calls(p)
           if any(k.arg == "trace_region_size" for k in c.keywords)]
    assert not bad, (f"these calls pass trace-region bytes; name the capture instead, "
                     f"get_device(trace=<TRACE_REGIONS key>): {bad}")


def test_the_byte_env_var_is_gone():
    bad = [str(p.relative_to(ROOT)) for p in _py(SHIPPED)
           if "TT_BIO_TRACE_REGION_SIZE" in p.read_text()]
    assert not bad, (f"TT_BIO_TRACE_REGION_SIZE handed raw bytes past the sizing table (the rfd3 "
                     f"hang ran on it); it is not read any more: {bad}")


def _self_sized(path):
    """Capture names the file registers itself, ``TRACE_REGIONS["name"] = {...}``.

    A measurement harness under ``perf/`` sizes its own capture at run time, right before it
    opens the card (``bcx_predictor/trace_wire.py`` asks for 768 MiB because metal refused the
    step at 510). That is the sizing this file asks for, written where the number was measured,
    so it is not an unsized capture.
    """
    names = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript)
                    and _name_of(target.value) == "TRACE_REGIONS"
                    and isinstance(target.slice, ast.Constant)):
                names.add(target.slice.value)
    return names


def _name_of(node):
    return node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)


def test_every_named_capture_has_an_entry():
    from tt_bio.tenstorrent import TRACE_REGIONS
    named = {}
    for p in _py(SHIPPED + [ROOT / "perf"]):
        try:
            sized_here = _self_sized(p)
        except SyntaxError:
            continue
        for c in _calls(p):
            if _name(c) != "get_device":
                continue
            for k in c.keywords:
                if (k.arg == "trace" and isinstance(k.value, ast.Constant) and k.value.value
                        and k.value.value not in sized_here):
                    named.setdefault(k.value.value, []).append(f"{p.relative_to(ROOT)}:{c.lineno}")
    assert named, "no get_device(trace=...) call found; this test no longer sees the callers"
    unknown = {n: w for n, w in named.items() if n not in TRACE_REGIONS}
    assert not unknown, (f"captures with no TRACE_REGIONS entry and none set at the call site: "
                         f"{unknown}")


def test_every_capture_site_is_sized():
    found = {str(p.relative_to(ROOT)) for p in _py([ROOT / "tt_bio"])
             for c in _calls(p) if _name(c) == "begin_trace_capture"}
    assert found == set(CAPTURE_SITES), (
        f"begin_trace_capture moved or a new site appeared: {sorted(found ^ set(CAPTURE_SITES))}. "
        f"Measure its bytes (perf/mgx_trace_region) and add a TRACE_REGIONS entry.")
    from tt_bio.tenstorrent import TRACE_REGIONS
    assert set(CAPTURE_SITES.values()) <= set(TRACE_REGIONS)


def test_no_entry_can_eat_a_bank():
    """A quarter of a Wormhole bank at most, on every arch: the region is taken from every bank,
    so anything near the bank size starves the model and at the bank size wedges the chip.
    The largest entry, ESMC's eight live traces on Wormhole, is 221 MiB."""
    from tt_bio.tenstorrent import TRACE_REGIONS
    for cap, per_arch in TRACE_REGIONS.items():
        assert set(per_arch) == {"wormhole_b0", "blackhole"}, (cap, sorted(per_arch))
        for arch, n in per_arch.items():
            assert 0 < n <= WORMHOLE_BANK // 4, (cap, arch, n)
            assert n % (1 << 20) == 0, (cap, arch, n)


def test_lookup_is_by_arch_and_refuses_unknown_names(monkeypatch):
    from tt_bio import tenstorrent as T
    monkeypatch.setattr(T, "arch_name", lambda: "wormhole_b0")
    assert T.trace_region_bytes("diffusion") == T.TRACE_REGIONS["diffusion"]["wormhole_b0"]
    monkeypatch.setattr(T, "arch_name", lambda: "blackhole")
    assert T.trace_region_bytes("diffusion") == T.TRACE_REGIONS["diffusion"]["blackhole"]
    with pytest.raises(ValueError, match="known captures"):
        T.trace_region_bytes("1 << 30")


def test_rfd3_traces_refuse_by_name():
    """Both RFD3 traces are withdrawn: the encoder's hung a Wormhole chip after its first replay,
    the decoder's was slower, changed the design and ran out of memory at 1280 residues. Each
    flag must raise, naming itself, before anything is captured (rfd3 has no capture site left,
    which test_every_capture_site_is_sized pins)."""
    src = (ROOT / "tt_bio/rfd3/model.py").read_text()
    init = src[src.index("class RFD3DiffusionModule"):]
    init = init[:init.index("self.encoder = LocalAtomTransformer(")]
    assert '("RFD3_TRACE_ENCODER",' in init and '("RFD3_TRACE_DECODER",' in init
    assert "is withdrawn" in init and "raise ValueError(" in init

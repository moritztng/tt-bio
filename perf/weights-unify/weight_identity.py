"""Every model must load byte-identical weights before and after the registry.

Weight loading is not compute: if the resolution path hands a model the same bytes, the
fold is unchanged by construction. That is the whole correctness claim here, and it is
checkable on a host with no device, which matters because pc's card 0 cannot host
hash-equality gating (memory pc-card0-512aa-fold-nondeterminism).

Run twice, once per tree, then diff the JSON. Copy this file somewhere that is NOT a
tt-bio package directory first: python puts the script's own directory on sys.path, which
would shadow the tree PYTHONPATH is meant to select.

    git archive origin/main | tar -x -C /tmp/ttbio-main
    cp weight_identity.py /tmp/wi/ && cd anywhere
    PYTHONPATH=/tmp/ttbio-main  python3 /tmp/wi/weight_identity.py before.json
    PYTHONPATH=<worktree>       python3 /tmp/wi/weight_identity.py after.json
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def sha(path: Path, cap: int = 512 << 20) -> str:
    """SHA-256 of the file, or of its first 512 MiB plus its size for the huge ones
    (a re-download changes the size or the leading bytes; hashing 24 GiB per row does
    not buy anything)."""
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while n < cap and (chunk := fh.read(1 << 22)):
            h.update(chunk)
            n += len(chunk)
    return f"{h.hexdigest()}:{path.stat().st_size}"


def digest(path: Path) -> str:
    p = Path(path)
    if p.is_file():
        return sha(p)
    files = sorted(f for f in p.rglob("*") if f.is_file() and ".complete-" not in f.name)
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(p).as_posix().encode())
        h.update(str(f.stat().st_size).encode())
    return f"dir:{h.hexdigest()}:{len(files)}"


try:
    from tt_bio import weights as _probe          # noqa: F401
    HAVE_REGISTRY = True
except ImportError:
    HAVE_REGISTRY = False


def resolve_all() -> dict:
    """Every model's weight resolution, called the way a worker calls it."""
    from tt_bio.worker import _ensure_local_artifacts

    out = {}
    for model in ("boltz2", "protenix-v2", "openfold3", "esmfold2", "esmfold2-fast",
                  "opendde", "opendde-abag", "rf3", "esmc-300m", "esmc-600m", "esmc-6b",
                  "saprot-35m", "saprot-650m"):
        cfg = {"model": model, "msa_dir": None}
        try:
            _ensure_local_artifacts(cfg)
        except Exception as e:
            out[model] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
            continue
        row = {}
        for k, v in sorted(cfg.items()):
            if k.endswith(("_ckpt", "_dir")) and v and k != "msa_dir" and Path(str(v)).exists():
                row[k] = {"path": str(v), "digest": digest(Path(str(v)))}
        out[model] = row

    # the embedding/design loaders resolve outside _ensure_local_artifacts
    from tt_bio.esmc import CONFIGS as ESMC_CONFIGS
    from tt_bio.saprot import CONFIGS as SAPROT_CONFIGS
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        for name, (_c, repo, wpath) in ESMC_CONFIGS.items():
            p = Path(hf_hub_download(repo, wpath))
            out.setdefault(name, {})["weights"] = {"path": str(p), "digest": digest(p)}
        for name, (_c, repo) in SAPROT_CONFIGS.items():
            if name == "saprot-1.3b":
                continue                     # not cached on this host
            p = Path(snapshot_download(repo))
            out.setdefault(name, {})["snapshot"] = {"path": str(p), "digest": digest(p)}
        out["esmc-6b"] = {"snapshot": {"path": str(Path(snapshot_download("biohub/ESMC-6B"))),
                                       "digest": digest(Path(snapshot_download("biohub/ESMC-6B")))}}
    except Exception as e:
        out["_embed_error"] = f"{type(e).__name__}: {str(e)[:120]}"

    # The whole-repo rows load lazily inside the model, so _ensure_local_artifacts puts
    # no path in cfg for them. Resolve them the way each tree actually does: the old code
    # called snapshot_download / hf_hub_download directly, the new code calls the registry.
    repos = {"esmfold2": ("biohub/ESMFold2", None),
             "esmfold2-fast": ("biohub/ESMFold2-Fast", None),
             "opendde": ("aurekaresearch/OpenDDE", "opendde.pt"),
             "opendde-abag": ("aurekaresearch/OpenDDE", "opendde_abag.pt")}
    hub = {}
    for key, (repo, fname) in repos.items():
        try:
            if HAVE_REGISTRY:
                from tt_bio import weights
                p_ = Path(weights.fetch(key, quiet=True))
            else:
                from huggingface_hub import hf_hub_download, snapshot_download
                p_ = Path(hf_hub_download(repo, fname) if fname else snapshot_download(repo))
            hub[key] = {"path": str(p_), "digest": digest(p_)}
        except Exception as e:
            hub[key] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    out["_hub_repos"] = hub

    # BoltzGen's six, through its own resolver
    import argparse

    from tt_bio.boltzgen.cli.boltzgen import ARTIFACTS as BG, get_artifact_path
    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser()))) / "boltzgen"
    args = argparse.Namespace(cache=cache, force_download=False)
    bg = {}
    for name, (spec, rtype) in BG.items():
        want = cache / spec.rsplit(":", 1)[-1]
        if not want.exists():
            bg[name] = {"missing": str(want)}
            continue
        try:
            import zipfile
            if want.suffix in (".ckpt", ".pt", ".zip") and not zipfile.is_zipfile(want):
                bg[name] = {"corrupt_on_this_host": str(want), "size": want.stat().st_size}
                continue
            p = get_artifact_path(args, spec, repo_type=rtype, verbose=False)
            bg[name] = {"path": str(p), "digest": digest(Path(p))}
        except Exception as e:
            bg[name] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    out["boltzgen"] = bg
    return out


if __name__ == "__main__":
    import tt_bio
    data = {"tree": str(Path(tt_bio.__file__).parent), "models": resolve_all()}
    Path(sys.argv[1]).write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"wrote {sys.argv[1]} from {data['tree']}")

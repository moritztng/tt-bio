"""Two tt-bio processes sharing one weight cache must not download into each other.

`fetch_file` stages into a stable `.<name>.part` so an interrupted multi-GB download
resumes rather than restarts. That is worth keeping, and it means two processes fetching
the same artifact write the same file and each other's retry unlinks it;
`sweep_stale_staging` separately deletes staging older than an hour whoever owns it. Five
workers ran `tt-bio weights --download` over one cache and hit both. One lock per artifact
key is the fix that keeps the resume.
"""
import multiprocessing as mp
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import tt_bio.weights as weights

HOLD_S = 1.5


def _hold(key, root, q):
    with weights._artifact_lock(key, root, quiet=True):
        q.put("held")
        time.sleep(HOLD_S)


def _grab(key, root, q):
    t0 = time.time()
    with weights._artifact_lock(key, root, quiet=True):
        q.put(time.time() - t0)


def _contend(first, second, root):
    # fork, not spawn: a spawned child re-imports tt_bio, which pulls in ttnn and costs
    # more than this test is measuring.
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    a = ctx.Process(target=_hold, args=(first, root, q))
    a.start()
    assert q.get(timeout=30) == "held"
    b = ctx.Process(target=_grab, args=(second, root, q))
    b.start()
    waited = q.get(timeout=60)
    a.join(30)
    b.join(30)
    return waited


@pytest.mark.parametrize("second,serialised", [("rf3", True), ("boltz2-conf", False)])
def test_one_lock_per_artifact_key(tmp_path, second, serialised):
    """The same key waits; a different key is the negative control and must not."""
    waited = _contend("rf3", second, tmp_path)
    if serialised:
        assert waited > HOLD_S / 2, f"second process on the same key waited {waited:.2f}s"
    else:
        assert waited < HOLD_S / 2, f"different keys blocked each other for {waited:.2f}s"


def test_lock_is_released_when_the_holder_dies(tmp_path):
    """A killed downloader must not wedge every other process out of the cache."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    a = ctx.Process(target=_hold, args=("rf3", tmp_path, q))
    a.start()
    assert q.get(timeout=30) == "held"
    a.kill()
    a.join(30)
    b = ctx.Process(target=_grab, args=("rf3", tmp_path, q))
    b.start()
    waited = q.get(timeout=60)
    b.join(30)
    assert waited < HOLD_S / 2, f"lock survived its holder ({waited:.2f}s)"



def test_concurrent_fetchers_download_one_file_once(tmp_path, monkeypatch):
    """Six BoltzGen shards on a cold cache: every one gets the file, and it comes down once.

    They call `fetch_file` directly, not through a registry key, which is how the dev
    fleet lost a shard. The download count is the assertion that matters: a check on
    exit status alone passes on the racy code whenever the race happens to be won.
    Threads rather than processes: flock is held per open file, so they contend exactly
    as processes do, and a fork after `tt_bio.main` has started threads can deadlock."""
    import huggingface_hub

    downloads = []

    def slow_hub(repo_id, filename, local_dir, **kw):
        downloads.append(filename)
        time.sleep(0.5)                  # every shard is mid-download at once
        out = Path(local_dir) / filename
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("data.pkl", b"w" * 4096)
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", slow_hub)
    dest = tmp_path / "cache" / "w.ckpt"
    with ThreadPoolExecutor(6) as pool:
        shards = [pool.submit(weights.fetch_file, ("hf://repo/w.ckpt",), dest, quiet=True)
                  for _ in range(6)]
        results = [f.result(timeout=60) for f in shards]
    assert results == [dest] * 6
    assert len(downloads) == 1, f"downloaded {len(downloads)} times, not once"
    assert weights.artifact_intact(dest)

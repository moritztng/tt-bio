#!/usr/bin/env python3
"""How often did the card run a different checkpoint from the JAX around it, per git rev?

Every device acceptance reading this campaign holds was produced before 1127f9ea8. That
commit fixed a resolution defect: `predict` and `sequence_gradients` resolve `model`
themselves (`bindcraft/af2.py:287,381`) and resolving `None` SAMPLES a name, so a second
resolution to pick the card's trunk drew a second, independent name. `bcx-multimer` found
it and put a number on the fixed arm. This asks the other question -- what the number was
on the trees that actually produced the trajectories we read rejections off.

No card, no weights: which name each side gets is decided before any arithmetic runs.

    python3 perf/bcx_mutate/ckpt_mismatch_by_tree.py <rev> [<rev> ...]
"""
import pathlib
import subprocess
import sys
import tempfile

DRAWS = 20
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

PROBE = '''
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import jax
import bc2_state  # noqa: F401  -- puts BindCraft 2 on the path
from bindcraft.af2 import AlphaFoldDesignModel
from multimer_pool import POOL
from ttbio_predictor import TTBioAlphaFoldDesignModel

DRAWS = {draws}
card = []


class FakePool:
    def use(self, name):
        card.append(name)


m = object.__new__(TTBioAlphaFoldDesignModel)
m.key = jax.random.PRNGKey(0)
m.models = POOL
m.model_families = dict((n, ("multimer", "v3")) for n in POOL)
m.pool, m.trunk, m.card, m._device = FakePool(), "device", None, None

jax_side = []


def predict(self, protein_states, model=None, *a, **k):
    jax_side.append(self._resolve_model_name(model))
    return {{}}


def sequence_gradients(self, protein_states, losses, model=None, *a, **k):
    jax_side.append(self._resolve_model_name(model))
    return {{}}, {{}}


AlphaFoldDesignModel.predict = predict
AlphaFoldDesignModel.sequence_gradients = sequence_gradients

for i in range(DRAWS):
    if i % 2:
        m.predict({{}}, None)
    else:
        m.sequence_gradients({{}}, {{}}, None)

n = min(len(card), len(jax_side))
bad = sum(1 for a, b in zip(card, jax_side) if a != b)
print(str(bad) + "/" + str(n))
'''.format(draws=DRAWS)


def measure(rev):
    sha = subprocess.run(['git', 'rev-parse', '--short', rev], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        tar = subprocess.run(['git', 'archive', rev, 'perf/bcx_predictor'], cwd=REPO,
                             capture_output=True, check=True).stdout
        subprocess.run(['tar', '-x', '-C', tmp], input=tar, check=True)
        d = pathlib.Path(tmp) / 'perf' / 'bcx_predictor'
        probe = d / '_probe.py'
        probe.write_text(PROBE)
        r = subprocess.run([sys.executable, str(probe)], cwd=str(REPO),
                           capture_output=True, text=True,
                           env={'PYTHONPATH': str(REPO), 'PATH': '/usr/bin:/bin',
                                'JAX_PLATFORMS': 'cpu'})
        if r.returncode:
            tail = r.stderr.strip().splitlines()
            return sha + '  ERROR: ' + (tail[-1] if tail else str(r.returncode))
        return sha + '  card on a different checkpoint: ' + r.stdout.strip() + ' draws'


if __name__ == '__main__':
    for rev in (sys.argv[1:] or ['cad102bc5', '1127f9ea8', 'HEAD']):
        print(measure(rev))

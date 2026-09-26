"""Every attribute perf/bcx_extrawire/round_ab.py reaches into, checked without a device.

The A/B run is the expensive thing this row is waiting on, and the harness monkeypatches
BindCraft 2 and tt-bio internals by name. A name that moved dies in minute one on a card nobody
else can have. None of this opens a device: it is imports and introspection only.
"""
import os, sys, json, inspect
os.environ.setdefault("TT_VISIBLE_DEVICES", "")
SELFTEST = "--selftest" in sys.argv
sys.path.insert(0, "perf/bcx_extrawire")
sys.path.insert(0, "perf/bcx_round")
sys.path.insert(0, "perf/bcx_predictor")

fail, ok = [], []

def chk(name, cond, detail=""):
    (ok if cond else fail).append(f"{name}{(' -- ' + detail) if detail else ''}")

import round_ab as R
import meter as M
from tt_bio import bindcraft2 as bc2
import bindcraft.af2 as bc2_af2
import bindcraft.campaign as campaign
import bindcraft.trajectory as trajectory
import bindcraft.sequence_optimization as seqopt

# 1. the device seams round_ab retags
for cls_name in ("EvoformerOnDevice", "ExtraMsaOnDevice"):
    cls = getattr(bc2, cls_name, None)
    chk(f"bc2.{cls_name}", cls is not None)
    for m in ("_primal", "_taped", "_backward"):
        chk(f"bc2.{cls_name}.{m}", callable(getattr(cls, m, None)))

# 2. what meter.install patches
cls = bc2.design_model_class()
for m in ("sequence_gradients", "predict"):
    chk(f"design_model_class().{m}", callable(getattr(cls, m, None)))
chk("meter.install(splice_mod) needs .EvoformerOnDevice", hasattr(bc2, "EvoformerOnDevice"))
sig = inspect.signature(M.install)
chk("meter.install arity 5", len(sig.parameters) == 5, str(sig))

# 3. what per_arm_cache overrides -- the shared-trace guard, the part that must not silently break
chk("AlphaFoldDesignModel._compiled_sequence_gradients",
    callable(getattr(bc2_af2.AlphaFoldDesignModel, "_compiled_sequence_gradients", None)))
chk("bindcraft.af2.CompiledModelCache", hasattr(bc2_af2, "CompiledModelCache"))
cm = getattr(bc2_af2, "CompiledModelCache", None)
if cm is not None:
    chk("CompiledModelCache(max_size) accepts one positional",
        len([p for p in inspect.signature(cm).parameters.values()
             if p.default is inspect.Parameter.empty]) <= 1, str(inspect.signature(cm)))
    chk("CompiledModelCache instance exposes .max_size",
        "max_size" in (inspect.getsource(cm) if hasattr(cm, "__module__") else ""))
src = inspect.getsource(bc2_af2.AlphaFoldDesignModel)
for attr in ("gradient_compile_cache", "alphafold_runners"):
    chk(f"AlphaFoldDesignModel sets self.{attr}", f"self.{attr}" in src)

# 4. the rest of the campaign surface round_ab touches
chk("campaign.MULTIMER_POOL", hasattr(campaign, "MULTIMER_POOL"))
chk("campaign.AlphaFoldDesignModel", hasattr(campaign, "AlphaFoldDesignModel"))
chk("campaign.run_campaign", callable(getattr(campaign, "run_campaign", None)))
chk("trajectory stage fns present for meter",
    any(hasattr(trajectory, f) for f in ("run_gradient_design_stage",
                                         "run_sequence_mutation_stage")))
chk("seqopt has an update_sequence somewhere",
    any("update_sequence" in getattr(getattr(seqopt, n), "__dict__", {})
        for n in dir(seqopt) if isinstance(getattr(seqopt, n), type)))
from bindcraft.af.alphafold.model import layer_stack as LS
chk("layer_stack.layer_stack is patchable", callable(getattr(LS, "layer_stack", None)))

# 5. predictor really takes the keyword round_ab passes
chk("predictor(extra_msa=) exists",
    "extra_msa" in inspect.signature(bc2.predictor.__wrapped__).parameters)

# 6. The discriminator. Both device swaps pick their target out of `layer_stack`'s argument by
# `fn.__name__` -- `tt_bio/bindcraft2.py:784-786` for the shipped code, `round_ab.by_arm` for the
# harness. Three things have to hold at once or a swap goes INERT rather than loud: the names have
# to still be BindCraft 2's, and `hk.remat` has to carry them through.
#
# The remat half is the load-bearing one and it is not obvious. `bindcraft/af2.py:249` sets
# `use_remat = True` on every AF2 runner it builds, so by the time either stack reaches the
# dispatcher it is an `hk.remat` wrapper, not the function BindCraft 2 wrote. Its `__name__`
# survives only because haiku wraps with `functools.wraps` (`haiku/_src/stateful.py:355,365`).
# A haiku release that drops that makes BOTH swaps stop firing with no error at all: the fold
# still runs, the answer is still right, and every perf number on this campaign quietly becomes
# BindCraft 2's own JAX. That is worth five seconds of import here.
import bindcraft.af.alphafold.model.modules as bc2_modules

modules_src = inspect.getsource(bc2_modules)
# Only the extra-MSA name appears in the harness: `by_arm` hands everything else to the spliced
# factory, so the Evoformer keeps going to the card in both arms and the swap under test is the
# only thing that moves. The Evoformer name is the shipped dispatcher's business, checked below.
chk("round_ab dispatches on extra_msa_stack_fn",
    "extra_msa_stack_fn" in inspect.getsource(R.main))
chk("bindcraft2.choose dispatches on both names",
    all(n in inspect.getsource(bc2.evoformer_on_device)
        for n in ("extra_msa_stack_fn", "evoformer_fn")))
chk("BindCraft 2 turns use_remat ON, so the remat path is the live one",
    "use_remat = True" in inspect.getsource(bc2_af2.AlphaFoldDesignModel))
chk("modules.py remats both stacks before layer_stack sees them",
    modules_src.count("hk.remat(") >= 2)


def remat_keeps_the_name(remat):
    """Does `remat` hand back something still called `extra_msa_stack_fn`?

    The wrapper is an argument so `--selftest` can pass one that drops the name and watch this
    return False. A guard nobody has seen fail is a guard nobody knows discriminates.
    """
    def extra_msa_stack_fn(y):
        return y
    return getattr(remat(extra_msa_stack_fn), "__name__", None) == "extra_msa_stack_fn"


def names_present(src, names):
    """Are BindCraft 2's two stack functions still spelled the way both dispatchers match?"""
    return all(f"def {n}(" in src for n in names)


STACK_FNS = ("extra_msa_stack_fn", "evoformer_fn")
chk("BindCraft 2 still names both stack functions", names_present(modules_src, STACK_FNS))

_seen = {}
try:
    import jax
    import haiku as hk

    def _probe(x):
        _seen["ok"] = remat_keeps_the_name(hk.remat)
        return x
    hk.transform(_probe).init(jax.random.PRNGKey(0), 1.0)
except Exception as exc:                                        # pragma: no cover - defensive
    _seen["error"] = f"{type(exc).__name__}: {exc}"
chk("hk.remat preserves __name__, so neither swap goes silently inert",
    _seen.get("ok") is True, str(_seen))

if SELFTEST:
    # Both predicates, fed a broken premise, must say so. Without this the two checks above are
    # two more lines that have only ever printed "passed".
    _renamed = modules_src.replace("def extra_msa_stack_fn(", "def stack_fn_renamed_upstream(")
    assert names_present(modules_src, STACK_FNS), "the real source should pass"
    assert not names_present(_renamed, STACK_FNS), "a renamed stack should fail"
    assert not remat_keeps_the_name(lambda f: lambda *a, **k: f(*a, **k)), \
        "a wrapper without functools.wraps should fail"
    assert remat_keeps_the_name(lambda f: f), "an identity wrapper should pass"
    print("wirecheck selftest ok: both discriminator guards fail on a broken premise")

print(json.dumps({"checked": len(ok) + len(fail), "passed": len(ok),
                  "FAILED": fail}, indent=1))
sys.exit(1 if fail else 0)

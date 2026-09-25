# Make the design predictor selectable, so `DifferentiableProteinPredictor` is a supported contract

*Draft. Nothing has been posted. Text below is ready to open as an issue on PacesaLab/BindCraft2
against `301efdd1937fb963cc40a5b0ecc1bc2f9b2b2d15`, with `FACTORY.patch` attached or pasted.*

---

**Title:** Resolve the design predictor through a factory so alternative `DifferentiableProteinPredictor` implementations can be used

`bindcraft/prediction.py:14` defines `DifferentiableProteinPredictor(ProteinPredictor, Protocol)`
with a single method, `sequence_gradients`, and the design loop is typed against that Protocol
rather than against a concrete class: 7 signatures in `bindcraft/trajectory.py` take
`design_model: DifferentiableProteinPredictor`, `run_trajectory` and `run_gradient_design_stage`
among them. `campaign.py:60 refuse_predictor_without_distogram` reads `provides_distogram` off the
predictor by `getattr`, which only makes sense if a predictor might not be AlphaFold.

So the abstraction is already there. The one thing that closes it off is that the concrete class is
named directly at the two places a campaign builds one, both in `bindcraft/campaign.py`:

```
262  alphafold_model = AlphaFoldDesignModel(presets=selected_models.design_models, ...)
265  build_validation_model = lambda validation_models: AlphaFoldDesignModel(presets=validation_models, ...)
```

Those are the only two constructions in the package (`bindcraft/af/` excluded; checked by AST walk
over every `.py`, not by grep on the name).

**What this asks for.** Route both through one resolver that reads a setting and returns
`AlphaFoldDesignModel` when the setting is absent. Diff attached: **+18 / -2 in one file**, no new
module, no new dependency, no change to any existing setting.

- `design_model` absent, `None`, or `"alphafold2"` → `AlphaFoldDesignModel`, so every existing
  campaign, example and test behaves exactly as now.
- `register_design_model(name, factory)` lets a package register at import time.
- An unknown name raises `ValueError` naming the registry contents rather than failing later inside
  the optimiser.

**Why both sites matter and not just line 262.** Design and validation are built from the same
resolved callable in the patch. If only the design construction were changed, a campaign could
optimise against one predictor and validate against another without saying so anywhere in its
output, and the resulting accepted-design counts would not mean what they appear to mean. That
split is the failure mode worth designing out, and it is why the patch resolves once and closes the
validation lambda over the same callable.

**What it unblocks.** Anyone with a differentiable structure predictor that satisfies the Protocol
can run a BindCraft 2 campaign against it without patching the package, which also means their
results are reproducible against a released version rather than against a fork. Today the only way
in is to rebind the module-level name at import time, which works by accident of how Python
resolves globals and is not something you have promised to keep.

**What I checked, and where it stops.** The patch compiles, and the resolver plus both construction
sites pass 9 checks: identical class returned for absent / `None` / `"alphafold2"` settings, the
registered factory returned for a registered name, `ValueError` for an unknown one, and — read off
the compiled bytecode of `run_campaign` — the design site using the resolved callable, the
validation lambda closing over that same callable, and no remaining construction bypassing the
resolver. Those checks run without JAX, and they are static plus unit-level. **I have not run a full
campaign against this patch**, so what I can say is that the default path resolves to the same class
by the same call, not that an end-to-end campaign was re-measured. If you want that before taking
it, that is a fair thing to ask for.

Happy for this to land in whatever form you prefer, including reimplemented — the useful part is
that the extension point exists, not whose commit it arrives in.

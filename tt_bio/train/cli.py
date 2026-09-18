"""Tier 0: ``tt-bio finetune``. Every knob is a flag, and no flag is a callable.

The cut line, and it is a test rather than a description: **no callables in the signature.**
That is what makes a Tier-0 run's legality decidable before a device opens -- a flag
combination can be checked, printed and refused on a laptop, and ``--dry-run`` does exactly
that. A surface that took a callable could not, because the only way to find out what a
callable does is to run it.

So this module imports nothing from ttnn at module scope, and ``--dry-run`` returns without
importing it either. The model's forward is resolved by name from the shipped catalogue only
at the point the run actually starts.

Registered lazily on the main CLI group, which is why the command body lives here rather than
in ``tt_bio/main.py``: ``tt-bio predict`` must not pay for the training stack, and
``tt-bio --help`` must not import the tape to print one line of help.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

__all__ = ["finetune"]


# Models whose forward routes through `tt_bio.ops.linear`, so an adapter can attach to it.
# A name absent here is not adaptable today and the refusal says so rather than failing later
# with an empty census. Kept as a list because it is a fact about the attach work that has
# landed, not a preference -- when a model gets routed it gets added here in the same change.
ADAPTABLE = ("protenix-v2", "openfold3")

# The Tier-1 bodies, by name. Duplicated from `recipes._RECIPES` on purpose and pinned by a
# test: validating `--recipe` must not import the bodies, because importing them imports the
# tape and `--dry-run` promises not to. The test fails if the two ever disagree.
RECIPE_NAMES = ("lora",)


@click.command("finetune")
@click.argument("data", type=click.Path(exists=True, dir_okay=True))
@click.option("--model", required=True, type=click.Choice(ADAPTABLE),
              help="Which shipped forward to adapt.")
@click.option("--out", "out_dir", required=True, type=click.Path(),
              help="Where adapters and provenance are written.")
@click.option("--global-batch", required=True, type=int,
              help="Examples per optimizer step. REQUIRED, and never derived from the chip "
                   "count: it is the axis a published recipe pins.")
@click.option("--steps", required=True, type=int, help="Optimizer steps to run.")
@click.option("--objective", default="af3", show_default=True,
              help="A named objective row. `--list-objectives` prints them.")
@click.option("--recipe", default="lora", show_default=True,
              help="A named Tier-1 body. `tt-bio finetune --show-recipe` prints its source, "
                   "which is a Tier-2 program you can edit and run yourself.")
@click.option("--tokens", default=None, type=int,
              help="Crop size, for the fit check. Defaults to the dataset's own.")
@click.option("--chips", default=1, show_default=True, type=int,
              help="Width of the data-parallel axis.")
@click.option("--rank", default=8, show_default=True, type=int, help="LoRA rank.")
@click.option("--alpha", default=16.0, show_default=True, type=float, help="LoRA alpha.")
@click.option("--target", "targets", multiple=True,
              help="Regex matched against site names; repeatable. Default adapts every site "
                   "the census finds.")
@click.option("--lr", default=3e-4, show_default=True, type=float, help="Peak learning rate.")
@click.option("--warmup-steps", default=1000, show_default=True, type=int)
@click.option("--checkpoint-every", default=100, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--dry-run", is_flag=True,
              help="Answer 'will this fit and how long' and exit, WITHOUT opening a device.")
@click.option("--show-recipe", is_flag=True,
              help="Print the Tier-2 source of --recipe and exit. The escape hatch.")
@click.option("--list-objectives", is_flag=True, help="Print the objective rows and exit.")
def finetune(data, model, out_dir, global_batch, steps, objective, recipe, tokens, chips,
             rank, alpha, targets, lr, warmup_steps, checkpoint_every, seed, dry_run,
             show_recipe, list_objectives):
    """Fine-tune a shipped model with LoRA adapters.

    \b
        tt-bio finetune data/ --model protenix-v2 --out runs/a \\
            --global-batch 8 --steps 2000 --dry-run

    \b
    Progressive disclosure, if this is not enough:
      --show-recipe        the Tier-2 program this command runs, as source
      python -c "from tt_bio import train; train.finetune(...)"    Tier 1
      tt_bio.autograd + train.gradcheck                            Tier 3
    """
    from . import objectives
    from .dryrun import plan

    if list_objectives:
        for name in objectives.names():
            click.echo(objectives.objective(name))
        return
    if show_recipe:
        # Imported only in this branch. `recipes` reaches the tape, and every other path
        # through this command -- --dry-run above all -- has to answer without a device.
        from . import recipes
        click.echo(recipes.source(recipe))
        return

    # Legality, decided here and not on the card. Every one of these is a flag reading a
    # flag; none of them needs a device, which is the Tier-0 cut line holding.
    if objective not in objectives.names():
        raise click.BadParameter(f"{objective!r}; rows are {objectives.names()}",
                                 param_hint="--objective")
    # Checked against the recipe NAMES, read without importing the module: the names are
    # what a flag can be wrong about, and importing the bodies to validate a string would
    # pull the tape into a dry run.
    if recipe not in RECIPE_NAMES:
        raise click.BadParameter(f"{recipe!r}; recipes are {sorted(RECIPE_NAMES)}",
                                 param_hint="--recipe")
    if chips < 1:
        raise click.BadParameter("must be at least 1", param_hint="--chips")
    if global_batch % chips:
        raise click.BadParameter(
            f"--global-batch {global_batch} does not divide by --chips {chips}. Rounding it "
            f"would change the recipe on a box with a different chip count",
            param_hint="--global-batch")
    if steps < 1:
        raise click.BadParameter("must be at least 1", param_hint="--steps")
    if rank < 1:
        raise click.BadParameter("must be at least 1", param_hint="--rank")

    fit = plan(tokens=tokens or 256, chips=chips, global_batch=global_batch,
               frozen_trunk=True)
    click.echo(str(fit))
    if fit.verdict == "refused":
        raise click.ClickException(
            "refusing to start on a configuration measured not to fit. Lower --tokens, or "
            "wait for the crop re-measure named above")
    if dry_run:
        if not fit.measured:
            click.echo("\nnote: UNMEASURED is an answer, not an error. It means we have no "
                       "measurement for this shape and will not print a projection shaped "
                       "like one.")
        return

    # Only now does anything reach a device.
    from .catalogue import load
    forward, dataset = load(model, Path(data), tokens=tokens)
    from .lora import LoraConfig
    from .loop import finetune as run_finetune

    run = run_finetune(
        forward, dataset, out_dir=out_dir, global_batch=global_batch, steps=steps,
        objective=objective, recipe=recipe, seed=seed, lr=lr, warmup_steps=warmup_steps,
        checkpoint_every=checkpoint_every, tokens=tokens,
        lora=LoraConfig(rank=rank, alpha=alpha, targets=tuple(targets)),
        mesh=_mesh(chips))
    click.echo(str(run))
    out = Path(out_dir) / "run.json"
    out.write_text(json.dumps({"history": run.history,
                               "provenance": run.provenance.as_dict(),
                               "displacement": run.displacement}, indent=2, default=str))
    click.echo(f"wrote {out}")


def _mesh(chips: int):
    from .mesh import Mesh
    return Mesh({"dp": list(range(chips))})

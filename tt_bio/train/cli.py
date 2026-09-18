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

import ast
import json
from pathlib import Path

import click

# Models whose forward routes through `tt_bio.ops.linear`, so an adapter can attach to it.
# A name absent here is not adaptable today and the refusal says so rather than failing later
# with an empty census. Kept as a list because it is a fact about the attach work that has
# landed, not a preference -- when a model gets routed it gets added here in the same change.
ADAPTABLE = ("protenix-v2", "openfold3")

# The Tier-1 bodies, by name. Duplicated from `recipes._RECIPES` on purpose and pinned by a
# test: validating `--recipe` must not import the bodies, because importing them imports the
# tape and `--dry-run` promises not to. The test fails if the two ever disagree.
RECIPE_NAMES = ("lora",)

__all__ = ["finetune", "ADAPTABLE", "RECIPE_NAMES"]


def _echo_objectives(ctx, param, value):
    """`--list-objectives`, eager so it answers before the required arguments are checked."""
    if not value or ctx.resilient_parsing:
        return
    from . import objectives
    for name in objectives.names():
        click.echo(objectives.objective(name))
        if objectives.objective(name).doc:
            click.echo(f"    {objectives.objective(name).doc}")
    ctx.exit()


def _chips(ctx, param, value):
    """`--chips` as a count or as the chips themselves. A flag either way, so `--dry-run` decides.

    A count is the first N chips, which is the right default on an idle box and the wrong one on
    every shared box: chip 0 is routinely the busy one, and a count gives no way to say so. The
    list form takes tt-smi ids, which are NOT `/dev/tenstorrent` node numbers -- the mesh takes
    the same ids, and the launcher reads back which node each rank actually got.
    """
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    try:
        if "," in text:
            ids = tuple(int(p) for p in text.split(",") if p.strip() != "")
        else:
            n = int(text)
            if n < 1:
                raise ValueError
            ids = tuple(range(n))
    except ValueError:
        raise click.BadParameter(f"{value!r}; pass a count like 2 or chips like 0,2",
                                 param_hint="--chips") from None
    if not ids:
        raise click.BadParameter("names no chips", param_hint="--chips")
    if len(set(ids)) != len(ids):
        raise click.BadParameter(f"repeats a chip: {ids}", param_hint="--chips")
    return ids


def _recipe_text(name: str) -> str:
    """A recipe's source, read off the file rather than through ``inspect``.

    ``recipes.source()`` is the canonical API and it is what a Tier-2 user calls. This reads
    the same text without importing the module, because importing it imports the tape, and
    printing a program should not need a wheel and a card. The two are pinned equal by
    ``tests/test_train_interface.py``.
    """
    src = (Path(__file__).with_name("recipes.py")).read_text()
    tree = ast.parse(src)
    shipped = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_RECIPES":
            shipped = {ast.literal_eval(k): v.id for k, v in zip(node.value.keys,
                                                                 node.value.values)}
    if name not in shipped:
        raise click.BadParameter(f"{name!r}; recipes are {sorted(shipped)}",
                                 param_hint="--show-recipe")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == shipped[name]:
            return ast.get_source_segment(src, node)
    raise click.ClickException(f"recipes.py maps {name!r} to {shipped[name]!r}, which it does "
                               f"not define")


def _echo_recipe(ctx, param, value):
    """`--show-recipe [NAME]`. The escape hatch, and it must not need a dataset to print.

    Eager for the same reason `--help` is: asking a user for `--out` and `--global-batch`
    before it will show them the loop is a papercut on the one path that exists to make the
    next tier reachable.
    """
    if not value or ctx.resilient_parsing:
        return
    click.echo(_recipe_text(value))
    ctx.exit()


@click.command("finetune")
@click.argument("data", type=click.Path(exists=True, dir_okay=True), required=False)
@click.option("--model", type=click.Choice(ADAPTABLE),
              help="Which shipped forward to adapt.")
@click.option("--out", "out_dir", type=click.Path(),
              help="Where adapters and provenance are written.")
@click.option("--global-batch", type=int, default=None,
              help="Examples per optimizer step. REQUIRED, and never derived from the chip "
                   "count: it is the axis a published recipe pins.")
@click.option("--steps", type=int, default=None, help="Optimizer steps to run.")
@click.option("--objective", default="af3", show_default=True,
              help="A named objective row. `--list-objectives` prints them.")
@click.option("--recipe", default="lora", show_default=True,
              help="A named Tier-1 body. `tt-bio finetune --show-recipe` prints its source, "
                   "which is a Tier-2 program you can edit and run yourself.")
@click.option("--tokens", default=None, type=int,
              help="Crop size, for the fit check. Defaults to the dataset's own.")
@click.option("--chips", "chip_ids", default="1", show_default=True, callback=_chips,
              help="The data-parallel axis: a count, e.g. 2, or the chips by their tt-smi id, "
                   "e.g. 0,2. A count means the first N, which is not what you want on a box "
                   "where chip 0 is busy.")
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
@click.option("--show-recipe", is_flag=False, flag_value="lora", default=None,
              metavar="[NAME]", is_eager=True, expose_value=False,
              callback=_echo_recipe,
              help="Print a Tier-1 body as Tier-2 source and exit. The escape hatch.")
@click.option("--list-objectives", is_flag=True, is_eager=True, expose_value=False,
              callback=_echo_objectives, help="Print the objective rows and exit.")
def finetune(data, model, out_dir, global_batch, steps, objective, recipe, tokens, chip_ids,
             rank, alpha, targets, lr, warmup_steps, checkpoint_every, seed, dry_run):
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

    chips = len(chip_ids)
    if data is None:
        raise click.UsageError("DATA is required for a run. To look around without one, try "
                               "--show-recipe, --list-objectives or --help")

    # Legality, decided here and not on the card. Every one of these is a flag reading a
    # flag; none of them needs a device, which is the Tier-0 cut line holding.
    for name, value in (("--model", model), ("--out", out_dir),
                        ("--global-batch", global_batch), ("--steps", steps)):
        if value is None:
            raise click.UsageError(f"{name} is required for a run")
    if objective not in objectives.names():
        raise click.BadParameter(f"{objective!r}; rows are {objectives.names()}",
                                 param_hint="--objective")
    # Checked against the recipe NAMES, read without importing the module: the names are
    # what a flag can be wrong about, and importing the bodies to validate a string would
    # pull the tape into a dry run.
    if recipe not in RECIPE_NAMES:
        raise click.BadParameter(f"{recipe!r}; recipes are {sorted(RECIPE_NAMES)}",
                                 param_hint="--recipe")
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

    # The featuriser is resolved before anything reaches a device, so a model with no
    # training adapter registered costs a message rather than a card and a traceback.
    from .catalogue import load
    try:
        forward, dataset = load(model, Path(data), tokens=tokens)
    except NotImplementedError as exc:
        raise click.ClickException(str(exc)) from exc

    # Only now does anything reach a device.
    from .lora import LoraConfig
    from .loop import finetune as run_finetune

    run = run_finetune(
        forward, dataset, out_dir=out_dir, global_batch=global_batch, steps=steps,
        objective=objective, recipe=recipe, seed=seed, lr=lr, warmup_steps=warmup_steps,
        checkpoint_every=checkpoint_every, tokens=tokens,
        lora=LoraConfig(rank=rank, alpha=alpha, targets=tuple(targets)),
        mesh=_mesh(chip_ids))
    click.echo(str(run))
    # Every rank of a data-parallel run reaches this line, so the path is per rank: rank 0
    # keeps out_dir and the others get a subdirectory. One path shared by N writers is a
    # truncated file, not a duplicate one.
    from .launcher import out_dir as rank_dir

    out = rank_dir(out_dir) / "run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"history": run.history,
                               "provenance": run.provenance.as_dict(),
                               "displacement": run.displacement}, indent=2, default=str))
    click.echo(f"wrote {out}")


def _mesh(chip_ids):
    from .mesh import Mesh
    return Mesh({"dp": list(chip_ids)})

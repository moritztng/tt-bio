"""Tier 0: ``tt-bio train``. Every knob is a flag, and no flag is a callable.

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

The flags come in three layers. ``tt-bio train DATA --model openfold3`` is a whole run on
one chip. ``--help`` adds what a user reaches for next: steps, chips, batch, learning rate,
checkpoint interval. ``--help-all`` shows the rest, which the performance and accuracy work
used and a training run does not need. Every run writes ``status.json`` and
``progress.jsonl`` into ``--out`` so a program can follow it without parsing the log, and
running the same command again resumes it.
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
RECIPE_NAMES = ("default",)

# What the optimizer may own. Duplicated from `loop.TRAIN_MODES` for the reason RECIPE_NAMES is
# duplicated, and pinned by the same test: validating `--train` must not import the tape.
TRAIN_MODES = ("adapters", "weights")

#: Models with a shipped featuriser, so `tt-bio train` can read their data. Pinned equal to
#: `catalogue.SHIPPED` by a test; kept literal so `--help` imports nothing.
TRAINABLE = ("openfold3",)

__all__ = ["train", "ADAPTABLE", "TRAINABLE", "RECIPE_NAMES", "TRAIN_MODES"]


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


class _Layered(click.Command):
    """`--help` shows the first two layers; `--help-all` shows every option."""

    def format_options(self, ctx, formatter):
        full = ctx.meta.get("help_all", False)
        rows = [p.get_help_record(ctx) for p in self.get_params(ctx)
                if full or not getattr(p, "expert", False)]
        with formatter.section("Options"):
            formatter.write_dl([r for r in rows if r])
        if not full:
            formatter.write_paragraph()
            formatter.write_text("Precision, objective, recipe and LoRA options: --help-all")


def _help_all(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    ctx.meta["help_all"] = True
    click.echo(ctx.get_help())
    ctx.exit()


def _expert(*a, **kw):
    """An option that `--help` leaves out and `--help-all` shows."""
    def deco(f):
        f = click.option(*a, **kw)(f)
        f.__click_params__[-1].expert = True
        return f
    return deco


@click.command("train", cls=_Layered)
@click.argument("data", type=click.Path(exists=True, dir_okay=True), required=False)
@click.option("--model", type=click.Choice(TRAINABLE), help="The model to train.")
@click.option("--out", "out_dir", type=click.Path(), default=None,
              help="Run directory: checkpoints, status.json, progress.jsonl. Default "
                   "runs/<model>. Running again with the same --out resumes.")
@click.option("--steps", type=int, default=None,
              help="Optimizer steps. Default: one pass over DATA.")
@click.option("--chips", "chip_ids", default="1", show_default=True, callback=_chips,
              help="Data-parallel chips: a count, e.g. 4, or tt-smi ids, e.g. 0,2.")
@click.option("--dry-run", is_flag=True,
              help="Say whether it fits and how long a step takes, without opening a device.")
@click.option("--global-batch", type=int, default=None,
              help="Samples per optimizer step. Default: one per chip. Fix it to compare "
                   "runs across chip counts.")
@click.option("--lr", default=3e-4, show_default=True, type=float, help="Peak learning rate.")
@click.option("--warmup-steps", default=1000, show_default=True, type=int)
@click.option("--checkpoint-every", default=100, show_default=True, type=int,
              help="Steps between checkpoints. The last step is always saved.")
@click.option("--tokens", default=None, type=int,
              help="Crop size in tokens. Default: the model's first training stage, 384 for "
                   "OpenFold3.")
@click.option("--seed", default=0, show_default=True, type=int,
              help="Fixes the sample order and the crops.")
@click.option("--train", "train_mode", type=click.Choice(TRAIN_MODES), default="weights",
              show_default=True,
              help="`weights` trains the model's own weights; `adapters` trains LoRA pairs "
                   "on a frozen model (--help-all for rank and targets).")
@click.option("--help-all", is_flag=True, is_eager=True, expose_value=False,
              callback=_help_all, help="Show every option, including the expert ones.")
@_expert("--exact/--device-ops", default=None,
         help="--exact runs softmax and layer norm in float64 on the host, a diagnostic "
              "reference many times slower. --device-ops keeps them on the device, which "
              "clears the same accuracy bar. Neither: the device path, unless the crop fits "
              "only with --exact.")
@_expert("--objective", default="af3", show_default=True,
         help="A named objective row. `--list-objectives` prints them.")
@_expert("--recipe", default="default", show_default=True,
         help="A named Tier-1 body. `--show-recipe` prints its source as a Tier-2 program.")
@_expert("--rank", default=8, show_default=True, type=int, help="LoRA rank (adapters).")
@_expert("--alpha", default=16.0, show_default=True, type=float,
         help="LoRA alpha (adapters).")
@_expert("--target", "targets", multiple=True,
         help="Regex over site names, repeatable (adapters). Default: every site.")
@_expert("--show-recipe", is_flag=False, flag_value="default", default=None,
         metavar="[NAME]", is_eager=True, expose_value=False, callback=_echo_recipe,
         help="Print a Tier-1 body as Tier-2 source and exit.")
@_expert("--list-objectives", is_flag=True, is_eager=True, expose_value=False,
         callback=_echo_objectives, help="Print the objective rows and exit.")
def train(data, model, out_dir, steps, chip_ids, dry_run, global_batch, lr, warmup_steps,
          checkpoint_every, tokens, seed, train_mode, exact, objective, recipe, rank, alpha,
          targets):
    """Train a model on the chips of this machine.

    \b
        tt-bio train --model openfold3
        tt-bio train data/ --model openfold3
        tt-bio train data/ --model openfold3 --steps 1000 --chips 4

    DATA for OpenFold3 is upstream's training-set directory (pdb_training_set/ plus a
    training_cache*.json). Without DATA the run fetches upstream's 8-structure sample
    (73 MB) and trains on that. The run writes OUT/status.json, whose `status` is running,
    succeeded or failed, and OUT/progress.jsonl, one JSON line per step. Run the same
    command again to resume from the last checkpoint. docs/training.md has the rest.
    """
    from . import objectives
    from .dryrun import needs_exact, plan

    if model is None:
        raise click.UsageError("--model is required, e.g. `tt-bio train data/ --model "
                               "openfold3`")
    chips = len(chip_ids)
    global_batch = chips if global_batch is None else global_batch
    out_dir = Path(out_dir or Path("runs") / model)
    if objective not in objectives.names():
        raise click.BadParameter(f"{objective!r}; rows are {objectives.names()}",
                                 param_hint="--objective")
    if recipe not in RECIPE_NAMES:
        raise click.BadParameter(f"{recipe!r}; recipes are {sorted(RECIPE_NAMES)}",
                                 param_hint="--recipe")
    if global_batch < 1 or global_batch % chips:
        raise click.BadParameter(
            f"--global-batch {global_batch} does not divide over {chips} chips",
            param_hint="--global-batch")
    for name, value in (("--steps", steps), ("--rank", rank)):
        if value is not None and value < 1:
            raise click.BadParameter("must be at least 1", param_hint=name)

    crop = tokens or _DEFAULT_CROP[model]
    fit = plan(tokens=crop, model=model, chips=chips, global_batch=global_batch,
               frozen_trunk=train_mode == "adapters")
    click.echo(str(fit))
    if fit.verdict == "refused":
        raise click.ClickException("refusing to start on a configuration measured not to "
                                   "fit. Lower --tokens")
    if dry_run:
        return

    from . import launcher
    if data is None:
        from .openfold3 import sample_data
        data = sample_data(quiet=launcher.rank() != 0)
        click.echo(f"no DATA given, so this trains on upstream's 8-structure sample in {data}. "
                   f"Pass a training-set directory to train on your own.")
    status = _Status(out_dir, writer=launcher.rank() == 0)
    config = {"model": model, "data": str(Path(data).resolve()), "train": train_mode,
              "global_batch": global_batch, "lr": lr, "warmup_steps": warmup_steps,
              "tokens": crop, "seed": seed, "objective": objective, "recipe": recipe}
    status.claim(config, chips=list(chip_ids))
    try:
        from .catalogue import load
        forward, dataset = load(model, Path(data), tokens=tokens, seed=seed)
        steps = steps or max(1, len(dataset) // global_batch)
        status.update(steps=steps)
        if exact is None and needs_exact(model, crop):
            exact = True
            click.echo(f"{crop} tokens fits {model} only with --exact on one card, so it is "
                       f"on: a step runs many times slower. Lower --tokens for the device path.")

        from ..autograd import exact_training
        from .lora import LoraConfig
        from .loop import finetune as run_finetune

        with exact_training(bool(exact)):
            run = run_finetune(
                forward, dataset, out_dir=out_dir, global_batch=global_batch, steps=steps,
                objective=objective, train=train_mode, recipe=recipe, seed=seed, lr=lr,
                warmup_steps=warmup_steps, checkpoint_every=checkpoint_every, tokens=tokens,
                lora=LoraConfig(rank=rank, alpha=alpha, targets=tuple(targets)),
                mesh=_mesh(chip_ids), on_step=status.step, resume=True)
    except Exception as exc:                                           # noqa: BLE001
        status.fail(exc)
        raise click.ClickException(f"{type(exc).__name__}: {exc}\n(status and traceback in "
                                   f"{status.path})") from None
    click.echo(str(run))
    status.succeed(run)
    click.echo(f"wrote {status.path}")


#: The crop a run uses when --tokens is not given: each model's first training stage.
_DEFAULT_CROP = {"openfold3": 384}


class _Status:
    """``status.json`` and ``progress.jsonl`` in the run directory, for a program to follow.

    ``status`` uses JapanFold's job words: ``running``, ``succeeded``, ``failed``, with an
    error as ``{title, detail}``. Only rank 0 writes; on a data-parallel run the driver writes
    the start and the end and rank 0 the steps in between.
    """

    def __init__(self, out_dir: Path, *, writer: bool = True):
        import time
        self.dir, self.writer, self.clock = Path(out_dir), writer, time.perf_counter
        self.last = self.clock()
        self.path = self.dir / "status.json"
        self.progress = self.dir / "progress.jsonl"

    def read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _write(self, **fields) -> None:
        if not self.writer:
            return
        import time
        doc = {**self.read(), **fields,
               "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2, default=str) + "\n")
        tmp.replace(self.path)

    def claim(self, config: dict, *, chips) -> None:
        """Start or resume. A different config in the same directory is refused, not mixed."""
        import os
        import socket
        from . import launcher
        old = self.read()
        if old.get("config") and old["config"] != config:
            diff = {k: (old["config"].get(k), v) for k, v in config.items()
                    if old["config"].get(k) != v}
            raise click.ClickException(
                f"{self.dir} holds a run with other settings {diff} (was, now). Pass a new "
                f"--out, or the same settings to resume it")
        if launcher.driving():
            # One chip runs where TT_VISIBLE_DEVICES points; the ids are the launcher's only
            # when it spawns ranks, so that is what gets recorded.
            chips = chips if len(chips) > 1 else [os.environ.get("TT_VISIBLE_DEVICES", "0")]
            self._write(status="running", config=config, chips=chips, pid=os.getpid(),
                        host=socket.gethostname(), error=None, out=str(self.dir.resolve()),
                        progress=str(self.progress.resolve()))

    def update(self, **fields) -> None:
        from . import launcher
        if launcher.driving():
            self._write(**fields)

    def step(self, row: dict) -> None:
        import math
        loss, g = row.get("loss"), row.get("grad_norm")
        healthy = all(v is None or math.isfinite(float(v)) for v in (loss, g))
        now = self.clock()
        line = {k: row.get(k) for k in ("step", "loss", "lr", "grad_norm")}
        line["s"], self.last = round(row.get("s") or now - self.last, 3), now
        line["healthy"] = healthy
        if self.writer:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.progress, "a") as fh:
                fh.write(json.dumps(line, default=float) + "\n")
        self._write(step=row.get("step"), loss=loss, healthy=healthy)
        if not healthy:
            raise FloatingPointError(
                f"step {row.get('step')}: loss {loss}, gradient norm {g}. The run stops rather "
                f"than train on it; resume from the last checkpoint with a lower --lr")

    def fail(self, exc: BaseException) -> None:
        import traceback
        if self.writer:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "traceback.txt").write_text(traceback.format_exc())
        self._write(status="failed",
                    error={"title": type(exc).__name__, "detail": str(exc)})

    def succeed(self, run) -> None:
        """The summary goes in status.json; the full record, config included, in run.json."""
        prov = run.provenance.as_dict()
        if self.writer:
            (self.dir / "run.json").write_text(json.dumps(
                {"history": run.history, "provenance": prov, "dp": run.dp,
                 "displacement": run.displacement}, indent=2, default=str) + "\n")
        best, latest = run.best, run.checkpointer.latest()
        self._write(status="succeeded", step=run.history[-1]["step"] if run.history else None,
                    loss=run.loss, provenance={k: v for k, v in prov.items() if k not in ("config", "dp")},
                    dp=_dp_summary(run.dp), checkpoint=str(best["path"]) if best else None,
                    latest=str(latest) if latest else None, record=str(self.dir / "run.json"))


def _dp_summary(dp):
    """The data-parallel numbers an agent reads; the per-rank record stays in run.json."""
    keep = ("world", "nodes", "distinct_master_sha", "median_step_s", "comm_bytes",
            "median_transfer_s", "median_barrier_wait_s")
    return {k: dp.get(k) for k in keep} if dp else None


def _mesh(chip_ids):
    from .mesh import Mesh
    return Mesh({"dp": list(chip_ids)})

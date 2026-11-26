"""What `tt-bio dock` is asked to do: the options, the jobs and the output layout.

Three input forms reach the same ``DockJob`` list: one receptor on the command line, an
upstream four-column CSV, and a tt-bio YAML. Upstream has the first two; the YAML is the form
every other tt-bio model already takes, so a user who writes `tt-bio predict` input writes
this one too.

The option defaults ARE upstream's gated recipe. ``recipe.yaml`` is a verbatim copy of
`default_inference_args.yaml` at upstream 85c49b60, and :data:`RECIPE` is read from it rather
than typed out here, so a default cannot drift from the published recipe without the copy
changing. The nine low-temperature sampling constants stay off the command line (changing one
of a fitted set alone is meaningless) and are reachable through ``--config``.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml

from tt_bio.data.yaml_input import load_mapping

RECIPE_PATH = Path(__file__).with_name("recipe.yaml")

#: Upstream's `default_inference_args.yaml`, verbatim. The gated recipe (D-dfd-2).
RECIPE: dict = load_mapping(RECIPE_PATH)

#: Upstream config key -> the DockOptions field it sets. Keys absent here are sampling
#: constants carried through untouched in ``DockOptions.recipe``.
CONFIG_KEYS: dict[str, str] = {
    "samples_per_complex": "diffusion_samples",
    "inference_steps": "sampling_steps",
    "actual_steps": "actual_steps",
    "batch_size": "max_parallel_samples",
    "no_final_step_noise": "no_final_step_noise",
    "initial_noise_std_proportion": "initial_noise_std",
    "choose_residue": "init_at_residue",
    "ode": "_ode",
    "no_random": "deterministic",
    "no_model": "no_model",
    "save_visualisation": "save_trajectory",
    "limit_failures": "limit_failures",
    "gnina_minimize": "_gnina",
    "gnina_path": "gnina_path",
    "model_dir": "checkpoint",
    "confidence_model_dir": "confidence_checkpoint",
    "pocket_cutoff": "pocket_radius",
}

#: Weight generation -> (old_score_model, old_filtering_model), upstream's two booleans.
WEIGHTS: dict[str, tuple[bool, bool]] = {
    "diffdock-l": (False, True),
    "diffdock": (True, True),
}


@dataclass
class DockOptions:
    """Everything the sampler needs that is not the complex itself."""

    diffusion_samples: int = RECIPE["samples_per_complex"]
    sampling_steps: int = RECIPE["inference_steps"]
    actual_steps: int | None = RECIPE["actual_steps"]
    max_parallel_samples: int = 10          # upstream batch_size; not in the recipe file
    no_final_step_noise: bool = RECIPE["no_final_step_noise"]
    initial_noise_std: float = RECIPE["initial_noise_std_proportion"]
    init_at_residue: bool = False
    sampler: str = "sde"                    # "ode" is upstream's probability-flow sampler
    deterministic: bool = RECIPE["no_random"]
    no_model: bool = RECIPE["no_model"]
    weights: str = "diffdock-l"
    checkpoint: Path | None = None
    confidence_checkpoint: Path | None = None
    no_confidence: bool = False
    pocket: tuple[float, float, float] | None = None
    pocket_radius: float = 5.0              # evaluate.py --pocket_cutoff
    save_trajectory: bool = False
    seed: int | None = None                 # tt-bio extension; upstream uses the global RNG
    minimize: str = "none"
    gnina_path: str = "gnina"
    limit_failures: int = int(RECIPE["limit_failures"])
    #: The sampling constants, upstream's names, as they will be handed to the sampler.
    recipe: dict = field(default_factory=lambda: dict(RECIPE))

    @property
    def old_score_model(self) -> bool:
        return WEIGHTS[self.weights][0]

    @property
    def old_filtering_model(self) -> bool:
        return WEIGHTS[self.weights][1]

    @property
    def steps(self) -> int:
        """Denoising steps actually performed (upstream: actual_steps or inference_steps)."""
        return self.sampling_steps if self.actual_steps is None else self.actual_steps

    def with_config(self, path) -> "DockOptions":
        """Apply an upstream inference-args YAML, as `inference.py --config` does.

        Upstream's own `default_inference_args.yaml` is accepted verbatim, including the keys
        that are not command-line flags here: they ride along in ``recipe``.
        """
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping of inference args")
        changes: dict = {}
        for key, value in data.items():
            name = CONFIG_KEYS.get(key)
            if name == "_ode":
                changes["sampler"] = "ode" if value else "sde"
            elif name == "_gnina":
                changes["minimize"] = "gnina" if value else "none"
            elif name in ("checkpoint", "confidence_checkpoint"):
                changes[name] = Path(value) if value else None
            elif name == "pocket_radius":
                changes[name] = float(value)
            elif name:
                changes[name] = value
        if data.get("old_score_model"):
            changes["weights"] = "diffdock"
        recipe = {**self.recipe, **data}
        return replace(self, recipe=recipe, **changes)

    def as_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else v
        return out


@dataclass(frozen=True)
class DockJob:
    """One receptor and one ligand: the unit of work, and one output directory."""

    name: str
    receptor_path: Path | None = None
    receptor_sequence: str | None = None
    ligand: str | None = None               # SMILES or a path to a molecule file

    def __post_init__(self):
        if self.receptor_path is None and not self.receptor_sequence:
            raise ValueError(f"{self.name}: no receptor (give a structure file or a sequence)")
        if not self.ligand:
            raise ValueError(f"{self.name}: no ligand (give a SMILES or a molecule file)")


RECEPTOR_SUFFIXES = {".pdb", ".cif", ".mmcif", ".ent", ".pdb1"}
LIGAND_SUFFIXES = {".sdf", ".mol", ".mol2", ".pdb", ".smi", ".smiles"}
_SEQUENCE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZUO]{12,}$", re.IGNORECASE)


def looks_like_sequence(text: str) -> bool:
    """A bare protein sequence on the command line, rather than a path."""
    return bool(_SEQUENCE.match(text.strip()))


def read_csv_jobs(path) -> list[DockJob]:
    """Upstream's four-column CSV, unchanged, so an existing DiffDock CSV runs here.

    Columns: complex_name, protein_path, protein_sequence, ligand_description. An empty cell
    is None, as upstream's ``set_nones``; a missing complex_name becomes ``complex_<i>``.
    """
    path = Path(path)
    jobs: list[DockJob] = []
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if rows and "ligand_description" not in rows[0]:
        raise ValueError(f"{path.name}: not a DiffDock CSV (needs a ligand_description column; "
                         f"found {', '.join(rows[0])})")
    for i, row in enumerate(rows):
        cell = {k: (v.strip() if isinstance(v, str) else v) or None for k, v in row.items()}
        rec = cell.get("protein_path")
        jobs.append(DockJob(name=cell.get("complex_name") or f"complex_{i}",
                            receptor_path=Path(rec).expanduser() if rec else None,
                            receptor_sequence=cell.get("protein_sequence"),
                            ligand=cell.get("ligand_description")))
    if not jobs:
        raise ValueError(f"{path.name}: no rows")
    return jobs


def read_yaml_job(path, options: DockOptions) -> tuple[list[DockJob], DockOptions]:
    """tt-bio's YAML form: the `sequences:` block every model takes, plus `docking:`.

    Returns the job and the options the `docking:` block changed. Keys diffdock cannot
    honour are caught by ``tt_bio.capabilities``, not here, so the refusal reads the same as
    it does for a fold.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict) or "sequences" not in data:
        raise ValueError(f"{Path(path).name}: not a tt-bio job YAML (no `sequences:` block)")
    receptor_path = receptor_seq = ligand = None
    for entry in data["sequences"]:
        (kind, body), = entry.items()
        if kind == "protein":
            if body.get("file"):
                receptor_path = Path(body["file"]).expanduser()
            elif body.get("sequence"):
                receptor_seq = body["sequence"]
        elif kind == "ligand":
            ligand = body.get("smiles") or body.get("file") or (
                f"CCD_{body['ccd']}" if body.get("ccd") else None)
    name = data.get("name") or Path(path).stem
    opts = _apply_docking_block(options, data.get("docking") or {}, Path(path).name)
    return [DockJob(name=name, receptor_path=receptor_path,
                    receptor_sequence=receptor_seq, ligand=ligand)], opts


#: What the YAML `docking:` block may set. Exactly the command line, no second config system.
DOCKING_KEYS = ("diffusion_samples", "sampling_steps", "actual_steps", "max_parallel_samples",
                "no_final_step_noise", "initial_noise_std", "init_at_residue", "sampler",
                "deterministic", "no_model", "weights", "no_confidence", "pocket",
                "pocket_radius", "save_trajectory", "seed", "minimize", "limit_failures")


def _apply_docking_block(options: DockOptions, block: dict, where: str) -> DockOptions:
    unknown = [k for k in block if k not in DOCKING_KEYS]
    if unknown:
        raise ValueError(f"{where}: `docking:` has no {', '.join(sorted(unknown))}. "
                         f"Known keys: {', '.join(DOCKING_KEYS)}")
    changes = dict(block)
    if "pocket" in changes and changes["pocket"] is not None:
        changes["pocket"] = parse_pocket(changes["pocket"])
    if changes.get("sampler") not in (None, "sde", "ode"):
        raise ValueError(f"{where}: sampler must be sde or ode")
    if changes.get("weights") not in (None, *WEIGHTS):
        raise ValueError(f"{where}: weights must be one of {', '.join(WEIGHTS)}")
    return replace(options, **changes)


def parse_pocket(value) -> tuple[float, float, float]:
    """`--pocket x,y,z` or a YAML `[x, y, z]`. One concept, one flag (FEATURE-PLAN)."""
    parts = value if isinstance(value, (list, tuple)) else str(value).replace(" ", "").split(",")
    if len(parts) != 3:
        raise ValueError(f"--pocket takes three coordinates (x,y,z), got {value!r}")
    return tuple(float(p) for p in parts)


def read_jobs(data, options: DockOptions) -> tuple[list[DockJob], DockOptions]:
    """Resolve the positional argument into jobs: CSV, YAML, directory, or one receptor."""
    path = Path(data).expanduser()
    if path.is_dir():
        jobs: list[DockJob] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in (".yaml", ".yml"):
                more, options = read_yaml_job(child, options)
                jobs += more
        if not jobs:
            raise ValueError(f"{path}: no .yaml job files in this directory")
        return jobs, options
    if path.suffix.lower() == ".csv":
        return read_csv_jobs(path), options
    if path.suffix.lower() in (".yaml", ".yml"):
        return read_yaml_job(path, options)
    return [], options      # a receptor; the caller pairs it with --ligand and --name

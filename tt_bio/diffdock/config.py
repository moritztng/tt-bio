"""DiffDock-L hyperparameters, read from the released checkpoint and pinned.

Two things here are easy to get wrong and both are fatal:

* **The constructor defaults are not the released model.** ``models/cg_model.py`` defaults to
  ns=16, nv=4, two conv layers; the v1.1 score checkpoint is ns=48, nv=10, eleven conv layers
  plus three protein-embedding layers. Anything sized off the defaults is sized wrong by an
  order of magnitude.
* **The score network is the coarse-grained model, not the all-atom one.** The score
  checkpoint sets ``all_atoms: false`` and ``old_score_model`` is false, so upstream builds
  ``CGModel``. The confidence checkpoint sets ``all_atoms: true`` with ``old_filtering_model:
  true``, so it builds ``AAOldModel``. They are different architectures with different
  receptor graphs, and the confidence model's receptor graph carries explicit atoms.

So :func:`load` reads ``model_parameters.yml`` out of the checkpoint directory and refuses a
file that disagrees with the pin below, rather than quietly running a model we did not port.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tt_bio.diffdock.irreps import (
    Irreps,
    irrep_seq,
    spherical_harmonics_irreps,
    weight_numel,
)

# The ESM-2 checkpoint upstream embeds receptors with (utils/inference_utils.py).
# tt-bio already runs it: see tt_bio.diffdock.esm, which reuses tt_bio.saprot.load_esm2.
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
ESM2_DIM = 1280
ESM2_TRUNCATION = 1022          # upstream's per-chain cap, enforced on both sides


#: ``CGModel.__init__``'s own default for ``rec_max_radius``. ``utils/utils.get_model`` passes
#: ``lig_max_radius=args.max_radius`` and ``cross_max_distance=args.cross_max_distance`` but
#: never passes ``rec_max_radius``, so every released checkpoint was built with this value --
#: independently of ``receptor_radius``, which is the dataset's kNN radius and a different
#: parameter. The checkpoint's own ``rec_distance_expansion.offset`` buffer spans 0..30.
_REC_MAX_RADIUS_DEFAULT = 30.0


@dataclass(frozen=True)
class DiffDockConfig:
    """What the port needs from ``model_parameters.yml``, in upstream's names."""

    name: str
    all_atoms: bool
    old_model: bool
    ns: int
    nv: int
    num_conv_layers: int
    num_prot_emb_layers: int
    sh_lmax: int
    use_second_order_repr: bool
    reduce_pseudoscalars: bool
    embed_also_ligand: bool
    differentiate_convolutions: bool
    batch_norm: bool
    no_torsion: bool
    scale_by_sigma: bool
    odd_parity: bool
    tp_weights_layers: int
    sigma_embed_dim: int
    distance_embed_dim: int
    cross_distance_embed_dim: int
    embedding_type: str
    embedding_scale: int
    lig_max_radius: float
    #: Sets the centres of ``rec_distance_expansion`` only. See ``_REC_MAX_RADIUS_DEFAULT``.
    rec_max_radius: float
    c_alpha_max_neighbors: int
    atom_radius: float
    atom_max_neighbors: int
    cross_max_distance: float
    dynamic_max_cross: bool
    center_max_distance: float
    crop_beyond: float | None
    tr_sigma_min: float
    tr_sigma_max: float
    rot_sigma_min: float
    rot_sigma_max: float
    tor_sigma_min: float
    tor_sigma_max: float
    in_lig_edge_features: int = 4
    num_confidence_outputs: int = 1
    parallel_aggregators: tuple[str, ...] = ()

    # -- derived -----------------------------------------------------------------
    @property
    def sh_irreps(self) -> Irreps:
        return spherical_harmonics_irreps(self.sh_lmax)

    @property
    def irrep_seq(self) -> list[str]:
        return irrep_seq(self.ns, self.nv, self.use_second_order_repr, self.reduce_pseudoscalars)

    @property
    def faster_tensor_product(self) -> bool:
        """Upstream's ``faster=`` flag for every conv layer it builds.

        ``FasterTensorProduct`` asserts its spherical harmonics are ``1x0e+1x1o``, so it is
        reachable only at ``sh_lmax == 1``. The released DiffDock-L runs at ``sh_lmax == 2``,
        which makes that path **dead code for these weights**: every tensor product in the
        shipped model is ``o3.FullyConnectedTensorProduct``. Nothing in the port needs to
        implement the faster path to reach parity on the released checkpoint.
        """
        return self.sh_lmax == 1 and not self.use_second_order_repr

    def conv_edge_groups(self, index: int) -> int:
        """Edge groups of conv layer *index*, counting from the first conv layer.

        The groups are contiguous slices of one concatenated edge list, each with its own FC
        weight block. CGModel splits into 4 (lig-lig, lig<-rec, rec-rec, rec<-lig) and the
        last layer into 2, because only ligand nodes are read afterwards. AAModel splits into
        9 and 3. The split is not cosmetic: a wrong group count loads the wrong FC weights.
        """
        if not self.differentiate_convolutions:
            return 1
        last = index == self.num_conv_layers - 1
        if self.all_atoms:
            return 3 if last else 9
        return 2 if last else 4

    def layer_plan(self) -> list[dict]:
        """Every tensor-product conv layer the checkpoint contains, in load order.

        Each entry carries what the port must size: the irreps either side, the number of
        per-edge weights the FC block emits, the FC block's input width and its edge-group
        count. Verified against the released score checkpoint: all 17 ``weight_numel`` values
        match the shipped ``fc.*.<last>.weight`` row counts exactly.
        """
        seq, sh, ns = self.irrep_seq, str(self.sh_irreps), self.ns
        last = len(seq) - 1
        plan: list[dict] = []

        def add(name, i, groups, n_edge_features, hidden):
            a, b = seq[min(i, last)], seq[min(i + 1, last)]
            plan.append(dict(name=name, in_irreps=a, out_irreps=b, edge_groups=groups,
                             n_edge_features=n_edge_features, hidden_features=hidden,
                             weight_numel=weight_numel(a, sh, b), residual=True))

        for i in range(self.num_prot_emb_layers):
            add(f"rec_emb_layers.{i}", i, 1 if self.all_atoms is False else 4, 3 * ns, 3 * ns)
        if self.embed_also_ligand:
            for i in range(self.num_prot_emb_layers):
                add(f"lig_emb_layers.{i}", i, 1, 3 * ns, 3 * ns)
        for j in range(self.num_conv_layers):
            add(f"conv_layers.{j}", self.num_prot_emb_layers + j, self.conv_edge_groups(j),
                3 * ns, 3 * ns)

        if not self.all_atoms:
            out_last = seq[min(self.num_prot_emb_layers + self.num_conv_layers, last)]
            final_out = "2x1o + 2x1e" if not self.odd_parity else "1x1o + 1x1e"
            plan.append(dict(name="final_conv", in_irreps=out_last, out_irreps=final_out,
                             edge_groups=1, n_edge_features=2 * ns, hidden_features=2 * ns,
                             weight_numel=weight_numel(out_last, str(self.sh_irreps), final_out),
                             residual=False))
            if not self.no_torsion:
                # The torsion conv's "spherical harmonics" are the full tensor product of the
                # edge sh with the 2e bond direction, so its sh irreps are neither the 0e+1o+2e
                # sequence nor an input the irrep_seq knows about.
                tor_sh = _full_tensor_product_irreps(self.sh_irreps, Irreps.parse("1x2e"))
                tor_out = f"{ns}x0o + {ns}x0e" if not self.odd_parity else f"{ns}x0o"
                plan.append(dict(name="tor_bond_conv", in_irreps=out_last, out_irreps=tor_out,
                                 edge_groups=1, n_edge_features=3 * ns, hidden_features=3 * ns,
                                 weight_numel=weight_numel(out_last, str(tor_sh), tor_out),
                                 residual=False, sh_irreps=str(tor_sh)))
        return plan


def _full_tensor_product_irreps(a: Irreps, b: Irreps) -> Irreps:
    """``o3.FullTensorProduct(a, b).irreps_out``.

    e3nn collects the paths in enumeration order and then **sorts** the result; ``Irrep`` is a
    ``tuple`` there, so the sort is ``(l, p)`` ascending -- odd before even at every l. That
    sorted layout is what ``tor_bond_conv``'s weights were trained against, so anything
    reading this string to lay out a kernel needs the sort, not the enumeration order.
    Verified against e3nn 0.5.1; see tt_bio/diffdock/full_tensor_product.py.
    """
    from tt_bio.diffdock.irreps import Irrep

    out = []
    for ta in a:
        for tb in b:
            for l in range(abs(ta.l - tb.l), ta.l + tb.l + 1):
                out.append(Irrep(ta.mul * tb.mul, l, ta.p * tb.p))
    return Irreps(tuple(sorted(out, key=lambda t: (t.l, t.p))))


# ---------------------------------------------------------------------------
# The pin: what the v1.1 release actually contains
# ---------------------------------------------------------------------------
# Read from ~/scratch/dfd/ckpt/extracted/{score,confidence}_model/model_parameters.yml on
# 2026-11-26, out of diffdock_models.zip from the **v1.1** release
# (sha256 5a95b6a1555be47ab1d6f0a8ffd25152f7fe32f5956005bb821e13e7a37d4a3d). The
# releases/latest URL in upstream's inference.py 404s: v1.1.2 and v1.1.3 ship no assets.
SCORE_PIN = dict(
    all_atoms=False, ns=48, nv=10, num_conv_layers=11, num_prot_emb_layers=3, sh_lmax=2,
    use_second_order_repr=False, reduce_pseudoscalars=True, embed_also_ligand=True,
    tp_weights_layers=2, sigma_embed_dim=64, distance_embed_dim=64,
    cross_distance_embed_dim=64, tr_sigma_max=19.0, embedding_scale=1000,
)
CONFIDENCE_PIN = dict(
    all_atoms=True, ns=24, nv=6, num_conv_layers=5, num_prot_emb_layers=0, sh_lmax=2,
    use_second_order_repr=False, reduce_pseudoscalars=False, embed_also_ligand=False,
    tp_weights_layers=2, sigma_embed_dim=32, distance_embed_dim=32,
    cross_distance_embed_dim=32, tr_sigma_max=34.0, embedding_scale=10000,
)

SCORE_PARAM_COUNT = 30_229_368       # tensors in best_ema_inference_epoch_model.pt
CONFIDENCE_PARAM_COUNT = 4_778_902   # tensors in best_model_epoch75.pt


def load(model_dir: str | Path, *, old_model: bool, name: str = "") -> DiffDockConfig:
    """Read ``<model_dir>/model_parameters.yml`` into a :class:`DiffDockConfig`.

    *old_model* is upstream's ``--old_score_model`` / ``--old_filtering_model``: it selects
    the pre-DiffDock-L architecture and is not recorded in the yaml, so the caller passes it
    from the inference args (the default recipe sets it false for score, true for confidence).
    """
    from tt_bio.data.yaml_input import load_mapping

    path = Path(model_dir) / "model_parameters.yml"
    raw = load_mapping(path)
    agg = raw.get("parallel_aggregators") or ""
    cfg = DiffDockConfig(
        name=name or str(raw.get("run_name", Path(model_dir).name)),
        all_atoms=bool(raw.get("all_atoms", False)),
        old_model=old_model,
        ns=int(raw["ns"]), nv=int(raw["nv"]),
        num_conv_layers=int(raw["num_conv_layers"]),
        num_prot_emb_layers=int(raw.get("num_prot_emb_layers", 0)),
        sh_lmax=int(raw.get("sh_lmax", 2)),
        use_second_order_repr=bool(raw.get("use_second_order_repr", False)),
        reduce_pseudoscalars=bool(raw.get("reduce_pseudoscalars", False)),
        embed_also_ligand=bool(raw.get("embed_also_ligand", False)),
        # upstream stores the negation; get_model reads args.differentiate_convolutions,
        # which parsing.py fills from `not no_differentiate_convolutions`.
        differentiate_convolutions=not bool(raw.get("no_differentiate_convolutions", False)),
        batch_norm=not bool(raw.get("no_batch_norm", False)),
        no_torsion=bool(raw.get("no_torsion", False)),
        scale_by_sigma=bool(raw.get("scale_by_sigma", True)),
        odd_parity=bool(raw.get("odd_parity", False)),
        tp_weights_layers=int(raw.get("tp_weights_layers", 2)),
        sigma_embed_dim=int(raw["sigma_embed_dim"]),
        distance_embed_dim=int(raw["distance_embed_dim"]),
        cross_distance_embed_dim=int(raw["cross_distance_embed_dim"]),
        embedding_type=str(raw.get("embedding_type", "sinusoidal")),
        embedding_scale=int(raw.get("embedding_scale", 10000)),
        lig_max_radius=float(raw["max_radius"]),
        # NOT receptor_radius. That is the dataset's kNN preprocessing radius (15 A here);
        # the model's own rec_max_radius is a separate constructor argument that
        # utils/utils.get_model never passes, so it keeps CGModel's default of 30. It sets
        # the centres of rec_distance_expansion, and the released checkpoint stores those
        # centres spanning 0..30, which is the proof. Reading receptor_radius here made the
        # receptor Gaussians 4x too narrow.
        rec_max_radius=_REC_MAX_RADIUS_DEFAULT,
        c_alpha_max_neighbors=int(raw.get("c_alpha_max_neighbors") or 0),
        atom_radius=float(raw.get("atom_radius") or 0),
        atom_max_neighbors=int(raw.get("atom_max_neighbors") or 0),
        cross_max_distance=float(raw["cross_max_distance"]),
        dynamic_max_cross=bool(raw.get("dynamic_max_cross", False)),
        center_max_distance=float(raw.get("center_max_distance", 30)),
        crop_beyond=None if raw.get("crop_beyond") is None else float(raw["crop_beyond"]),
        tr_sigma_min=float(raw["tr_sigma_min"]), tr_sigma_max=float(raw["tr_sigma_max"]),
        rot_sigma_min=float(raw["rot_sigma_min"]), rot_sigma_max=float(raw["rot_sigma_max"]),
        tor_sigma_min=float(raw["tor_sigma_min"]), tor_sigma_max=float(raw["tor_sigma_max"]),
        num_confidence_outputs=(len(raw["rmsd_classification_cutoff"]) + 1
                                if isinstance(raw.get("rmsd_classification_cutoff"), list) else 1),
        parallel_aggregators=tuple(agg.split()) if agg else (),
    )
    pin = CONFIDENCE_PIN if cfg.all_atoms else SCORE_PIN
    bad = {k: (v, getattr(cfg, k)) for k, v in pin.items() if getattr(cfg, k) != v}
    if bad:
        raise ValueError(
            f"{path}: not the pinned DiffDock-L checkpoint. "
            + ", ".join(f"{k}: expected {w}, got {g}" for k, (w, g) in sorted(bad.items()))
            + ". Refusing to load: every shape, parity bar and roofline in this port is "
              "derived from the pinned values."
        )
    return cfg

# OpenDDE port

Resume anchor for porting OpenDDE onto Tenstorrent inside tt-bio, following the
same playbook as the Protenix-v2 / OpenFold3 ports (skill
`port-bio-model-to-tenstorrent`). Status is a **plan + identity/redundancy
analysis**; on-device implementation and parity numbers are **blocked on
hardware** (see [Status](#status)).

## Identity (re-verified 2026-07-12)

- **What:** OpenDDE = "Open Drug Discovery Engine", Aureka AI Research. Apache-2.0.
  Repo `github.com/aurekaresearch/OpenDDE`, weights
  `huggingface.co/aurekaresearch/OpenDDE`, paper arXiv:2607.03787 ("Folding,
  Reasoning, and Scaling with Open-source Drug Discovery Engine", v1 4 Jul 2026).
- **Scale:** 655M trainable params (paper Fig. 4 footnote:
  `2.04e10 tokens x 655M ≈ 1.33e19` training-cost estimate).
- **Family:** all-atom AF3-lineage, explicitly "building on recent open co-folding
  systems such as Protenix-v1 and OpenFold3"; benchmarked against AlphaFold3,
  Chai-1, Boltz-1, OpenFold3, Protenix-v1/v2, ESMFold2. Models proteins, nucleic
  acids, ligands, ions and modified residues in one all-atom system.
- **Release scope — important:** the released **preview is co-folding
  (structure-prediction) only**. Key Contribution #2 states the unified
  architecture "currently focuses on structure prediction, but is designed to
  support *future* de novo molecular design, affinity prediction, and other
  structure-conditioned modules." So the masked fold/design duality is a *roadmap*
  property of the architecture, not something the shipped checkpoints do. This
  corrects the task brief's premise that the release does both fold and design.
- **Checkpoints:** `opendde.pt` (general) and `opendde_abag.pt` (antibody-antigen
  tuned). CLI verbs: `opendde pred | json | msa | mt | prep | doctor`; model id
  `opendde_v1`. Preview caveat on the repo: "predictions are not guaranteed to be
  reproducible across releases" — relevant to the parity metric (see below).
- **Headline result:** best open-model antibody-antigen co-folding. Rank-based
  DockQ success: PXMeter-AB 51.0%, FoldBench-AB 70.0%, 2026ARK-AB 66.4%; oracle
  65.9 / 81.9 / 80.1. Claims "IsoDDE-level" (Isomorphic's closed engine) accuracy.
- **Novel block:** "atomic latent reasoning" — a latent refinement over
  biomolecular tokens *before* all-atom structure generation, on top of an
  otherwise AF3/Protenix-style trunk + atom-diffusion stack. `Fold-CP` is
  training/inference context-parallelism (a data-parallel concern, not a compute
  block to port).

## Redundancy verdict: complementary, with honest overlap

tt-bio already has `boltz2` (co-folding + affinity) and `boltzgen` (design). Read
against the actual release:

- **Overlaps** Boltz-2 / Protenix-v2 for *general* co-folding — same AF3 family,
  same task. It does **not** add a second design stack: design is roadmap, not in
  the preview, so there is no overlap with (or replacement of) `boltzgen`.
- **Genuinely additive** on one axis that matters: **antibody-antigen accuracy**.
  OpenDDE leads every open model on the AB benchmarks above and ships a dedicated
  `opendde_abag.pt`. Boltz-2 and Protenix-v2 are materially weaker there (Fig. 2).
  Ab-Ag is the single most requested therapeutic co-folding regime, so a
  best-in-class open AB checkpoint on-device is a real capability gain, not a
  duplicate.
- All-atom nucleic-acid coverage is **not** unique here (Protenix-v2 already
  covers NA), so that is not the differentiator — the AB strength is.

**Conclusion:** worth porting, scoped as a co-folding model specialised for
antibody-antigen (`tt-bio predict --model opendde`), not as a fold+design engine.
Revisit the design/affinity modules only if/when Aureka ships them.

## Architecture → tt-bio primitive mapping

OpenDDE is AF3-family, so it maps onto the primitives already validated for
Protenix-v2 in `tt_bio/tenstorrent.py` — reuse, do not duplicate:

| OpenDDE block | Reuse from tt-bio | Notes |
|---|---|---|
| Input / atom featurization | `protenix.py` AtomFeaturization + AtomAttentionEncoder | AF3 atom encoder; re-verify feature spec vs OpenDDE config |
| MSA module | `tenstorrent.MSA` / `MSALayer` / `OuterProductMean` / `PairWeightedAveraging` | MSA-dependent like boltz2/protenix-v2 |
| Trunk (Pairformer) | `tenstorrent.Pairformer` / `PairformerLayer` / `TriangleMultiplication` / `TriangleAttention` / `AttentionPairBias` / `Transition` | block count/dims per OpenDDE config |
| **Atomic latent reasoning** | **new** (small, likely a Miniformer/`Miniformer`-shaped latent refiner) | the one genuinely-new module; port + PCC-gate first |
| Diffusion structure module | `tenstorrent.DiffusionTransformer` + `protenix.py` diffusion atom encoder/decoder, AdaLN, ConditionedTransitionBlock | EDM-style sampler as in protenix |
| Confidence head | `protenix.py` ConfidenceHead | pae/pde/plddt/resolved |

Weight loading uses the shared `Module` / `WeightScope` / `weight_cache`
machinery; remap OpenDDE `opendde.pt` names via the ttnn weight-remap approach
(skill `ttnn-weight-remap`), as Protenix-v2 does.

## Port plan (phases, playbook order)

0. Vendor inference-only deps under `tt_bio/_vendor/opendde/`; pin the checkpoint
   source (HF `aurekaresearch/OpenDDE`), mirror to the tt-bio artifact bucket if
   license permits (Apache-2.0 → yes).
1. Random-weight reference harness (skill `ttnn-port-parity-methodology`):
   per-module PCC > 0.98 vs the vendored torch reference, component by component,
   in the table order above. Gate the **atomic-latent-reasoning** block first
   since it is the only novel compute.
2. Assemble the full pipeline (atom encoder → trunk → latent reasoning →
   diffusion → confidence), reusing `protenix.py`'s assembly as the template.
3. Real-weight load + on-device fold to PDB/mmCIF; `--fast` mode and multi-card
   `--devices` fanout via the existing predict scheduler (memory
   `predict-multicard-already-exists` — predict already fans out; just wire the
   model in, do not add a new fanout path).
4. CLI: extend `tt-bio predict --model` choices with `opendde` (and optionally an
   `opendde-abag` alias selecting `opendde_abag.pt`), matching the protenix-v2
   wiring in `tt_bio/main.py`. Co-folding → `predict` (not `gen`), because the
   release has no design mode.
5. One unified README section (memory `readme-audience-bio`): user-facing only,
   no kernel/L1/tile detail; link here for internals.

## Accuracy gate

- **Metric:** Ca-RMSD vs ground truth for the fold/co-folding path
  (`scripts/release_gate.py` method), plus **DockQ on antibody-antigen**
  complexes, since AB interface quality is the whole reason to add this model and
  is exactly what the paper reports (PXMeter-AB / FoldBench-AB / 2026ARK-AB).
- **No designability gate.** `scripts/boltzgen_designability.py` /
  `docs/boltzgen-designability.md` is for the design path, which OpenDDE's release
  does not have. Using it here would be measuring a mode that does not exist.
- **Stochasticity:** diffusion sampler is seed-stochastic and the repo warns
  outputs are not reproducible across releases, so parity is per-target
  Ca-RMSD/DockQ within sample variance (as for Boltz-2/Protenix-v2), not bit-exact.

## Status

- Identity, redundancy, architecture mapping, and gate choice: **done** (this doc).
- Implementation, per-module PCC, and end-to-end accuracy: **blocked, no numbers
  yet.** The assigned host qb2 (tt-quietbox2) — which holds this task's worktree
  and card — was powered off and un-wakeable on 2026-07-12 (WiFi desktop, WoL from
  poweroff needs ethernet physically connected; two automated wake attempts plus
  one manual attempt all failed while qb1 stayed reachable). On-device work cannot
  start until qb2 is physically powered on. Do **not** relocate the port to qb1
  (workspace isolation; qb1 saturated). No PCC/accuracy numbers are reported
  because none have been measured — nothing here is estimated or fabricated.

**Next action when qb2 is back:** phase 0 (vendor + checkpoint) then phase 1
starting with the atomic-latent-reasoning PCC gate.

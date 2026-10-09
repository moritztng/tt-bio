# Constraint-guided sampling (OpenDDE)

OpenDDE 1.2.0 added Test-Time Structure-Space Search: if you know part of an antibody-antigen
interface, you can give it to the sampler and the antibody is moved as a rigid body until the
prediction agrees. The weights and the trunk are untouched. tt-bio runs the same algorithm for
`--model opendde` and `--model opendde-abag`.

```bash
tt-bio predict examples/tfg/1a14_contact.yaml --model opendde-abag --use_tfg_guidance
```

## Examples

`examples/tfg/` holds upstream's four worked cases, complexes the unguided prediction gets
wrong, each as `<pdb>_unconstrained.yaml`, `<pdb>_contact.yaml` (four residue pairs) and
`<pdb>_pocket.yaml` (four epitope residues). The constraints are read off the deposited
structure, so they are correct; your own come from experiments.

| PDB | Antibody (movable) | Antigen |
|---|---|---|
| 1a14 | Fv: H, L | N |
| 9lh2 | VHH: C (copy 1 of C, D) | A, B |
| 9sat | Fab: A, B | C |
| 9xqn | Fab: B, C | A |

9lh2 and 9sat carry N-linked glycans upstream; the examples leave them out.

## The `constraint:` block

Put upstream's `constraint` field at the top level of the YAML input. Its content is the
same as in upstream's JSON, so a constraint written for OpenDDE works unchanged.

```yaml
version: 1
sequences:
  - protein: {id: [H, L], sequence: ...}   # entity 1: copy 1 is H, copy 2 is L
  - protein: {id: N, sequence: ...}        # entity 2
constraint:
  movable_chains: [H, L]
  contact:
    - {entity1: 2, copy1: 1, position1: 248, atom1: CA,
       entity2: 1, copy2: 1, position2: 93, atom2: CA,
       min_distance: 3.5, max_distance: 8.0}
```

`entity` is the 1-based index of an entry of `sequences`, `copy` the 1-based position of the
chain in that entry's `id` list, and `position` the 1-based residue. `atom` defaults to `CA`
on protein. Distances are in Angstrom and default to 3.5 and 8.0.

A pocket request names antigen residues instead of pairs:

```yaml
constraint:
  movable_chains: [H, L]
  epitope:
    residues:
      - {entity: 2, copy: 1, position: 248, residue: P}
    paratope: all        # or cdr
    min_fraction: 0.5
```

An epitope residue counts as reached when one of its heavy atoms is within 5 A of a heavy
atom of a movable chain; the request is met when `ceil(min_fraction * n)` residues are
reached. `contact` and `epitope` cannot be combined in one input. `movable_chains` is
required for an epitope, and for contacts unless the complex has exactly two chains.

## What `--use_tfg_guidance` does

- At steps 100, 103, ..., 187 of the 200 diffusion steps the denoiser's clean estimate is
  refined, searched for samples that still miss the request, and refined again; the sampler
  continues from the corrected estimate.
- From step 190 the sampled state gets a coarse pose search at 190 and at the last step and
  a refinement at every even step and at the last step.
- Refinement is gradient descent on `0.5 * ||violations||^2 + 10 * clash` over rigid motions
  of the movable chains. A move that creates a severe atomic overlap or does not lower the
  energy is refused, and a sample that already meets the request is never moved.
- Every step also applies OpenDDE's physics restraints (steric clash, bond geometry,
  chirality, planarity) to the clean estimate. This part runs for any input, constrained or
  not, exactly as upstream's flag does.

Without the flag nothing changes: the fold is the same as before, bit for bit. Contact pairs
given without the flag are ignored with a warning; an epitope without the flag is an error,
as upstream.

Contact constraints are strict: a wrong pair can pull the antibody off the true interface.
A pocket constraint with `min_fraction` below 1 tolerates a few wrong residues.

## Several constraints on one input

`--trunk_cache DIR` keeps each fold's trunk output in `DIR`. Inputs that differ only in
their `constraint:` (and the unconstrained input) then compute the trunk once. The key
covers every model input, the recycle count, the checkpoint and `--fast`, so a changed input
never reuses a stale trunk.

## Python

```python
from tt_bio.tfg.guidance import Guidance
coords = model.fold(feats, n_step=200, n_sample=5, guidance=Guidance(guidance_feats))
```

`tt_bio.tfg.features` builds `guidance_feats` from a validated constraint and the fold's
features (`validate_constraint`, `constraint_features`, `geometry_features`).

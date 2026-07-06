---
name: japanfold
description: >-
  Predict 3D biomolecular structures and binding affinity (Boltz-2, ESMFold2,
  Protenix) and design de-novo binders/proteins (BoltzGen) via the JapanFold
  API, running on Tenstorrent accelerators. Use for protein/complex structure
  prediction, protein–ligand affinity, nanobody/antibody/peptide/miniprotein
  binder design, and folding a sequence into a PDB/mmCIF structure.
when_to_use: >-
  When the user wants to fold a protein or complex, predict a structure from a
  sequence or FASTA, estimate protein–ligand binding affinity, or design binders
  against a target. Also when a workflow needs 3D structures or design candidates
  as inputs to downstream analysis.
allowed-tools:
  - Bash(japanfold *)
  - Bash(pip install *)
  - Bash(pipx install *)
  - Bash(cat *)
  - Bash(ls *)
---

# JapanFold: structure prediction & binder design

JapanFold runs Boltz-2 / ESMFold2 / Protenix (structure + affinity prediction)
and BoltzGen (binder design) on Tenstorrent hardware, behind an async HTTP API.
You drive it with the dependency-free `japanfold` CLI. Jobs take minutes; the
CLI's `--wait` / `download` commands **block in the foreground until done and
then download results** — run them as a normal long-running command. Never
background them with `&` or `nohup`.

## 1. Ensure the CLI is installed and authenticated

```bash
japanfold --version || curl -fsSL https://install.japanfold.com/install.sh | sh
```

Authentication is via an API key. In a sandboxed/agent environment the key is
provided as the `JAPANFOLD_API_KEY` environment variable — check it is set:

```bash
japanfold auth status
```

If it is not set, ask the user for their JapanFold API key (from
https://japanfold.com/account) and export it as `JAPANFOLD_API_KEY`. Do not hard-code
or echo the key.

## 2. Predict a structure

Fold a single sequence (waits, then writes results to `./out/<name>/`):

```bash
japanfold predict --sequence MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ --model boltz2 --name mytarget --wait --out ./out --json
```

For a complex, protein–ligand affinity, multiple chains, or constraints, write a
FASTA or Boltz YAML file and pass it as the input:

```bash
japanfold predict complex.yaml --model boltz2 --wait --out ./out --json
```

- Models: `boltz2` (default; MSA + ligands + affinity), `esmfold2`,
  `esmfold2-fast` (single-sequence, fast), `protenix-v2`.
- Useful flags: `--use-msa-server` (on by default for Boltz-2), `--fast`,
  `--diffusion-samples N`, `--recycling-steps N`, `--output-format cif|pdb`.
- Run `japanfold models` to see all models, design protocols, parameters and limits.

## 3. Design binders (BoltzGen)

Write a YAML design spec (target + what to design), then:

```bash
japanfold design spec.yaml --protocol nanobody-anything --num-designs 10 --wait --out ./out --json
```

Protocols: `protein-anything`, `peptide-anything`, `nanobody-anything`,
`antibody-anything`, `protein-small_molecule`, `protein-redesign`.

## 4. Submit-and-poll separately (optional)

If you'd rather not block one command, submit first and poll later:

```bash
JOB=$(japanfold predict target.fasta --model boltz2 --name t1)   # prints the job id
japanfold jobs get "$JOB" --json                                  # poll status
japanfold download "$JOB" --out ./out                             # waits + downloads when ready
```

`download` is resume-safe: re-running after a completed download is a no-op.

## 5. What you get back

For a **prediction**, `./out/<name>/` contains predicted structures under
`structures/` (`.cif`/`.pdb`), `results.json` (per-target confidence/affinity
scores), plus `job.json` and `results.json` manifests. For a **design**, it
contains the ranked designs (`final_ranked_designs/`) with a metrics CSV and the
top-ranked structures. Read `results.json` to report scores; open the `.cif`
files for the 3D structures.

## Notes

- All commands accept `--json` for machine-readable output — prefer it when you
  need to parse status, scores, or the output directory.
- `--base-url` (or `JAPANFOLD_BASE_URL`) points the CLI at a specific deployment
  (e.g. an on-prem JapanFold server); it defaults to https://japanfold.com.
- If a job fails, `japanfold logs <job_id>` prints the run log for diagnosis.

import json, sys
r = json.load(open(sys.argv[1]))
r = r[0] if isinstance(r, list) else r
for k in ('n_tokens', 'n_residues', 'n_chains', 'n_atoms', 'msa', 'msa_depth',
          'runtime_s', 'plddt', 'ptm', 'samples'):
    print(f'  {k}: {r.get(k)}')

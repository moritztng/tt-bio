"""Check the accepted designs before any timing measured beside them is quoted.

    python validate_designs.py results/pdl1 --binder-length 96 --report designs.json

A timing from a broken pipeline is worse than no timing, so this runs first and the denominator is
only quoted if it passes. Per design it checks:

  * the structure parses and holds exactly two polypeptide chains;
  * the binder chain is the one that is not the target, and it is the requested length;
  * no degenerate stretch -- a run of one residue type longer than `--max-run` (8 by default),
    which is what a collapsed hallucination looks like when the metrics still read well.

mmCIF and PDB are both read here with the standard library, so this runs on a machine that has no
structure package installed, which is the machine the designs get copied back to.
"""
import argparse
import json
import os
import sys

BACKBONE = 'CA'
DEFAULT_MAX_RUN = 8


def parse_pdb(text):
    chains = {}
    for line in text.splitlines():
        if line.startswith(('ATOM  ', 'HETATM')) and line[12:16].strip() == BACKBONE:
            chains.setdefault(line[21], []).append((line[22:27].strip(), line[17:20].strip()))
    return chains


def parse_cif(text):
    """Read `_atom_site` out of an mmCIF loop by column name, not by position."""
    lines = text.splitlines()
    chains = {}
    index = 0
    while index < len(lines):
        if lines[index].strip() != 'loop_':
            index += 1
            continue
        index += 1
        columns = []
        while index < len(lines) and lines[index].lstrip().startswith('_'):
            columns.append(lines[index].strip())
            index += 1
        if not any(column.startswith('_atom_site.') for column in columns):
            continue
        position = {name.split('.', 1)[1]: number for number, name in enumerate(columns)}
        needed = ('label_atom_id', 'label_asym_id', 'label_seq_id', 'label_comp_id')
        if not all(name in position for name in needed):
            continue
        chain_key = 'auth_asym_id' if 'auth_asym_id' in position else 'label_asym_id'
        sequence_key = 'auth_seq_id' if 'auth_seq_id' in position else 'label_seq_id'
        while index < len(lines):
            row = lines[index]
            if not row.strip() or row.strip().startswith(('#', 'loop_', '_')):
                break
            fields = row.split()
            index += 1
            if len(fields) < len(columns) or fields[position['label_atom_id']].strip('"\'') != BACKBONE:
                continue
            chains.setdefault(fields[position[chain_key]], []).append(
                (fields[position[sequence_key]], fields[position['label_comp_id']]))
    return chains


def read_chains(path):
    with open(path) as handle:
        text = handle.read()
    return parse_cif(text) if path.lower().endswith(('.cif', '.mmcif')) else parse_pdb(text)


def longest_run(residues):
    longest, running, previous = 0, 0, None
    for _, name in residues:
        running = running + 1 if name == previous else 1
        previous = name
        longest = max(longest, running)
    return longest


def check(path, binder_length, target_length=None, max_run=DEFAULT_MAX_RUN):
    chains = read_chains(path)
    result = {'path': path, 'chains': {name: len(residues) for name, residues in sorted(chains.items())}}
    problems = []
    if len(chains) != 2:
        problems.append(f'{len(chains)} chains, expected 2')
    lengths = {name: len(residues) for name, residues in chains.items()}
    binder = [name for name, length in lengths.items() if length == binder_length]
    if not binder:
        problems.append(f'no chain of the requested binder length {binder_length}: {lengths}')
    else:
        name = binder[0]
        result['binder_chain'] = name
        result['binder_longest_residue_run'] = longest_run(chains[name])
        if result['binder_longest_residue_run'] > max_run:
            problems.append(f'degenerate binder: {result["binder_longest_residue_run"]} identical residues in a row')
        if target_length is not None:
            others = [length for other, length in lengths.items() if other != name]
            if others and others[0] != target_length:
                problems.append(f'target chain is {others[0]} residues, expected {target_length}')
    result['problems'] = problems
    result['ok'] = not problems
    return result


# Files under an accepted-design folder that are deliberately NOT two-chain complexes, and so are
# not what the two-chain check is about: `_monomer` is the binder folded alone, written when
# save_binder_monomers is on (campaign.py), and BindCraft 2's trajectory folders hold intermediate
# and failed structures that were never accepted. Walking them would report failures that are not
# failures, which is worse than no check at all -- it trains the reader to ignore the gate.
NOT_A_COMPLEX = ('_monomer',)


def find_structures(root, skip=NOT_A_COMPLEX):
    found = []
    for directory, _subdirectories, names in os.walk(root):
        for name in sorted(names):
            if not name.lower().endswith(('.cif', '.pdb')):
                continue
            if any(marker in name for marker in skip):
                continue
            found.append(os.path.join(directory, name))
    return sorted(found)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', help='a campaign folder, or one structure file')
    parser.add_argument('--binder-length', type=int, required=True)
    parser.add_argument('--target-length', type=int)
    parser.add_argument('--max-run', type=int, default=DEFAULT_MAX_RUN)
    parser.add_argument('--report')
    arguments = parser.parse_args(argv)
    paths = [arguments.root] if os.path.isfile(arguments.root) else find_structures(arguments.root)
    results = [check(path, arguments.binder_length, arguments.target_length, arguments.max_run) for path in paths]
    summary = {'checked': len(results),
               'passing': sum(1 for result in results if result['ok']),
               'failing': [result for result in results if not result['ok']],
               'designs': results}
    text = json.dumps(summary, indent=2, sort_keys=True)
    if arguments.report:
        with open(arguments.report, 'w') as handle:
            handle.write(text + '\n')
    print(f'{summary["passing"]}/{summary["checked"]} structures pass')
    for result in summary['failing']:
        print(f'  {result["path"]}: {"; ".join(result["problems"])}')
    return 0 if results and not summary['failing'] else 1


if __name__ == '__main__':
    sys.exit(main())

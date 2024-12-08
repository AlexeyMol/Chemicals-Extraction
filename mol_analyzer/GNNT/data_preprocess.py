import pickle as pkl
import concurrent.futures
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, List

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.rdchem import Atom, Bond
from torch_geometric.data import Data
from tqdm import tqdm


@dataclass
class Sample:
    input: Data = field()
    output: Data = field()
    idx: int = field()


def load_pair(in_file, out_file, idx, node_feats_idx=None, edge_feats_idx=None):
    if any(not Path(x).is_file() for x in (in_file, out_file)):
        return None, []

    in_mol = Chem.MolFromMolFile(in_file, True, True, True)
    out_mol = Chem.MolFromMolFile(out_file, True, True, True)

    failed = []

    if in_mol is None:
        in_mol = Chem.MolFromMolFile(in_file, True, True, False)
        if in_mol is None:
            # print(f'Failed pair {idx + 1} (input, lost)', file=sys.stderr)
            failed.append((idx + 1, False, True))
        else:
            # print(f'Failed pair {idx + 1} (input, malformed)', file=sys.stderr)
            failed.append((idx + 1, False, False))

    if out_mol is None:
        out_mol = Chem.MolFromMolFile(out_file, True, True, False)
        if out_mol is None:
            # print(f'Failed pair {idx + 1} (output, lost)', file=sys.stderr)
            failed.append((idx + 1, True, True))
        else:
            # print(f'Failed pair {idx + 1} (output, malformed)', file=sys.stderr)
            failed.append((idx + 1, True, False))

    if in_mol is None or out_mol is None:
        return None, failed

    input_data = mol_to_graph(in_mol, node_feats_idx=node_feats_idx, edge_feats_idx=edge_feats_idx)
    output_data = mol_to_graph(out_mol, node_feats_idx=node_feats_idx, edge_feats_idx=edge_feats_idx)
    # print(f'Pair {idx + 1} succeeded')
    return Sample(input=input_data, output=output_data, idx=idx + 1), failed


def _init_worker(disable_rdkit_log=True):
    if disable_rdkit_log:
        for x in ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error'):
            RDLogger.DisableLog(x)


def load_dataset(data_root, use_cache=True, num_samples=None,
                 disable_rdkit_log=True, max_workers=15,
                 cache_suffix=None,
                 node_feats_idx=None, edge_feats_idx=None) -> Tuple[Dict[int, Sample], List[Tuple[int, bool, bool]]]:
    if disable_rdkit_log:
        for x in ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error'):
            RDLogger.DisableLog(x)

    print(f'Listing MOL files...')
    data_root = Path(data_root)

    std_samples = {int(x[:-4]) for x in os.listdir(data_root / 'std_zip') if x.endswith('.MOL')}
    unstd_samples = {int(x[:-4]) for x in os.listdir(data_root / 'unstd_zip') if x.endswith('.MOL')}
    all_samples = std_samples.intersection(unstd_samples)
    if num_samples is not None:
        all_samples = list(all_samples)[:num_samples]
    else:
        all_samples = list(all_samples)

    num_samples = len(all_samples)

    input_mol_files = [str(data_root / 'unstd_zip' / f'{idx}.MOL') for idx in all_samples]
    output_mol_files = [str(data_root / 'std_zip' / f'{idx}.MOL') for idx in all_samples]

    print(f'Dataset {num_samples=}')

    preprocessed_file = data_root / 'cache' / (
        'preprocessed.pkl' if cache_suffix is None else f'preprocessed_{cache_suffix}.pkl'
    )
    preprocessed_file.parent.mkdir(exist_ok=True, parents=True)
    if use_cache:
        if preprocessed_file.is_file():
            with open(preprocessed_file, 'rb') as f:
                dataset, failed = pkl.load(f)
                dataset = {idx: Sample(input=inpt, output=output, idx=idx) for idx, (inpt, output) in dataset.items()}
                return dataset, failed

    dataset = []
    failed = []

    with tqdm(total=num_samples,
              desc='Loading mol files') as progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, initializer=_init_worker,
                                                   initargs=(disable_rdkit_log,)) as executor:
            results = [
                executor.submit(load_pair, i, o, idx, node_feats_idx, edge_feats_idx)
                for i, o, idx in zip(input_mol_files, output_mol_files, range(num_samples))
            ]
            for future in concurrent.futures.as_completed(results):
                sample, fail = future.result()
                if sample is not None:
                    dataset.append(sample)
                if fail:
                    failed.extend(fail)
                progress.update(1)

    dataset = {s.idx: s for s in dataset}

    if len(failed) > 0:
        failed_completely = [idx for idx, out, completely in failed if completely]
        print(f'Having malformed MOLs: {len(failed_completely)} failed to read, {len(failed) - len(failed_completely)} '
              f'saved with malformed data', file=sys.stderr)

    with open(preprocessed_file, 'wb') as f:
        pkl.dump(({sample.idx: (sample.input, sample.output) for sample in dataset.values()}, failed), f)

    if disable_rdkit_log:
        for x in ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error'):
            RDLogger.EnableLog(x)

    return dataset, failed


def mol_to_graph(mol, double_directions=True, node_feats_idx=None, edge_feats_idx=None):
    """
    Из MOL в векторы для нейросети
    """
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()

    if node_feats_idx is not None and 0 not in node_feats_idx:
        raise ValueError('0 (atomic num) is required to be in node feats idx list')
    if edge_feats_idx is not None and 0 not in edge_feats_idx:
        raise ValueError('0 (bond type) is required to be in edge feats idx list')

    atom: Atom
    node_feats = [
        [
            x for idx, x in enumerate((
            # 0
            atom.GetAtomicNum(),

            # 1
            atom.GetFormalCharge(),

            # 2
            atom.GetDegree(),

            # 3
            atom.GetExplicitValence(),

            # 4
            atom.GetImplicitValence(),

            # 5
            atom.GetHybridization().real,

            # 6
            atom.GetChiralTag().real,

            # 7
            atom.GetIsAromatic() and 1. or 0.,

            # 8
            atom.GetNoImplicit() and 1. or 0.,

            # 9
            atom.IsInRing() and 1. or 0.,

            # 10
            atom.GetIsotope(),

            # 11
            atom.GetNumRadicalElectrons(),
        ))
            if node_feats_idx is None or idx in node_feats_idx
        ] for atom in atoms
    ]
    node_feats = torch.tensor(node_feats, dtype=torch.float)

    edge_feats = []
    edge_index = []
    for bond in bonds:
        bond: Bond
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        edge_index.append([start, end])
        if double_directions:
            edge_index.append([end, start])

        edge_feats += (
                [[
                    x for idx, x in enumerate((
                        bond.GetBondTypeAsDouble(),
                        bond.GetIsConjugated(),
                        bond.GetIsAromatic(),
                        bond.GetStereo().real,
                        bond.IsInRing()
                    ))
                    if edge_feats_idx is None or idx in edge_feats_idx
                ]] * (double_directions and 2 or 1)
        )

    edge_feats = torch.tensor(edge_feats, dtype=torch.float)
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)

    return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)


def graph_to_mol(node_feats, edge_index, edge_feats, node_feats_idx=None, edge_feats_idx=None):
    mol = Chem.RWMol()

    def _is_specified(_idx, _idx_list):
        if _idx_list is None:
            return True
        return _idx in _idx_list

    def _get_node_cast_map(_atom):
        return {
            1: (_atom.SetFormalCharge, int),
            6: (_atom.SetChiralTag, lambda x: Chem.rdchem.ChiralType(int(x))),
            7: (_atom.SetIsAromatic, bool),
            8: (_atom.SetNoImplicit, bool),
            10: (_atom.SetIsotope, int),
            11: (_atom.SetNumRadicalElectrons, int),
        }

    def _get_edge_cast_map(_bond):
        return {
            1: (_bond.SetIsConjugated, bool),
            2: (_bond.SetIsAromatic, bool),
            3: (_bond.SetStereo, lambda x: Chem.rdchem.BondStereo(int(x))),
        }

    def _normalize_idx(_idx, _idx_list):
        if _idx_list is None:
            return _idx
        return {
            int(_m): _i
            for _i, _m in enumerate(sorted(_idx_list))
        }[_idx]

    for node in node_feats:
        atomic_num = int(node[0].item())
        atom = Chem.Atom(atomic_num)

        cast_map = _get_node_cast_map(atom)
        for idx in range(1, (max(node_feats_idx) + 1) if node_feats_idx is not None else 12):
            if _is_specified(idx, node_feats_idx) and idx < len(node):
                if idx in cast_map:
                    fnc, cast = cast_map[idx]
                    fnc(cast(node[_normalize_idx(idx, node_feats_idx)]))

        mol.AddAtom(atom)

    num_edges = edge_index.shape[1] // 2
    for i in range(num_edges):
        start, end = edge_index[:, i * 2]
        edge = edge_feats[i * 2]
        bond_type = Chem.BondType(int(edge[0].item()))
        mol.AddBond(int(start.item()), int(end.item()), bond_type)

        bond = mol.GetBondBetweenAtoms(int(start.item()), int(end.item()))
        cast_map = _get_edge_cast_map(bond)
        for idx in range(1, (max(edge_feats_idx) + 1) if edge_feats_idx is not None else 4):
            if _is_specified(idx, edge_feats_idx) and idx < len(edge):
                if idx in cast_map:
                    fnc, cast = cast_map[idx]
                    fnc(cast(edge[_normalize_idx(idx, edge_feats_idx)].item()))

    # Sanitize the molecule to adjust implicit valences, aromaticity, etc.
    # rdmolops.SanitizeMol(mol)

    return mol.GetMol()


if __name__ == '__main__':
    load_dataset(sys.argv[1], cache_suffix=sys.argv[2] if len(sys.argv) > 2 else None, use_cache=False)

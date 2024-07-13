import concurrent.futures
import math
import os
import sys
from pathlib import Path
from typing import Union, List, Tuple

import torch
from rdkit import RDLogger
from torch_geometric.data import Dataset
from tqdm import tqdm

from .data_preprocess import Sample, load_pair
from .data_analysis import convert_to_one_hot


class ChemDataset(Dataset):
    def __init__(self, root, force_reload=False, limit_samples=None, **kwargs):
        self.data_root = Path(root)

        print('Listing MOLs...', file=sys.stderr)
        std_samples = {int(x[:-4]) for x in os.listdir(self.data_root / 'std_zip') if x.endswith('.MOL')}
        unstd_samples = {int(x[:-4]) for x in os.listdir(self.data_root / 'unstd_zip') if x.endswith('.MOL')}
        all_samples = std_samples.intersection(unstd_samples)

        print('Verifying...', file=sys.stderr)
        if limit_samples is None:
            self.valid_ids = list(all_samples)
        else:
            self.valid_ids = list(all_samples)[:limit_samples]

        self.failed_ids = []
        self.succeeded_ids = []
        self.stats = None

        super().__init__(root, force_reload=force_reload, **kwargs)

        if os.path.isfile(self.dataset_stats_path):
            self._update_stats()

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return [f'pair_{idx}.pt' for idx in self.valid_ids] + ['statistics.pt']

    @property
    def dataset_stats_path(self):
        return os.path.join(self.processed_dir, 'statistics.pt')

    def process(self) -> None:
        self.failed_ids = []
        self.succeeded_ids = []
        self.stats = None

        intermediate_stats_path = Path(self.dataset_stats_path + '.temp')

        if self.force_reload or not intermediate_stats_path.is_file():
            self._load_mols(100000)
            torch.save((self.succeeded_ids, self.stats), intermediate_stats_path)
        else:
            print('Continuing aborted processing', file=sys.stderr)
            self.succeeded_ids, self.stats = torch.load(intermediate_stats_path)
        self._encode_data(100)

        torch.save((self.succeeded_ids, self.stats), self.dataset_stats_path)
        intermediate_stats_path.unlink()

    def _load_mols(self, batch_size):
        start = 0
        valid_len = len(self.valid_ids)

        def _init_rdkit_log():
            for x in ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error'):
                RDLogger.DisableLog(x)

        _init_rdkit_log()

        with tqdm(desc='Loading MOLs', total=valid_len) as pb:
            while start < valid_len:
                end = start + batch_size
                batch = self.valid_ids[start:min(end, valid_len)]
                batch_len = len(batch)

                in_mol = [str(self.data_root / 'unstd_zip' / f'{idx}.MOL') for idx in batch]
                out_mol = [str(self.data_root / 'std_zip' / f'{idx}.MOL') for idx in batch]
                with concurrent.futures.ThreadPoolExecutor(max_workers=64, initializer=_init_rdkit_log) as ex:
                    batch_loaded = list(ex.map(load_pair, in_mol, out_mol, batch))

                for idx, (sample, fail) in zip(batch, batch_loaded):
                    if sample is not None:
                        self.succeeded_ids.append(idx)
                        self._calculate_stats(sample)

                    else:
                        self.failed_ids.append(idx)

                    torch.save(
                        (sample.input, sample.output) if sample is not None else None,
                        self.get_processed_data_path(idx)
                    )

                pb.update(batch_len)
                start += batch_size

        for x in ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error'):
            RDLogger.EnableLog(x)

    def _calculate_stats(self, sample):
        node_features = torch.vstack([sample.input.x, sample.output.x])
        if (all(
                x.dim() > 1 and x.size(1) > 0
                for x in (sample.output.edge_attr, sample.input.edge_attr)
        )):
            edge_features = torch.vstack([sample.input.edge_attr, sample.output.edge_attr])
        else:
            edge_features = None
        if self.stats is None:
            self.stats = dict()
        if 'node_features_min' in self.stats:
            node_features = torch.vstack([
                node_features, self.stats['node_features_min'], self.stats['node_features_max']
            ])
        if ('edge_features_min' in self.stats and
                edge_features is not None and
                edge_features.size(1) > 0):
            edge_features = torch.vstack([
                edge_features, self.stats['edge_features_min'], self.stats['edge_features_max']
            ])
        self.stats['node_features_min'] = torch.min(node_features, dim=0)[0]
        self.stats['node_features_max'] = torch.max(node_features, dim=0)[0]
        if edge_features is not None and edge_features.size(1) > 0:
            self.stats['edge_features_min'] = torch.min(edge_features, dim=0)[0]
            self.stats['edge_features_max'] = torch.max(edge_features, dim=0)[0]

    def _encode_data(self, batch_size):
        print('Encoding data...', file=sys.stderr)
        valid_len = len(self.succeeded_ids)

        def _handle_batch(batch: List[int]):
            batch = {
                idx: Sample(idx=idx, input=i, output=o) for idx, (i, o)
                in ((idx, torch.load(self.get_processed_data_path(idx))) for idx in batch)
            }
            convert_to_one_hot(batch, inplace=True, stats=self.stats)
            for idx, sample in batch.items():
                torch.save((sample.input, sample.output), self.get_processed_data_path(idx))

        batches = [
            self.succeeded_ids[i * batch_size:min((i + 1) * batch_size, valid_len)]
            for i in range(math.ceil(valid_len / batch_size))
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
            list(ex.map(_handle_batch, batches))

    def _update_stats(self):
        if self.stats is None:
            self.succeeded_ids, self.stats = torch.load(self.dataset_stats_path)

    def get_processed_data_path(self, idx):
        return os.path.join(self.processed_dir, f'pair_{idx}.pt')

    def get_stats(self):
        if self.stats is None:
            self.process()
        return self.stats

    def len(self) -> int:
        return len(self.succeeded_ids)

    def get(self, idx: int):
        data = torch.load(self.get_processed_data_path(self.succeeded_ids[idx]))
        for d in data:
            d.x = d.x.to(torch.float)
            d.edge_attr = d.edge_attr.to(torch.float)
        return data

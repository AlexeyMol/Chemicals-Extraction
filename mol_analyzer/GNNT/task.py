import json
import sys
import os
from os import PathLike
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List
import pickle as pkl
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Mol
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm as default_tqdm


from .data_analysis import normalize_sample, denorm_result, convert_to_one_hot, from_one_hot, \
    level_dataset, normalize_dataset, feats_identifier
    
from .data_ondisk import ChemDataset
from .data_preprocess import Sample, mol_to_graph, graph_to_mol, load_dataset
from .modeling import GNNTransformModel


class StandardizerTask:
    @classmethod
    def for_train(cls,
                  name='GNNT', dataset_path=None, node_feats_idx=None, edge_feats_idx=None,
                  **kwargs):
        """
        Инициализация для обучения модели
        """
        if dataset_path is None or not Path(dataset_path).is_dir():
            raise ValueError('Invalid dataset path')
        identifier = feats_identifier(node_feats_idx, edge_feats_idx)

        if 'num_samples' in kwargs:
            num_samples = kwargs['num_samples']
            if num_samples is not None:
                if num_samples >= 1000000:
                    ns_str = f'{num_samples // 1000000}kk'
                elif num_samples >= 1000:
                    ns_str = f'{num_samples // 1000}k'
                elif num_samples >= 100:
                    ns_str = f'{num_samples / 1000 :.1f}k'
                else:
                    ns_str = 'tiny'
                identifier = f'{identifier}-{ns_str}'


        return cls(name=name, dataset_path=dataset_path,
                   node_feats_idx=node_feats_idx, edge_feats_idx=edge_feats_idx,
                   cache_suffix=identifier, load_ckp=identifier,
                   **kwargs)

    def __init__(self, name='GNNT', load_ckp="0dc3-03-1kk", model_home='./GNNT/ckp',
                 model_params=None, model_kwargs=None, one_hot=True, ondisk_dataset=False,
                 filter_output_only=True,
                 tqdm=None,
                 dataset_path=None, node_feats_idx=None, edge_feats_idx=None,
                 cache_suffix=None, num_samples=None):
        """
        Инициализация класса StandardizerTask.

        :param name: имя модели
        :param load_ckp: имя загружаемого чекпойнта
        :param model_home: путь к директории с чекпойнтами
        :param model_params: параметры модели
        :param model_kwargs: дополнительные параметры модели
        :param one_hot: использовать one-hot кодирование
        :param ondisk_dataset: использовать датасет на диске
        :param tqdm: функция прогресс-бара
        """
        this_dir, this_filename = os.path.split(__file__)

        DATA_PATH = os.path.join(this_dir, "ckp")
        self.model_home = Path(DATA_PATH)
        self.model_home.mkdir(parents=True, exist_ok=True)

        self.name = name
        self.one_hot = one_hot
        self.ckp_name = load_ckp
        self.ondisk_dataset = ondisk_dataset


        self.tqdm = tqdm or default_tqdm

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        stats_file = self.model_home / f'{self.name}-stats-{load_ckp}.json' if load_ckp else None
        ckp_weights_file = self.model_home / f'{self.name}-ckp-{load_ckp}-weights.pt' if load_ckp else None
        ckp_file = self.model_home / f'{self.name}-ckp-{load_ckp}.pt' if load_ckp else None
        self.dataset_original: Optional[Dict[int, Sample]] = None
        self.dataset: Optional[Union[Dict[int, Sample], ChemDataset]] = None
        self.stats: Optional[dict] = None
        if dataset_path is not None:
            self.load_dataset(dataset_path, True, cache_suffix,
                              other_only=False, num_samples=num_samples,
                              node_feats_idx=node_feats_idx, edge_feats_idx=edge_feats_idx,
                              filter_output_only=filter_output_only)

        def _init_model():
            if model_params is None:
                if self.stats is not None and all(
                        x in self.stats for x in ('node_embedding_size', 'edge_embedding_size')
                ):
                    if all(x in self.stats for x in ('node_embedding_output_size', 'edge_embedding_output_size')):
                        params = (
                            self.stats['node_embedding_size'], self.stats['edge_embedding_size'],
                            self.stats['node_embedding_output_size'], self.stats['edge_embedding_output_size'],
                        )
                    else:
                        params = (self.stats['node_embedding_size'], self.stats['edge_embedding_size'])
                else:
                    print('WARN: Using default hardcoded embedding sizes', file=sys.stderr)
                    params = (446, 14)  # (214, 13)
            else:
                params = model_params
            kwargs = {} if model_kwargs is None else model_kwargs
            self.model = GNNTransformModel(*params, **kwargs).to(self.device)

        if load_ckp and ckp_weights_file.is_file() and stats_file.is_file():
            with open(stats_file, 'r') as f:
                self.stats = self.to_tensors(json.load(f))
            _init_model()
            self.model.load_state_dict(torch.load(str(ckp_weights_file.resolve())))
        elif load_ckp and stats_file.is_file() and ckp_file.is_file():
            self.model = torch.load(str(ckp_file.resolve())).to(self.device)
            with open(stats_file, 'r') as f:
                self.stats = self.to_tensors(json.load(f))
        else:
            _init_model()

    def predict_mol(self, mol: Union[Mol, PathLike], disable_rdkit_log=False, save: Optional[PathLike] = None):
        """
        Предсказание структуры молекулы на основе one-hot кодирования.

        :param mol: молекула или путь к файлу с молекулой
        :param disable_rdkit_log: отключение логов RDKit
        :param save: путь для сохранения предсказанной молекулы
        :return: предсказанная молекула
        """
        if not self.one_hot:
            raise NotImplementedError("MOL prediction implemented only for one-hot trained model. "
                                      "Use StandardizerTask(one_hot=True).")
        if disable_rdkit_log:
            for x in ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error'):
                RDLogger.DisableLog(x)
        if isinstance(mol, Mol):
            in_mol = mol
        else:
            in_mol = Chem.MolFromMolFile(mol, True, True, False)

        node_feats_idx = self.stats.get('node_feats_idx', None)
        edge_feats_idx = self.stats.get('edge_feats_idx', None)
        filter_output_only = self.stats.get('filter_output_only', False)

        if node_feats_idx is None or edge_feats_idx is None:
            print('WARN: using full node and/or edge features')

        data = mol_to_graph(in_mol,
                            node_feats_idx=node_feats_idx if not filter_output_only else None,
                            edge_feats_idx=edge_feats_idx if not filter_output_only else None)
        pred_nodes, pred_edges = self.predict(data,
                                              node_feats_idx=node_feats_idx if filter_output_only else None,
                                              edge_feats_idx=edge_feats_idx if filter_output_only else None)

        out_mol = graph_to_mol(pred_nodes, data.edge_index, pred_edges,
                               node_feats_idx if filter_output_only else None,
                               edge_feats_idx if filter_output_only else None)

        if save:
            if not isinstance(save, (PathLike, str)) and isinstance(mol, (PathLike, str)):
                save = Path(mol).with_suffix('.STD.MOL')
            elif not isinstance(save, (PathLike, str)):
                raise ValueError('\"Save\" must be a path when \"mol\" is an object')
            Chem.MolToMolFile(out_mol, save, True, -1, True)

        return out_mol

    def predict(self, sample: Union[Sample, Data, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                norm_input=True, round_output=False, node_feats_idx=None, edge_feats_idx=None):
        """
        Предсказание структуры молекулы на основе one-hot кодирования.

        :param sample: выборка данных
        :param norm_input: нормализация входных данных
        :param round_output: округление выходных данных
        :return: предсказанная структура молекулы
        """
        if self.stats is None:
            raise RuntimeError('The model seems not trained, nor loaded from checkpoint, nor dataset loaded')

        if isinstance(sample, Sample):
            sample = sample.input
        elif isinstance(sample, tuple):
            node_feats, edge_index, edge_feats = sample
            sample = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)
        elif not isinstance(sample, Data):
            raise ValueError('Input sample must be either of types '
                             '[data.Sample, torch_geometric.data.Data, Tuple[torch.Tensor, torch.Tensor]]')

        if norm_input:
            if self.one_hot:
                sample, _ = convert_to_one_hot(sample, self.stats)
            else:
                sample = normalize_sample(sample, self.stats)

        if self.one_hot:
            sample.x = sample.x.to(torch.float)
            sample.edge_attr = sample.edge_attr.to(torch.float)

        self.model.eval()
        sample = sample.to(self.device)
        result = self.model(sample.x, sample.edge_index, sample.edge_attr)
        result = tuple(x.cpu().detach() for x in result)

        if self.one_hot:
            result = tuple(torch.sigmoid(x) for x in result)
            result = from_one_hot(*result, stats=self.stats,
                                  node_feats_idx=node_feats_idx,
                                  edge_feats_idx=edge_feats_idx)
        else:
            result = denorm_result(*result, statistics=self.stats)

        if round_output and not self.one_hot:
            return tuple(torch.round(x) if len(x) > 0 else x for x in result)

        return result

    def compare(self, true_sample: Union[Data, Sample], predict: Tuple[torch.Tensor, torch.Tensor],
                booleans=True, denorm=False):
        if denorm and self.stats is None:
            raise RuntimeError('The model seems not trained, nor loaded from checkpoint, nor dataset loaded, '
                               'cannot run compare with denorm=True')
        if self.one_hot:
            denorm = False

        if isinstance(true_sample, Sample):
            true_sample = true_sample.output

        predict = tuple(x.cpu() for x in predict)
        true_sample = true_sample.cpu()

        if self.one_hot:
            true_sample.x, true_sample.edge_attr = from_one_hot(true_sample.x, true_sample.edge_attr, stats=self.stats)

        node_feats, edge_feats = predict if not denorm else denorm_result(*predict, statistics=self.stats)
        node_diff = true_sample.x - node_feats
        edge_diff = true_sample.edge_attr - edge_feats

        if not booleans:
            return node_diff, edge_diff

        return torch.abs(node_diff) < .5, torch.abs(edge_diff) < .5

    def load_dataset(self, dataset_path=None, force=False, cache_suffix=None, rebuild_cache=False, other_only=False,
                     original=False, num_samples=None, node_feats_idx=None, edge_feats_idx=None,
                     filter_output_only=True):
        if self.ondisk_dataset and self.dataset is None:
            self.dataset = ChemDataset(dataset_path)
            self.stats = self.dataset.get_stats()
            return
        elif self.ondisk_dataset:
            return

        if self.dataset is not None and not force and (not original or self.dataset_original is not None):
            return
        elif not force and dataset_path is None:
            raise ValueError('Dataset not loaded, nor dataset_path provided')
        if dataset_path is None:
            raise ValueError('dataset_path not provided')

        cache_name = f'leveled_{cache_suffix}.pkl' if cache_suffix else 'leveled.pkl'
        if other_only:
            cache_name = 'oo_' + cache_name
        cache_file = Path(dataset_path) / 'cache' / cache_name

        oh_cache_file = Path(dataset_path) / 'cache' / f'oh_{cache_name}'

        if rebuild_cache or not cache_file.is_file():
            print('Loading dataset...', file=sys.stderr)
            dataset, _ = load_dataset(dataset_path,
                                      cache_suffix=cache_suffix, num_samples=num_samples,
                                      node_feats_idx=node_feats_idx if not filter_output_only else None,
                                      edge_feats_idx=edge_feats_idx if not filter_output_only else None)
            print('Leveling dataset...', file=sys.stderr)
            self.dataset_original = level_dataset(dataset, 0.6, other_only=other_only)
            print('Writing to cache file...', file=sys.stderr)
            with open(cache_file, 'wb') as f:
                pkl.dump({idx: (d.input, d.output) for idx, d in self.dataset_original.items()}, f)
        elif original or not self.one_hot or self.one_hot and not oh_cache_file.is_file():
            print('Loading cached dataset...', file=sys.stderr)
            with open(cache_file, 'rb') as f:
                data = pkl.load(f)
            self.dataset_original = {idx: Sample(idx=idx, input=i, output=o) for idx, (i, o) in data.items()}

        if self.one_hot:
            if rebuild_cache or not oh_cache_file.is_file():
                print('Calculating one-hot...', file=sys.stderr)
                self.dataset, self.stats = convert_to_one_hot(
                    self.dataset_original,
                    node_feats_idx=node_feats_idx if filter_output_only else None,
                    edge_feats_idx=edge_feats_idx if filter_output_only else None
                )
                data = {idx: (d.input, d.output) for idx, d in self.dataset.items()}
                self.stats['node_feats_idx'] = node_feats_idx
                self.stats['edge_feats_idx'] = edge_feats_idx
                self.stats['filter_output_only'] = filter_output_only
                with open(oh_cache_file, 'wb') as f:
                    pkl.dump((data, self.stats), f)
                if not original:
                    del self.dataset_original
                    self.dataset_original = None
            else:
                print('Loading cached one-hot...', file=sys.stderr)
                with open(oh_cache_file, 'rb') as f:
                    data, self.stats = pkl.load(f)
                self.dataset = {idx: Sample(idx=idx, input=i, output=o) for idx, (i, o) in data.items()}
        else:
            print('Normalizing dataset...', file=sys.stderr)
            self.dataset, self.stats = normalize_dataset(self.dataset_original)
        print('Dataset loaded.', file=sys.stderr)

    @staticmethod
    def to_json_obj(d):
        return {
            k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
            for k, v in d.items()
        }

    @staticmethod
    def to_tensors(d):
        return {
            k: torch.tensor(v) if isinstance(v, list) else v
            for k, v in d.items()
        }

import sys
from pathlib import Path
from typing import Iterable, Callable, Dict, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data

from .data_preprocess import load_dataset, Sample

OH_EXCLUDE_NODE_FEATS = {1, }
OH_EXCLUDE_EDGE_FEATS = set()


def _filter_generator(dataset: Iterable[Sample], predicate: Callable[[Data, Data], bool]):
    return (
        s for s in dataset
        if predicate(s.input, s.output)
    )


def _l(gen):
    return len(list(gen))


def filter_dataset(dataset, keep_edgeless=True, keep_differ_edges=False):
    no_edges_idx = {s.idx for s in no_edges(dataset.values())} if not keep_edgeless else set()
    node_count_diffs_idx = {s.idx for s in node_count_diffs(dataset.values())}
    edge_count_diffs_idx = {s.idx for s in edge_count_diffs(dataset.values())}
    edge_set_diffs_idx = {s.idx for s in edge_set_diffs(dataset.values())} if not keep_differ_edges else set()

    to_remove = no_edges_idx.union(node_count_diffs_idx).union(edge_count_diffs_idx).union(edge_set_diffs_idx)
    return {
        s.idx: s for s in dataset.values()
        if s.idx not in to_remove
    }


def level_dataset(dataset, other_samples_rate=.6, other_only=False, changed_only=False):
    samples_filtered = filter_dataset(dataset).values()
    samples_filtered_edges_required = filter_dataset(dataset, keep_edgeless=False).values()
    node_feats_changed_set = set(x.idx for x in node_feats_changed(samples_filtered))
    edge_feats_changed_set = set(x.idx for x in edge_feats_changed(samples_filtered_edges_required))

    valuable_samples_idx = node_feats_changed_set.union(edge_feats_changed_set)
    other_samples_idx = sorted(x.idx for x in samples_filtered if x.idx not in valuable_samples_idx)

    if changed_only:
        return {idx: dataset[idx] for idx in valuable_samples_idx}
    if other_only:
        return {idx: dataset[idx] for idx in other_samples_idx}

    other_len = len(valuable_samples_idx) * other_samples_rate / (1. - other_samples_rate)
    other_samples_idx = other_samples_idx[:min(len(other_samples_idx), int(other_len))]

    valuable_samples = {idx: dataset[idx] for idx in valuable_samples_idx}
    other_samples = {idx: dataset[idx] for idx in other_samples_idx}

    return {**valuable_samples, **other_samples}


def analyze_dataset(dataset: Dict[int, Sample], use_input=True, use_output=True):
    if not any((use_output, use_input)):
        raise ValueError('Neither using input nor output')

    num_graphs = len(dataset) * 2
    num_nodes_list = []
    num_edges_list = []
    node_features_list = []
    edge_features_list = []

    def add(data):
        num_nodes_list.append(data.x.size(0))
        num_edges_list.append(data.edge_attr.size(0) if data.edge_attr.dim() > 1 else 0)
        node_features_list.append(data.x)
        if data.edge_attr.dim() > 1:
            edge_features_list.append(data.edge_attr)

    for data in dataset.values():
        if use_input:
            add(data.input)
        if use_output:
            add(data.output)

    avg_num_nodes = np.mean(num_nodes_list)
    avg_num_edges = np.mean(num_edges_list)
    avg_degree = np.mean([2 * e / n for e, n in zip(num_edges_list, num_nodes_list)])
    node_feature_dim = node_features_list[0].size(1)
    edge_feature_dim = edge_features_list[0].size(1)

    all_node_features = torch.cat(node_features_list, dim=0)
    all_edge_features = torch.cat(edge_features_list, dim=0)

    node_features_mean = torch.mean(all_node_features, dim=0)
    node_features_std = torch.std(all_node_features, dim=0)
    node_features_min = torch.min(all_node_features, dim=0)[0]
    node_features_max = torch.max(all_node_features, dim=0)[0]

    edge_features_mean = torch.mean(all_edge_features, dim=0)
    edge_features_std = torch.std(all_edge_features, dim=0)
    edge_features_min = torch.min(all_edge_features, dim=0)[0]
    edge_features_max = torch.max(all_edge_features, dim=0)[0]

    return {
        'num_graphs': num_graphs,
        'avg_num_nodes': avg_num_nodes,
        'avg_num_edges': avg_num_edges,
        'avg_degree': avg_degree,
        'node_feature_dim': node_feature_dim,
        'edge_feature_dim': edge_feature_dim,
        'node_features_mean': node_features_mean,
        'node_features_std': node_features_std,
        'node_features_min': node_features_min,
        'node_features_max': node_features_max,
        'edge_features_mean': edge_features_mean,
        'edge_features_std': edge_features_std,
        'edge_features_min': edge_features_min,
        'edge_features_max': edge_features_max,
    }, all_node_features, all_edge_features


def convert_to_one_hot(data: Union[Dict[int, Sample], Data], stats: Optional[dict] = None, inplace=False):
    is_one_sample = isinstance(data, Data)
    if not inplace:
        if is_one_sample:
            data = data.clone()
        else:
            data = {idx: Sample(idx=idx, input=s.input.clone(), output=s.output.clone()) for idx, s in data.items()}
    if stats is None:
        if is_one_sample:
            raise ValueError('Stats should be provided when data is single sample')
        stats, _, _ = analyze_dataset(data)

    node_features_min = stats['node_features_min'].to(torch.int64)
    node_features_max = stats['node_features_max'].to(torch.int64)
    edge_features_min = stats['edge_features_min'].to(torch.int64)
    edge_features_max = stats['edge_features_max'].to(torch.int64)

    node_feature_length = int(sum(max_val - min_val + 1 for idx, (min_val, max_val)
                                  in enumerate(zip(node_features_min.tolist(), node_features_max.tolist()))
                                  if idx not in OH_EXCLUDE_NODE_FEATS))
    edge_feature_length = int(sum(max_val - min_val + 1 for idx, (min_val, max_val)
                                  in enumerate(zip(edge_features_min.tolist(), edge_features_max.tolist()))
                                  if idx not in OH_EXCLUDE_EDGE_FEATS))

    if is_one_sample:
        _one_hot_encode_data(data, node_features_min, node_features_max, edge_features_min, edge_features_max,
                             node_feature_length, edge_feature_length)

    else:
        for pair_sample in data.values():
            for data_sample in [pair_sample.input, pair_sample.output]:
                if data_sample is None:
                    continue

                _one_hot_encode_data(data_sample, node_features_min, node_features_max,
                                     edge_features_min, edge_features_max,
                                     node_feature_length, edge_feature_length)

        stats['node_embedding_size'] = node_feature_length
        stats['edge_embedding_size'] = edge_feature_length

    return data, stats


def from_one_hot(node_features, edge_features, stats: dict):
    node_feature_min = [x for idx, x in enumerate(stats['node_features_min'].tolist())
                        if idx not in OH_EXCLUDE_NODE_FEATS]
    node_feature_max = [x for idx, x in enumerate(stats['node_features_max'].tolist())
                        if idx not in OH_EXCLUDE_NODE_FEATS]
    edge_feature_min = [x for idx, x in enumerate(stats['edge_features_min'].tolist())
                        if idx not in OH_EXCLUDE_EDGE_FEATS]
    edge_feature_max = [x for idx, x in enumerate(stats['edge_features_max'].tolist())
                        if idx not in OH_EXCLUDE_EDGE_FEATS]

    def _get_feat_value(feats, idx, x_min, x_max, feats_min, feats_max):
        if feats.dim() != 2 or feats.size(0) == 0:
            return feats
        start = int(sum(feats_max[:idx]) - sum(feats_min[:idx]))
        count = int(x_max - x_min)
        feats_slice = feats[:, start:start + count]
        return torch.argmax(feats_slice, dim=1).to(torch.float) + x_min

    def _return_excluded(feats: list, exclude):
        for e in exclude:
            feats.insert(e, torch.tensor([-1.0] * feats[0].size(0), dtype=torch.float))
        return feats

    return tuple(
        torch.stack(_return_excluded([
            _get_feat_value(feats.to(torch.float), idx, x_min, x_max, feats_min, feats_max)
            for idx, (x_min, x_max) in enumerate(zip(feats_min, feats_max))
        ], exclude)).t().contiguous()
        for feats, feats_min, feats_max, exclude in (
            (node_features, node_feature_min, node_feature_max, OH_EXCLUDE_NODE_FEATS),
            (edge_features, edge_feature_min, edge_feature_max, OH_EXCLUDE_EDGE_FEATS)
        )
    )


def _one_hot_encode_data(data, node_features_min, node_features_max, edge_features_min, edge_features_max,
                         node_feature_length, edge_feature_length):
    def _encode(num_entities, feats, feature_length, num_features, features_min, features_max, exclude):
        one_hot = torch.zeros((num_entities, feature_length), dtype=torch.bool)

        current_idx = 0
        for i in range(num_features):
            if i in exclude:
                continue
            min_val = features_min[i]
            max_val = features_max[i]
            one_hot_length = max_val - min_val + 1
            feature_values = feats[:, i].to(torch.int64)
            one_hot_indices = feature_values.unsqueeze(1) - min_val
            one_hot_indices = one_hot_indices + current_idx
            one_hot.scatter_(1, one_hot_indices, 1)
            current_idx += one_hot_length
        return one_hot

    num_nodes, num_node_features = data.x.shape
    data.x = _encode(num_nodes, data.x,
                     node_feature_length, num_node_features,
                     node_features_min, node_features_max,
                     OH_EXCLUDE_NODE_FEATS)

    if data.edge_attr.dim() == 2:
        num_edges, num_edge_features = data.edge_attr.shape
        data.edge_attr = _encode(num_edges, data.edge_attr,
                                 edge_feature_length, num_edge_features,
                                 edge_features_min, edge_features_max,
                                 OH_EXCLUDE_EDGE_FEATS)


def normalize_dataset(dataset: Dict[int, Sample], inplace=False, output_features=False):
    if not inplace:
        dataset = {idx: Sample(idx=idx, input=s.input.clone(), output=s.output.clone()) for idx, s in dataset.items()}
    statistics, edges, nodes = analyze_dataset(dataset)

    for sample in dataset.values():
        normalize_sample(sample.input, statistics, True)
        normalize_sample(sample.output, statistics, True)

    if output_features:
        return dataset, statistics, (edges, nodes)
    return dataset, statistics


def normalize_sample(sample, statistics, inplace=False):
    if not inplace:
        sample = sample.clone()

    node_features_mean = statistics['node_features_mean']
    node_features_std = statistics['node_features_std']
    edge_features_mean = statistics['edge_features_mean']
    edge_features_std = statistics['edge_features_std']
    node_features_std[node_features_std == 0] = 1
    edge_features_std[edge_features_std == 0] = 1
    sample.x = (sample.x - node_features_mean) / node_features_std
    sample.x[torch.isinf(sample.x) | torch.isnan(sample.x)] = 0
    if len(sample.edge_attr) > 0:
        sample.edge_attr = (sample.edge_attr - edge_features_mean) / edge_features_std
        sample.edge_attr[torch.isinf(sample.edge_attr) | torch.isnan(sample.edge_attr)] = 0

    return sample


def denorm_result(node_features, edge_features, statistics):
    node_features_mean = statistics['node_features_mean']
    node_features_std = statistics['node_features_std']
    edge_features_mean = statistics['edge_features_mean']
    edge_features_std = statistics['edge_features_std']
    node_features_std[node_features_std == 0] = 1
    edge_features_std[edge_features_std == 0] = 1

    return (
        node_features * node_features_std + node_features_mean,
        edge_features * edge_features_std + edge_features_mean if len(edge_features) > 0 else edge_features,
    )


def node_count_diffs(dataset):
    return _filter_generator(dataset, lambda i, o: i.x.shape[0] != o.x.shape[0])


def edge_count_diffs(dataset):
    return _filter_generator(dataset, lambda i, o: i.edge_attr.shape[0] != o.edge_attr.shape[0])


def node_count_equals_edge_count_diffs(dataset):
    return _filter_generator(edge_count_diffs(dataset), lambda i, o: i.x.shape[0] == o.x.shape[0])


def no_edges(dataset):
    return _filter_generator(dataset, lambda i, o: i.edge_attr.shape[0] == 0)


def no_nodes(dataset):
    return _filter_generator(dataset, lambda i, o: i.x.shape[0] == 0)


def no_edges_multiple_nodes(dataset):
    return _filter_generator(no_edges(dataset), lambda i, o: i.x.shape[0] > 1)


def node_feats_changed(filtered_dataset):
    return _filter_generator(filtered_dataset, lambda i, o: torch.max(torch.abs(i.x - o.x)).item() > 0)


def edge_feats_changed(filtered_dataset):
    return _filter_generator(filtered_dataset, lambda i, o: torch.max(torch.abs(i.edge_attr - o.edge_attr)).item() > 0)


def edge_set(graph):
    edges = graph.edge_index.t().numpy()
    return {'='.join(str(y) for y in sorted(list(x))) for x in (edges[i] for i in range(edges.shape[0]))}


def edge_set_diffs(dataset):
    return _filter_generator(dataset, lambda i, o: edge_set(i) != edge_set(o))


def node_count_reduces(dataset):
    return _filter_generator(node_count_diffs(dataset), lambda i, o: i.x.shape[0] > o.x.shape[0])


def generate_report(data_path, num_samples=None):
    dataset, failed = load_dataset(data_path, num_samples=num_samples)
    filtered_dataset = filter_dataset(dataset, keep_differ_edges=True)

    samples = dataset.values()
    samples_filtered = filtered_dataset.values()
    samples_filtered_edges_required = filter_dataset(dataset, keep_edgeless=False).values()

    failed_completely = {idx for idx, _, c in failed if c}
    failed_malformed = {idx for idx, _, c in failed if not c and idx not in failed_completely}

    node_feats_changed_set = set(x.idx for x in node_feats_changed(samples_filtered))
    edge_feats_changed_set = set(x.idx for x in edge_feats_changed(samples_filtered_edges_required))

    node_feats_changed_len = len(node_feats_changed_set)
    edge_feats_changed_len = len(edge_feats_changed_set)

    any_feats_changed = node_feats_changed_set.union(edge_feats_changed_set)
    any_feats_changed_len = len(any_feats_changed)
    both_feats_changed_len = len(node_feats_changed_set.intersection(edge_feats_changed_set))

    def _p(x):
        return f'{100 * x / len(dataset) :02.01f}%'

    out_md = f"""## Статистика по датасету

- Всего пар: `{len(dataset) + len(failed_completely)}`
    - **Загружено**: `{len(dataset)}`
        - Из них неполные: `{len(failed_malformed)}`
    - Не загружено: `{len(failed_completely)}`

- Различаются:
    - Кол-во атомов: `{_l(node_count_diffs(samples))}`
        - Кол-во уменьшается: `{_l(node_count_reduces(samples))}`
    - Кол-во связей: `{_l(edge_count_diffs(samples))}`
        - При равном кол-ве атомов: `{_l(node_count_equals_edge_count_diffs(samples))}`

- Отсутствуют:
    - Атомы: `{_l(no_nodes(samples))}`
    - Связи: `{_l(no_edges(samples))}`
    - Связи, но атомов > 1: `{_l(no_edges_multiple_nodes(samples))}`

- Атрибуты:
    - Меняются для атомов: `{node_feats_changed_len} ({_p(node_feats_changed_len)})`
    - Меняются для связей: `{edge_feats_changed_len} ({_p(edge_feats_changed_len)})`
    - Меняются для атомов и связей: `{both_feats_changed_len} ({_p(both_feats_changed_len)})`
    - **Меняются для атомов или связей**: `{any_feats_changed_len} ({_p(any_feats_changed_len)})`

- Множество связей:
    - Меняется: `{_l(edge_set_diffs(samples))}`
        - **Но атрибуты - не меняются** `{_l(s for s in edge_set_diffs(samples) if s.idx not in any_feats_changed)}`
    """

    return out_md


if __name__ == '__main__':
    out_md = generate_report(sys.argv[1])
    with open(Path(sys.argv[1]) / 'data_stats.md', 'w', encoding='utf-8') as f:
        f.write(out_md)

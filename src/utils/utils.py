import itertools
import os
import random

import numpy as np
import torch
import pandas as pd
from typing import Optional, Tuple
from torch_geometric.utils import scatter
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.model.tf_model import TF_model

# Build a namespace-like object for Parameters
class DictNamespace:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def get_model(run_params):
    if run_params.model in ['gcn', 'gat', 'mlp', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU', 'DCRNN', 'GraphWavenet', 'AGCRNModel', 'miniLSTM', 'miniGRU']:
        model = TF_model(run_params)
    else:
        raise Exception('Error in select the model!')
    return model


def directed_to_undirected(edges: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Convert a directed edge list to an undirected edge list by adding the reverse of each edge.
    Also handles edge weights by duplicating them for the reverse edges and sorting them.

    Args:
        edges (torch.Tensor): A tensor of shape (2, E) representing directed edges (i, j).
        edge_weights (Optional[torch.Tensor]): A tensor of shape (E,) representing weights for each directed edge.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - A tensor of shape (2, 2*E) representing undirected edges (i, j) and (j, i).
            - A tensor of shape (2*E,) representing weights for the undirected edges, or None if no weights are provided.
    """
    # Aggiungi gli archi inversi (j, i)
    reversed_edges = edges.flip(0)

    # Concatenare gli archi originali e quelli inversi
    undirected_edges = torch.cat([edges, reversed_edges], dim=1)

    # Ordina gli archi in base alla prima dimensione (i)
    sorted_indices = undirected_edges[0].argsort()
    undirected_edges = undirected_edges[:, sorted_indices]

    # Se sono stati passati i pesi, duplicarli e ordinarli
    if edge_weights is not None:
        reversed_weights = edge_weights.clone()
        undirected_weights = torch.cat([edge_weights, reversed_weights], dim=0)
        undirected_weights = undirected_weights[sorted_indices]
        return undirected_edges, undirected_weights

    return undirected_edges, None


def edge_to_node_aggregation(edge_index, edge_attr, num_nodes):
    """ edge_index: edges indices (2, E)
        edge_attr: Characteristics of the edges (E, F)
        num_nodes: Number of nodes in the graph
        edge_index[0]: Indexes of the origin nodes of the edges
        edge_index[1]: Indexes of the destination nodes of the edges
        """
    edge_index = edge_index[:, :int(edge_index.shape[1]/2)]

    # Aggregate the features of the edges on the source nodes
    node_features = scatter(edge_attr.cpu(), edge_index[0].cpu(), dim=0, dim_size=num_nodes, reduce='mean')

    # Aggregate the arc features on the target nodes
    node_features += scatter(edge_attr.cpu(), edge_index[1].cpu(), dim=0, dim_size=num_nodes, reduce='mean')

    return node_features/2


def get_callbacks(run_params):
    # Callbacks
    callbacks = list()
    checkpoint_callback = ModelCheckpoint(
        dirpath='../checkpoints',
        save_last=True,
        filename="{epoch}-{val_loss:.2f}",  # naming
        save_top_k=2,
        verbose=True,
        monitor='val_mse',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_mse',
        min_delta=0.00,
        patience=4,
        verbose=False,
        mode='min')

    if run_params.save_ckpts:
        callbacks += [checkpoint_callback]

    if run_params.early_stopping:
        callbacks += [early_stop_callback]

    return callbacks


def build_combinations(search_space: dict) -> list:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def initialize_log_parameters(cont: int, combo: dict) -> dict:
    METRICS = ['mse', 'rmse', 'mae']
    SPLITS = ['val', 'test']

    # colonne metriche: val_mse_mean, val_mse_std, ...
    metric_keys = [f'{split}_{metric}_{stat}'
                   for split in SPLITS
                   for metric in METRICS
                   for stat in ('mean', 'std')]

    grid_params = {'Run': cont, **combo, **{k: 0. for k in metric_keys}}

    print(' '.join(f'{k}: {v}' for k, v in grid_params.items()))
    return grid_params



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def update_seed_metrics(model, res_test, val_results, test_results):
    best_val_mse, best_val_rmse, best_val_mae = model.best_mse, model.best_rmse, model.best_mae

    # Testing
    test_mse = res_test[0]['test_mse']
    test_rmse = res_test[0]['test_rmse']
    test_mae = res_test[0]['test_mae']

    val_results.append([best_val_mse, best_val_rmse, best_val_mae])
    test_results.append([test_mse, test_rmse, test_mae])

    print(f'best_val_mse: {best_val_mse}')
    print(f'best_val_rmse: {best_val_rmse}')
    print(f'best_val_mae: {best_val_mae}')
    print(f'test_mse: {test_mse}')
    print(f'test_rmse {test_rmse}')
    print(f'test_mae: {test_mae}')
    return val_results, test_results

def update_run_metrics(val_results,
                       test_results,
                       grid_params_dict,
                       run_params):
    metrics = ['mse', 'rmse', 'mae']
    splits = ['val', 'test']
    results = {'val':  torch.tensor(val_results),
               'test': torch.tensor(test_results)}

    grid_params_dict.update({f'{split}_{metric}_{stat}': float(getattr(torch, stat)(results[split][:, i]))
                            for split in splits
                            for i, metric in enumerate(metrics)
                            for stat in ('mean', 'std')})

    print(' '.join(f'{k}: {v}' for k, v in grid_params_dict.items()))
    output_string = ' '.join([f'{k}: {v}' for k, v in grid_params_dict.items()])

    if run_params.save_logs:
        os.makedirs(run_params.logs_dir, exist_ok=True)
        with open(os.path.join(run_params.logs_dir, 'log.txt'), 'a') as file:
            print(output_string, file=file)
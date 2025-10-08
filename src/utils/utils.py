import torch
import pandas as pd
from typing import Optional, Tuple
from torch_geometric.utils import scatter
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.model.tf_model import TF_model


def get_model(run_params):
    if run_params.model in ['gcn', 'gat', 'mlp', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU', 'DCRNN', 'GraphWavenet', 'AGCRNModel', 'miniLSTM', 'miniGRU']:
        model = TF_model(run_params)
    else:
        raise Exception('Error in select the model!')
    return model



def directed_to_undirected(edges: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> Tuple[
    torch.Tensor, Optional[torch.Tensor]]:
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
    # edge_index: Indici degli archi (2, E)
    # edge_attr: Caratteristiche degli archi (E, F)
    # num_nodes: Numero di nodi nel grafo

    # edge_index[0]: Indici dei nodi di origine degli archi
    # edge_index[1]: Indici dei nodi di destinazione degli archi
    edge_index = edge_index[:, :int(edge_index.shape[1]/2)]

    # Aggrega le caratteristiche degli archi sui nodi di origine
    node_features = scatter(edge_attr.cpu(), edge_index[0].cpu(), dim=0, dim_size=num_nodes, reduce='sum')
    # Aggrega le caratteristiche degli archi sui nodi di destinazione
    node_features += scatter(edge_attr.cpu(), edge_index[1].cpu(), dim=0, dim_size=num_nodes, reduce='sum')

    return node_features



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



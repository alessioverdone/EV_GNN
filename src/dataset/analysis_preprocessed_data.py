import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import os

from src.config import Parameters


def check_inference_values(dataset='chicago',
                           id='005'):
    repo_csv = f'../../registry/inference_outputs/{dataset}/{id}/csv_out'
    print(f'Controlling inference data at {repo_csv}!')


    for elem in sorted(os.listdir(repo_csv)):
        if elem.endswith('.csv'):
            pred = pd.read_csv(repo_csv + '/' + elem)

            pred_np = np.array(pred)[:, 1:]
            pred_pt = torch.tensor(np.array(pred_np, dtype=float)[:, 1:])
            id_elem_neg = torch.stack(torch.where(pred_pt == -1.))

            print(f'{elem}: {id_elem_neg.shape[-1]}, min:{pred_np.min()}, max: {pred_np.max()}')


def check_negative_values_in_dataset(dataset='chicago'):
    dataset_path = f'../../data/processed/{dataset}/final_temporal_merged_data.pt'
    print(f'Controlling inference data at {dataset_path}!')

    db = torch.load(dataset_path)
    id_elem_neg = torch.stack(torch.where(db <= -1.))
    print(f'-1 elems in {dataset_path}: {id_elem_neg.shape[-1]}')
    print(f'shape: {db.shape}')
    print(f'min values: {db.min()}')
    print(f'max values: {db.max()}')

    plot_site(db)


def plot_site(db, site_id=None):
    if site_id is None:
        site_id = random.randint(0, db.shape[0])
    print(f'site_id: {site_id}')
    plt.plot(db[site_id,:,0])
    plt.show()


def analyze_processed_data(dataset='chicago',):
    preprocessed_data_path = f'../../data/processed/{dataset}'
    print(f'Controlling preprocessed data at {preprocessed_data_path}!')
    addes_edges_df  =pd.read_csv(os.path.join(preprocessed_data_path, 'added_edges_df.csv'))
    edges_df = pd.read_csv(os.path.join(preprocessed_data_path, 'edges_df.csv'))
    nodes_df = pd.read_csv(os.path.join(preprocessed_data_path, 'nodes_df.csv'))
    dataset_config = torch.load(os.path.join(preprocessed_data_path, 'dataset_config.pt'), weights_only=False)
    edge_index_traffic = torch.load(os.path.join(preprocessed_data_path, 'edge_index_traffic.pt'), weights_only=False)
    edge_weights_traffic = torch.load(os.path.join(preprocessed_data_path, 'edge_weights_traffic.pt'), weights_only=False)
    final_temporal_merged_data = torch.load(os.path.join(preprocessed_data_path, 'final_temporal_merged_data.pt'), weights_only=False)
    map_ev_node_traffic_node = torch.load(os.path.join(preprocessed_data_path, 'map_ev_node_traffic_node.pt'), weights_only=False)
    map_real_ev_node_traffic_node = torch.load(os.path.join(preprocessed_data_path, 'map_real_ev_node_traffic_node.pt'), weights_only=False)
    time_column = torch.load(os.path.join(preprocessed_data_path, 'time_column.pt'), weights_only=False)

    print(f'addes_edges_df: {addes_edges_df.shape}')
    print(f'edges_df: {edges_df.shape}')
    print(f'nodes_df: {nodes_df.shape}')
    print(f'dataset_config: {len(dataset_config.keys())}')
    print(f'edge_index_traffic: {edge_index_traffic.shape}')
    print(f'edge_weights_traffic: {edge_weights_traffic.shape}')
    print(f'final_temporal_merged_data: {final_temporal_merged_data.shape}')
    print(f'map_ev_node_traffic_node: {len(map_ev_node_traffic_node.keys())}')
    print(f'map_real_ev_node_traffic_node: {len(map_real_ev_node_traffic_node.keys())}')
    print(f'time_column: {time_column.shape}')
    #TODO: Controlla che valori EV non vengano normalizzati
    #TODO: Fai README su preprocessing dati e condividi predizione, tipo di lavoro (si lavora su grafo processato perchè nodi di traffico mergiati)


def analyze_chicago(dataset: str = 'chicago') -> None:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
    sys.path.insert(0, PROJECT_ROOT)

    params = Parameters()
    processed_path = os.path.join(PROJECT_ROOT, 'data', 'processed', dataset)

    nodes_df = pd.read_csv(os.path.join(processed_path, 'nodes_df.csv'))
    edges_df = pd.read_csv(os.path.join(processed_path, 'edges_df.csv'))
    added_edges_df = pd.read_csv(os.path.join(processed_path, 'added_edges_df.csv'))
    time_column = torch.load(os.path.join(processed_path, 'time_column.pt'), weights_only=False)
    merged_nodes_map = torch.load(os.path.join(processed_path, 'merged_traffic_nodes_map.pt'), weights_only=False)
    map_ev = torch.load(os.path.join(processed_path, 'map_ev_node_traffic_node.pt'), weights_only=False)

    # --- dataset & graph values ---
    raw_endpoints = len(merged_nodes_map)  # original coord IDs before spatial merging
    merged_nodes = len(nodes_df)  # nodes after merging (threshold ~100 m)
    real_edges = len(edges_df) - len(added_edges_df)
    virtual_edges = len(added_edges_df)  # MST edges added for connectivity
    ev_stations = len(map_ev)

    start = time_column.iloc[0]
    end = time_column.iloc[-1]
    span = end - start

    diffs = time_column.diff().dropna()
    sampling_min = round(diffs.median().total_seconds() / 60, 1)

    # --- sliding-window split (uses config defaults) ---
    T = len(time_column)
    lags = params.lags
    pred_win = params.prediction_window
    step = params.time_series_step

    total_windows = len(range(0, T - lags - pred_win, step))
    train_windows = int(params.train_ratio * total_windows)
    val_test = total_windows - train_windows
    val_windows = int(params.val_test_ratio * val_test)
    test_windows = val_test - val_windows

    # --- output ---
    sep = "=" * 56
    print(sep)
    print(f"  Dataset & Graph Summary — {dataset.capitalize()}")
    print(sep)
    print(f"  {'Raw segment endpoints':<28}: {raw_endpoints:>8,}")
    print(f"  {'Merged nodes':<28}: {merged_nodes:>8,}")
    print(f"  {'Real edges':<28}: {real_edges:>8,}")
    print(f"  {'Virtual edges (MST)':<28}: {virtual_edges:>8,}")
    print(f"  {'EV stations':<28}: {ev_stations:>8,}")
    print(f"  {'Time span':<28}: {start.date()} → {end.date()}  ({span.days} days)")
    print(f"  {'Sampling resolution':<28}: {sampling_min} min")
    print(f"  {'Total timesteps':<28}: {T:>8,}")
    print("-" * 56)
    print(f"  Sliding windows  lag={lags}  pred={pred_win}  step={step}")
    print(f"  {'Train windows':<28}: {train_windows:>8,}  ({params.train_ratio:.0%})")
    print(f"  {'Val windows':<28}: {val_windows:>8,}  ({params.val_test_ratio:.0%} of remainder)")
    print(f"  {'Test windows':<28}: {test_windows:>8,}")
    print(sep)


if __name__ == '__main__':
    check_inference_values()
    check_negative_values_in_dataset()
    analyze_processed_data()
    analyze_chicago()
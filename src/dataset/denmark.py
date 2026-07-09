import glob
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from tsl.data.preprocessing import MinMaxScaler
from torch_geometric.data import Data

from src.dataset.utils import (create_adjacency_matrix, haversine)


class Dataset_Denmark(Dataset):
    _DEFAULT_COLUMNS = [
        "avgMeasuredTime",
        "avgSpeed",
        "extID",
        "medianMeasuredTime",
        "TIMESTAMP",
        "vehicleCount"]

    def __init__(self, params, columns=None, dtype=torch.float32, device="cuda"):
        self.params = params
        self.columns = columns or self._DEFAULT_COLUMNS
        self.dtype = dtype
        self.device = torch.device(device)
        self.encoded_data = []
        self.lags = self.params.lags
        self.prediction_window = self.params.prediction_window
        self.time_series_step = self.params.time_series_step

        # Variables
        self.number_of_station = None
        self.features = None
        self.targets = None
        self.edge_index_traffic = None
        self.edge_weights_traffic = None
        self.coordinates_traffic = list()

        self.filepaths = sorted(glob.glob(os.path.join(self.params.traffic_temporal_data_folder, "*.csv")))
        if not self.filepaths:
            raise RuntimeError(f"Nessun CSV trovato in {self.params.traffic_temporal_data_folder}")

        # Dataset construction
        self._load_traffic_data()  # get self.data_tensor_traffic
        self._load_ev_data()  # get self.data_tensor_ev
        self.get_edges_traffic(threshold=self.params.graph_distance_threshold)
        self.assign_ev_node_to_traffic_node()
        self.preprocess_data()
        self.assemble_dataset()

    def _load_ev_data(self, pad_value=-1.0):
        ev_columns = ['time_stamp', 'available_count']

        dfs = []  # DataFrame list per site
        sites = []  # site ID
        print('Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv")))[
                    :self.params.num_of_ev_nodes_limit]:
            df = pd.read_csv(path, usecols=ev_columns)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            df = df.sort_values("time_stamp").reset_index(drop=True)

            # Set ts index
            ts = pd.to_datetime(df["time_stamp"])
            df["time_stamp"] = ts
            df = df.set_index("time_stamp")

            # Check and substitute duplicate values
            if df.index.has_duplicates:
                print(f'Find duplicate values in {os.path.basename(path)}')
                strategy = 'mean'  # ['discard', 'mean']
                if strategy == 'discard':
                    df = df[~df.index.duplicated(keep="first")]
                elif strategy == 'mean':
                    df = df.groupby(level=0).mean()
                else:
                    raise ValueError(strategy)

            # Cast
            feature_cols = [c for c in ev_columns if c != "time_stamp"]
            df = df[feature_cols]  # mantiene ordine voluto
            df = df.apply(pd.to_numeric, errors="coerce")
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]

            dfs.append(df)
            sites.append(site_id)

        # Index outer join
        union_index = dfs[0].index
        for d in dfs[1:]:
            union_index = union_index.union(d.index)

        # Realign and constant padding
        dfs_aligned = [d.reindex(union_index).fillna(pad_value) for d in dfs]

        # Concat
        df_all = pd.concat(dfs_aligned, axis=1, keys=sites, names=["site", "feature"])
        self.df_ev_all = df_all.sort_index()  # (T_max, N*M)

        # Build EV tensor
        data = np.stack([d.values.astype(np.float32) for d in dfs_aligned], axis=0)
        self.data_tensor_ev = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.N_ev, self.T_ev, self.M_ev = self.data_tensor_ev.shape
        print(f'Loaded EV data: {self.N_ev} nodes, {self.T_ev} timesteps, {self.M_ev} features')

    def _load_traffic_data(self, pad_value=-1.0):
        dfs = []  # DataFrame list per site
        sites = []  # siet id
        print('Loading Traffic data ...')
        for path in self.filepaths[:self.params.num_of_traffic_nodes_limit]:
            df = pd.read_csv(path, usecols=self.columns)

            # Control missing columns
            missing = set(self.columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            df = df.sort_values("TIMESTAMP").reset_index(drop=True)

            # Set ts index
            ts = pd.to_datetime(df["TIMESTAMP"])
            df["TIMESTAMP"] = ts
            df = df.set_index("TIMESTAMP")

            # Check and substitute duplicate values
            if df.index.has_duplicates:
                print(f'Find duplicate values in {os.path.basename(path)}')
                strategy = 'mean'  # ['discard', 'mean']
                if strategy == 'discard':
                    df = df[~df.index.duplicated(keep="first")]
                elif strategy == 'mean':
                    df = df.groupby(level=0).mean()
                else:
                    raise ValueError(strategy)

            # Cast
            feature_cols = [c for c in self.columns if c != "TIMESTAMP"]
            df = df[feature_cols]  # mantiene ordine voluto
            df = df.apply(pd.to_numeric, errors="coerce")
            site_id = str(df["extID"].iloc[0])  # o usa os.path.basename(path).split(".")[0]

            dfs.append(df)
            sites.append(site_id)

        # Index outer join
        union_index = dfs[0].index
        for d in dfs[1:]:
            union_index = union_index.union(d.index)

        # Realign and constant padding
        dfs_aligned = [d.reindex(union_index).fillna(pad_value) for d in dfs]

        # Concat
        df_all = pd.concat(dfs_aligned, axis=1, keys=sites, names=["site", "feature"])
        self.df_all = df_all.sort_index()  # (T_max, N*M)

        # Build Traffic tensor
        self.timestamp_final_traffic = union_index
        data = np.stack([d.values.astype(np.float32) for d in dfs_aligned], axis=0)
        self.data_tensor_traffic = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
        print(f'Loaded Traffic data: {self.N_t} nodes, {self.T_t} timesteps, {self.M_t} features')

    def get_edges_traffic(self, threshold=10000):
        # Load traffic metadata
        data = pd.read_csv(self.params.traffic_metadata_file)
        for row in data.iterrows():
            p1_lat = row[1]['POINT_1_LAT']
            p1_long = row[1]['POINT_1_LNG']
            p2_lat = row[1]['POINT_2_LAT']
            p2_long = row[1]['POINT_2_LNG']
            p_mean_lat = (p1_lat + p2_lat) / 2.0
            p_mean_lng = (p1_long + p2_long) / 2.0
            self.coordinates_traffic.append((p_mean_lat, p_mean_lng))

        # Create graph based on distance threshold (in Km)
        adj_matrix = create_adjacency_matrix(self.coordinates_traffic[:self.params.num_of_traffic_nodes_limit],
                                             threshold)
        edge_index, edge_weights = dense_to_sparse(adj_matrix)
        self.edge_index_traffic, self.edge_weights_traffic = edge_index.to('cuda'), edge_weights.to('cuda')

    def assign_ev_node_to_traffic_node(self):
        # Load EV metadata
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)

        # Get EV coordinates
        ev_coordinates = list()
        for row in ev_metadata.iterrows():
            lat, lng = row[1]['lat'], row[1]['lng']
            ev_coordinates.append((lat, lng))

        # Map each EV node to nearest traffic node
        self.map_ev_node_traffic_node = {}
        for ev_node_idx, ev_coord in enumerate(ev_coordinates[:self.params.num_of_ev_nodes_limit]):
            min_dist = float('inf')
            min_dist_traffic_node_idx = -1
            for traffic_node_idx, traffic_coord in enumerate(
                    self.coordinates_traffic[:self.params.num_of_traffic_nodes_limit]):
                lat1, lon1 = ev_coord
                lat2, lon2 = traffic_coord
                distance = haversine(lat1, lon1, lat2, lon2)
                if distance < min_dist:
                    min_dist = distance
                    min_dist_traffic_node_idx = traffic_node_idx
            self.map_ev_node_traffic_node[ev_node_idx] = min_dist_traffic_node_idx

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to self.map_ev_node_traffic_node: create a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(self.N_t)]
        for key in self.map_ev_node_traffic_node.keys():
            corrispective_traffic_node = self.map_ev_node_traffic_node[key]
            ev_values = self.data_tensor_ev[key]
            temp_list[corrispective_traffic_node].append(ev_values)

        lista_max = max(temp_list, key=len)
        tensor_ev_temp = torch.stack(lista_max)
        ev_timesteps = tensor_ev_temp.shape[1]

        new_temp_list = list()
        for elem in temp_list:
            if len(elem) == 0:
                new_temp_list.append(torch.zeros(ev_timesteps, 1).to(self.params.device))
            elif len(elem) == 1:
                new_temp_list.append(elem[0])
            else:
                new_temp_list.append(torch.stack(elem).sum(0).squeeze(0))

        # inner join con self.timestamp_final_traffic e check sincronicità
        ev_temporal_data_on_merged_nodes = torch.stack(new_temp_list)

        # ------------------------------------------ Temporary setup --------------------------------------------------
        # To delete! Modification to geenrate same number of ev and traffic timesteps
        # Repeat EV_timesteps dimension to match Traffic timesteps dimn
        repetition = int(self.data_tensor_traffic.shape[1] / ev_temporal_data_on_merged_nodes.shape[1]) + 1
        ev_temporal_data_on_merged_nodes = ev_temporal_data_on_merged_nodes.repeat(1, repetition, 1)
        ev_temporal_data_on_merged_nodes = ev_temporal_data_on_merged_nodes[:, :self.data_tensor_traffic.shape[1], :]
        self.data_tensor_merged = torch.cat([self.data_tensor_traffic, ev_temporal_data_on_merged_nodes], dim=-1)

        # 4 and 5 column order swapped in order to have last feature as label
        order = [0, 1, 2, 3, 5, 4]
        self.data_tensor_merged = self.data_tensor_merged[:, :, order]
        # ------------------------------------------------- End --------------------------------------------------------

        # TODO: fai grafico e diverse modalità di creazione grafo per visualizzare risultati
        # TODO: aumenta dimensioni timesteps per eguagliare traffic timesteps (dati sincroni) e hai finito

    def preprocess_data(self):
        stacked_target = self.data_tensor_merged.to('cpu')
        stacked_target = stacked_target[:, :, -1]

        # Normalization
        scaler = MinMaxScaler()
        scaler.fit(stacked_target)
        standardized_target = scaler.transform(stacked_target).T
        self.number_of_station = standardized_target.shape[1]

        # Input data
        self.features = [standardized_target[i: i + self.lags, :].T
                         for i in range(0, standardized_target.shape[0] - self.lags - self.prediction_window,
                                        self.time_series_step)]

        # Output data
        self.targets = [standardized_target[i:i + self.prediction_window, :].T
                        for i in range(self.lags, standardized_target.shape[0] - self.prediction_window,
                                       self.time_series_step)]

    def assemble_dataset(self):
        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=self.edge_index_traffic.long(),
                                          edge_attr=self.edge_weights_traffic.float(),
                                          y=torch.FloatTensor(self.targets[i])))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


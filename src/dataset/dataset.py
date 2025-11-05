<<<<<<< HEAD
import argparse
import glob
import json
import random
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as DataLoaderPyg
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from datetime import datetime, timedelta

from tsl.data import SpatioTemporalDataset, TemporalSplitter, SpatioTemporalDataModule
from tsl.data.preprocessing import MinMaxScaler
from tsl.datasets import MetrLA, Elergone
from torch_geometric.data import Data

from src.config import Parameters
from src.dataset.resamplling import resample_to_common_time
from src.dataset.utils import (create_adjacency_matrix, haversine, build_edges_with_node_ids, \
    create_adjacency_matrix_newyork, augment_graph_df_v3, append_along_N_torch, build_edges_with_node_ids_chicago,
                               process_traffic_metadata_newyork)
from src.utils.utils import directed_to_undirected, edge_to_node_aggregation


class EVDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.run_params = params

        # Denmark dataset
        if self.run_params.dataset_name == 'denmark':
            self.run_params.traffic_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                        'data',
                                                                        self.run_params.dataset_name,
                                                                        'traffic/citypulse_traffic_raw_data_surrey_feb_jun_2014')
            self.run_params.traffic_metadata_file = os.path.join(self.run_params.project_path,
                                                                 'data',
                                                                 self.run_params.dataset_name,
                                                                 'traffic/trafficMetaData.csv')
            self.run_params.ev_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                   'data',
                                                                   self.run_params.dataset_name,
                                                                   'ev/available_connectors_counts')
            self.run_params.ev_metadata_file = os.path.join(self.run_params.project_path,
                                                            'data',
                                                            self.run_params.dataset_name,
                                                            'ev/charging_stations.csv')
            dataset = Dataset_Denmark(self.run_params)

        # Newyork dataset
        elif self.run_params.dataset_name == 'newyork':
            # _TRAFFIC_DATA_COLUMNS = ["speed", "travel_time", "status", "data_as_of", "link_id"]
            # _TRAFFIC_METADATA_COLUMNS = ["id", "link_points", "points", "distances"]
            # _EV_DATA_COLUMNS = ["location_id", "timestamp", "Available", "Total", "Offline"]
            # _EV_METADATA_COLUMNS = ["LocID", "LocName", "Latitude", "Longitude"]
            self.run_params.traffic_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                        'data',
                                                                        self.run_params.dataset_name,
                                                                        'traffic/traffic_data')
            self.run_params.traffic_metadata_file = os.path.join(self.run_params.project_path,
                                                                 'data',
                                                                 self.run_params.dataset_name,
                                                                 'traffic/processed_newyork_traffic_graph.csv')
            self.run_params.ev_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                   'data',
                                                                   self.run_params.dataset_name,
                                                                   'ev/stations_connectors_counts_data')
            self.run_params.ev_metadata_file = os.path.join(self.run_params.project_path,
                                                            'data',
                                                            self.run_params.dataset_name,
                                                            'ev/location_meta_data.csv')
            self.run_params.traffic_columns_to_use =  ["speed", "travel_time"]
            self.run_params.ev_columns_to_use = ["Available"]
            dataset = DatasetNewyork(self.run_params)
            print(f'Dataset starting time: {dataset.start_time}')
            print(f'Dataset end time: {dataset.end_time}')
            print(f'Dataset traffic mean resolution: {dataset.traffic_resolution}')
            print(f'Dataset ev mean resolution: {dataset.ev_resolution}')


        # Chicago dataset
        elif self.run_params.dataset_name == 'chicago':
            self.run_params.traffic_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                        'data',
                                                                        self.run_params.dataset_name,
                                                                        'traffic/traffic')
            self.run_params.traffic_metadata_file = os.path.join(self.run_params.project_path,
                                                                 'data',
                                                                 self.run_params.dataset_name,
                                                                 'traffic/location_summary.csv')
            self.run_params.ev_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                   'data',
                                                                   self.run_params.dataset_name,
                                                                   'ev/ev_locations_availability')
            self.run_params.ev_metadata_file = os.path.join(self.run_params.project_path,
                                                            'data',
                                                            self.run_params.dataset_name,
                                                            'ev/ev_location_metadata.csv')
            self.run_params.traffic_columns_to_use =  ["speed", "lenght"]
            self.run_params.ev_columns_to_use = ["Available"]
            dataset = DatasetChicago(self.run_params)
            print(f'Dataset starting time: {dataset.start_time}')
            print(f'Dataset end time: {dataset.end_time}')
            print(f'Dataset traffic mean resolution: {dataset.traffic_resolution}')
            print(f'Dataset ev mean resolution: {dataset.ev_resolution}')

        else:
            raise ValueError(f'Dataset {self.run_params.dataset_name} not recognized')

        self.num_station = dataset.number_of_station
        self.run_params.num_nodes = self.num_station
        self.run_params.traffic_features = dataset.traffic_features
        self.run_params.ev_features = dataset.ev_features
        self.run_params.traffic_features_names = dataset.traffic_columns_used_in_data
        self.run_params.ev_features_names = dataset.ev_columns_used_in_data
        self.run_params.min_vals_normalization = dataset.min_vals_normalization
        self.run_params.max_vals_normalization = dataset.max_vals_normalization
        self.run_params.start_time = dataset.start_time
        self.run_params.end_time = dataset.end_time
        self.run_params.traffic_resolution = dataset.traffic_resolution
        self.run_params.ev_resolution = dataset.ev_resolution
        # self.run_params.start_time_test = dataset.start_time_test
        # self.run_params.end_time_test = dataset.end_time_test

        # Split data
        split_data_modality = 'sequential'  # ['random', 'sequential']
        if  split_data_modality == 'random':
            len_dataset = len(dataset)
            train_snapshots = int(self.run_params.train_ratio * len_dataset)
            val_test_snapshots = len_dataset - train_snapshots
            val_snapshots = int(self.run_params.val_test_ratio * val_test_snapshots)
            test_snapshots = len_dataset - train_snapshots - val_snapshots
            self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset,[train_snapshots,
                                                                                                    val_snapshots,
                                                                                                    test_snapshots])  # N, T, F
        elif split_data_modality == 'sequential':
            len_dataset = len(dataset)
            train_snapshots = int(self.run_params.train_ratio * len_dataset)
            val_test_snapshots = len_dataset - train_snapshots
            val_snapshots = int(self.run_params.val_test_ratio * val_test_snapshots)
            test_snapshots = len_dataset - train_snapshots - val_snapshots
            start_point = random.randint(0, len_dataset - test_snapshots)
            test_start = start_point
            test_end = start_point + test_snapshots
            train_start = test_end
            train_end = train_start + train_snapshots
            val_start = train_end
            val_end = val_start + val_snapshots
            dataset_indices = np.arange(len_dataset).tolist() + np.arange(len_dataset).tolist()  # 2 times because restart from the beginning
            train_idx = dataset_indices[train_start:train_end]
            val_idx = dataset_indices[val_start:val_end]
            test_idx = dataset_indices[test_start:test_end]
            self.train_data = torch.utils.data.Subset(dataset, train_idx)
            self.val_data = torch.utils.data.Subset(dataset, val_idx)
            self.test_data = torch.utils.data.Subset(dataset, test_idx)

            # Here i'm sure by construction that test doesn't overlap the end of the dataset
            self.run_params.start_time_test = list(self.test_data[0].time_input)[0]
            self.run_params.end_time_test = list(self.test_data[len(self.test_data)-1].time_output)[-1]

            # I don't need for train and val for now since i will utilize only test values
            # # If train idx don't restart from the beginning
            # if train_start < train_end:
            #     self.start_time_train = self.end_time_test
            #     self.end_time_train = self.start_time_train + timedelta(seconds=(train_end - train_start) * dataset.traffic_resolution * 60)
            # # If train idx restart from the beginning
            # else:
            #     self.start_time_train_0 = self.end_time_test
            #     self.end_time_train_0 = self.start_time_train + timedelta(seconds=(train_end - train_start) * dataset.traffic_resolution * 60)
            #
            # if val_start < val_end:
            #     self.start_time_val = self.end_time_train
            #     self.end_time_val = self.start_time_val + timedelta(seconds=(val_end - val_start) * dataset.traffic_resolution * 60)
            # else:
            #     pass
            # self.start_time_train = dataset.start_time + timedelta(seconds=train_start * dataset.traffic_resolution * 60)
            # self.end_time_train = dataset.start_time + timedelta(seconds=train_end * dataset.traffic_resolution * 60)
            # self.start_time_val = dataset.start_time + timedelta(seconds=val_start * dataset.traffic_resolution * 60)
            # self.end_time_val = dataset.start_time + timedelta(seconds=val_end * dataset.traffic_resolution * 60)


        else:
            raise ValueError(f'split_data_modality not recognized')

        # Check integrity
        if self.train_data is None:
            raise Exception("Dataset %s not supported" % self.run_params.dataset)

        # Get dataloaders
        self.train_loader = DataLoaderPyg(self.train_data,
                                          batch_size=self.run_params.batch_size,
                                          shuffle=True,
                                          drop_last=True)
        self.val_loader = DataLoaderPyg(self.val_data,
                                        batch_size=self.run_params.batch_size,
                                        drop_last=True)
        self.test_loader = DataLoaderPyg(self.test_data,
                                         batch_size=self.run_params.batch_size,
                                         drop_last=True)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


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
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv")))[:self.params.num_of_ev_nodes_limit]:
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
        dfs = []  #  DataFrame list per site
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

    def get_edges_traffic(self, threshold = 10000):
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
        adj_matrix = create_adjacency_matrix(self.coordinates_traffic[:self.params.num_of_traffic_nodes_limit], threshold)
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
        map_ev_node_traffic_node = {}
        for ev_node_idx, ev_coord in enumerate(ev_coordinates[:self.params.num_of_ev_nodes_limit]):
            min_dist = float('inf')
            min_dist_traffic_node_idx = -1
            for traffic_node_idx, traffic_coord in enumerate(self.coordinates_traffic[:self.params.num_of_traffic_nodes_limit]):
                lat1, lon1 = ev_coord
                lat2, lon2 = traffic_coord
                distance = haversine(lat1, lon1, lat2, lon2)
                if distance < min_dist:
                    min_dist = distance
                    min_dist_traffic_node_idx = traffic_node_idx
            map_ev_node_traffic_node[ev_node_idx] = min_dist_traffic_node_idx

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to map_ev_node_traffic_node: create a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(self.N_t)]
        for key in map_ev_node_traffic_node.keys():
            corrispective_traffic_node = map_ev_node_traffic_node[key]
            ev_values = self.data_tensor_ev[key]
            temp_list[corrispective_traffic_node].append(ev_values)

        lista_max = max(temp_list, key=len)
        tensor_ev_temp = torch.stack(lista_max)
        ev_timesteps = tensor_ev_temp.shape[1]

        new_temp_list = list()
        for elem in temp_list:
            if len(elem) == 0:
                new_temp_list.append(torch.zeros(ev_timesteps,1).to(self.params.device))
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
        self.data_tensor_merged = torch.cat([self.data_tensor_traffic,ev_temporal_data_on_merged_nodes], dim=-1)

        # 4 and 5 column order swapped in order to have last feature as label
        order = [0, 1, 2, 3, 5, 4]
        self.data_tensor_merged = self.data_tensor_merged[:, :, order]
        # ------------------------------------------------- End --------------------------------------------------------

        #TODO: fai grafico e diverse modalità di creazione grafo per visualizzare risultati
        # TODO: aumenta dimensioni timesteps per eguagliare traffic timesteps (dati sincroni) e hai finito


    def preprocess_data(self):
        stacked_target = self.data_tensor_merged.to('cpu')
        stacked_target = stacked_target[:,:,-1]

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


class DatasetNewyork(Dataset):
    _TRAFFIC_DATA_COLUMNS = ["speed","travel_time","status","data_as_of","link_id"]
    _TRAFFIC_METADATA_COLUMNS = ["id","link_points","points","distances"]
    _EV_DATA_COLUMNS = ["location_id","timestamp","Available","Total","Offline"]
    _EV_METADATA_COLUMNS = ["LocID","LocName","Latitude","Longitude"]

    def __init__(self, params, dtype=torch.float32, device="cuda"):
        # Params
        self.params = params
        self.dtype = dtype
        self.device = torch.device(device)

        # # Dataset spatial area (correlated to the specific dataset)
        self.min_lat, self.max_lat = 40.4, 40.95
        self.min_long, self.max_long = -74.5, -73.5

        # Common time between traffic and ev data
        self.start_time = None
        self.end_time = None

        # Other params
        self.traffic_resolution = None
        self.ev_resolution = None
        self.min_vals_normalization = None
        self.max_vals_normalization = None
        self.number_of_station = None
        self.features = None
        self.targets = None
        self.edge_index_traffic = None
        self.edge_weights_traffic = None
        self.edges_df = None
        self.nodes_df = None
        self.parsing_traffic_procedure = 'by_rows'
        self.added_edges_df = None
        self.coordinates_traffic = list()
        self.traffic_features, self.ev_features = None, None
        self.encoded_data = []

        # Traffic filepaths
        self.filepaths = sorted(glob.glob(os.path.join(self.params.traffic_temporal_data_folder, "*.csv")))
        if not self.filepaths:
            raise RuntimeError(f"Nessun XLS trovato in {self.params.traffic_temporal_data_folder}")

        # Process traffic metadata in order to incorporate distances
        if not self.params.use_traffic_metadata_processed:
            process_traffic_metadata_newyork(self.params,
                                             self.params.traffic_metadata_file)

        # Load temporal data
        if self.params.load_preprocessed_data and len(os.listdir(self.params.preprocessed_dataset_path)) > 0:
            # Load temporal data after temporal alignment and resolution resampling
            self.data_tensor_traffic, self.data_tensor_ev = self.load_preprocessed_data()

            # Visualize info
            self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
            N_e, T_e, M_e = self.data_tensor_ev.shape
            print(f'Loaded Traffic data by rows: {self.N_t} nodes, {self.T_t} timesteps, {self.M_t} features')
            print(f'Loaded EV data by rows: {N_e} nodes, {T_e} timesteps, {M_e} features')

        else:
            # Find temporal intersection window
            self.check_traffic_ev_time()

            # Load traffic temporal data
            self._load_traffic_data()

            # Load ev temporal data
            self._load_ev_data()

            # Since resolution is not precise, resample both traffic and ev data to the same timestep
            self.data_tensor_traffic, self.data_tensor_ev = resample_to_common_time(self.data_tensor_traffic,
                                                                                    self.data_tensor_ev,
                                                                                    target="A",  # resample to traffic resolution
                                                                                    method="linear")

            # Save processed results
            if self.params.save_preprocessed_data:
                # Save temporal data after temporal alignment and resolution resampling
                torch.save({"data_tensor_traffic": self.data_tensor_traffic, "data_tensor_ev": self.data_tensor_ev},
                           os.path.join(self.params.preprocessed_dataset_path, f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))

                # Visualize info
                self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
                N_e, T_e, M_e = self.data_tensor_ev.shape
                print(f'Saved Traffic data by rows: {self.N_t} nodes, {self.T_t} timesteps, {self.M_t} features')
                print(f'Saved EV data by rows: {N_e} nodes, {T_e} timesteps, {M_e} features')

        # Construct graph, with original nodes, original edges plus fake edges to allow whole graph communication
        self.get_edges_traffic(threshold=self.params.graph_distance_threshold)

        # Assign EV temporal features to the nearest traffic nodes and then traffic edges
        self.assign_ev_node_to_traffic_node()

        # Normalize and stack data into fixed-size input/output windows objects
        self.preprocess_and_assemble_data()


    def load_preprocessed_data(self):
        """
        Load preprocessed data after temporal alignment and resolution resampling
        """
        load_tensor = torch.load(os.path.join(self.params.preprocessed_dataset_path,f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))
        data_tensor_traffic = load_tensor["data_tensor_traffic"]
        data_tensor_ev = load_tensor["data_tensor_ev"]
        return data_tensor_traffic, data_tensor_ev

    def check_traffic_ev_time(self):
        """
        This function define the start and end time of the synchronous traffic and time series. To decide it, it gets
        the overlapping time between traffic and ev timeseries.
        """
        ## Parsing traffic time series
        dfs = []  # Collecting dataFrame list per site
        sites = []  # Collecting Site ID per site

        # 'by_time' procedure collect traffic temporal data between different sites by aligning them at same time.
        # However, with raw data alignment at same time can cause problems!
        if self.parsing_traffic_procedure == 'by_time':
            seen = set()
            print('Loading Traffic data by_time...')
            for path in self.filepaths[:self.params.num_of_traffic_nodes_limit]:
                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort values by time
                df = df.sort_values("data_as_of").reset_index(drop=True)

                # Set ts index - convert la colonna TIMESTAMP da stringa ISO 8601 a datetime
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce")

                # Arrotonda al minuto (toglie secondi e millisecondi)
                df["data_as_of"] = df["data_as_of"].dt.floor("min")

                # Check different time index between files
                new_vals = set(df["data_as_of"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("data_as_of")

                # Cast
                feature_cols = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

                dfs.append(df)
                sites.append(site_id)

            # Index outer join
            union_index = dfs[0].index
            for d in dfs[1:]:
                union_index = union_index.union(d.index)

        # 'by_rows'  procedure collect traffic temporal data between different sites by aligning them at same rows
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[check_traffic_ev_time] Loading Traffic data by_rows...')

            # Collect traffic temporal data
            dfs = []
            sites = []

            # 1) First pass: Read and clean each traffic CSV, but do NOT set the index to time
            cont = 0
            for path in self.filepaths:
                # Check number of traffic sites constraint (defined by user)
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} lacks columns {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df = df.sort_values("data_as_of").reset_index(drop=True)
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce").dt.floor("min")

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols_ = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                feature_cols = [c for c in feature_cols_ if c in self.params.traffic_columns_to_use]
                df = df[["data_as_of"] + feature_cols]  # maintain desired order, keeping time as a column (not index)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
                cont += 1
                print(f'[check_traffic_ev_time]: Traffic site #{cont}/{len(self.filepaths)}')

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and cut from the end
            min_len = min(len(d) for d in dfs)
            if min_len == 0:
                raise ValueError("After cleaning, at least one CSV has no useful rows.")

            dfs_trimmed = [d.tail(min_len).reset_index(drop=True) for d in dfs]

            # 3) Get min and max timestamp needed for EV rows selections
            self.start_time = dfs_trimmed[0]['data_as_of'][0]
            self.end_time = dfs_trimmed[0]['data_as_of'][len(dfs_trimmed[0]['data_as_of'])-1]

        ## Parsing ev time series
        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long))

        # Values of a specific column for the EXCLUDED rows
        excluded_vals = (ev_metadata.loc[~mask, "LocID"]).tolist()

        # Collect ev temporal data
        dfs = []  # DataFrame list per site
        sites = []  # site ID
        # ev_columns = ["timestamp","Available","Total","Offline"]
        ev_columns = DatasetNewyork._EV_DATA_COLUMNS  # TODO: choose the correct of features to use
        print('[check_traffic_ev_time] Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Check number of ev sites constraint (defined by user)
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break

            # Gather site id and do cut off distant nodes
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]
            if int(site_id) in excluded_vals:
                print(f'Skipping {site_id}')
                continue

            # Import EV data
            df = pd.read_csv(path, usecols=ev_columns)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            # Set timestamp to index
            df = df.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts
            df = df.set_index("timestamp")

            # Cast
            feature_cols_ = [c for c in ev_columns if c != "timestamp"]
            feature_cols = [c for c in feature_cols_ if c in self.params.ev_columns_to_use]
            df = df[feature_cols]  # mantiene ordine voluto
            df = df.apply(pd.to_numeric, errors="coerce")
            dfs.append(df)
            sites.append(site_id)

        # # Get low and high timestamp index
        union_index_ev = dfs[0].index
        for d in dfs[1:]:
            union_index_ev = union_index_ev.union(d.index)

        union_index_ev = union_index_ev.sort_values()
        low = union_index_ev.min()
        high = union_index_ev.max()

        # Finally, assign the highest timestamp values between traffic and ev starting time
        if low > self.start_time:
            self.start_time = low

        # Finally, assign the lowest timestamp values between traffic and ev finish time
        if high < self.end_time:
            self.end_time = high



    def _load_traffic_data(self,
                           pad_value=-1.0):
        """
        Load temporal traffic data collected for each edges
        """
        # Collect traffic data: dataFrame list per site and Site ID per site
        dfs = []  # Collecting dataFrame list per site
        sites = []  # Colelcting Site ID per site

        # Select procedure ['by_rows', 'by_time'] of aligning data on .csv files rows (starting from the end) or by time index (but unfiseable
        # since datetime differs among files of different amount of time)
        if self.parsing_traffic_procedure == 'by_time':
            seen = set()
            print('Loading Traffic data by_time...')
            for path in self.filepaths[:self.params.num_of_traffic_nodes_limit]:
                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort values by time
                df = df.sort_values("data_as_of").reset_index(drop=True)

                # Set ts index - convert la colonna TIMESTAMP da stringa ISO 8601 a datetime
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce")

                # Arrotonda al minuto (toglie secondi e millisecondi)
                df["data_as_of"] = df["data_as_of"].dt.floor("min")

                # Check different time index between files
                new_vals = set(df["data_as_of"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("data_as_of")

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
                feature_cols = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

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
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[_load_traffic_data] Loading Traffic data by_rows...')
            dfs = []
            sites = []
            resolutions = []

            # 1) First pass: Read and clean each CSV, but do NOT set the index to time.
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df = df.sort_values("data_as_of").reset_index(drop=True)
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce").dt.floor("min")

                # Get resolution
                diffs = df['data_as_of'].diff().dropna().mean()
                resolutions.append(diffs.total_seconds() / 60)

                # (Optional) Handle time duplicates BEFORE cutting: average per minute
                if df["data_as_of"].duplicated().any():
                    # Calculates the average across rows with the same timestamp, then reorders by time
                    df = (df.groupby("data_as_of", as_index=False)
                          .mean(numeric_only=True)
                          .sort_values("data_as_of")
                          .reset_index(drop=True))
                    print(f'Find duplicate timestamps in {os.path.basename(path)} (collapsed by mean)')


                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols_ = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                feature_cols = [c for c in feature_cols_ if c in self.params.traffic_columns_to_use]
                df = df[["data_as_of"] + feature_cols]  # maintain desired order, keeping time as a column (not index)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
            print(f'[_load_traffic_data] Loaded {len(sites)} traffic sites')

            # Get mean resolution in minutes
            self.traffic_resolution = float(np.asarray(resolutions).mean())

            # 1) Time window on each df (dfs is the list of DataFrames)
            dfs_windowed = []
            for d in dfs:
                mask = d["data_as_of"].between(self.start_time, self.end_time, inclusive="both")
                d_cut = d.loc[mask].reset_index(drop=True)
                dfs_windowed.append(d_cut)

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and trim from the end
            min_len = min(len(d) for d in dfs_windowed)
            if min_len == 0:
                raise ValueError("No rows fall within the time window for at least one CSV.")
            dfs_trimmed_ = [d.tail(min_len).reset_index(drop=True) for d in dfs_windowed]

            # Order dfs in order to respect ID increasing order
            sites = [int(s) for s in sites]
            order = sorted(range(len(sites)), key=lambda i: sites[i])
            self.sites_sorted = [sites[i] for i in order]
            dfs_trimmed = [dfs_trimmed_[i] for i in order]

            # 3) Align by POSITION (same rows for all), concatenating by columns
            #    Build a single multi-index DataFrame on the columns: (site, feature)
            #    The index remains a RangeIndex 0..min_len-1 (position)
            dfs_features_only = [d.drop(columns=["data_as_of"]) for d in dfs_trimmed]  # if you no longer want timestamp
            df_all = pd.concat(dfs_features_only, axis=1, keys=sites, names=["site", "feature"])
            self.df_all = df_all  # already aligned per line; no union/padding
            self.timestamp_final_traffic = pd.RangeIndex(start=0, stop=min_len, name="row")  # indice posizionale
            self.traffic_columns_used_in_data = dfs_features_only[0].columns
            self.time_column = dfs_trimmed[0]["data_as_of"]

            # 4) Build the tensor: (N nodes, T timesteps (=min_len), M features)
            data = np.stack([d.values.astype(np.float32) for d in dfs_features_only], axis=0)
            self.data_tensor_traffic = torch.tensor(data, dtype=torch.float32, device=self.device)
            self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
            print(f'Loaded Traffic data by rows: {self.N_t} edges, {self.T_t} rows (tail), {self.M_t} features')

    def get_edges_traffic(self,
                          threshold=0.001,
                          distances_col='__distances'):
        """
        Function for creating the graph. Staring by traffic sites, we first build the original graph.
        Then, if the graph is not connected, we add fake edges to guarantee the connectivity of the graph.
        TO each fake edges is correlated zeros features.
        """
        # Load traffic metadata
        data = pd.read_csv(self.params.traffic_metadata_file, sep=';')

        # Deserialize i dati JSON, se necessario
        data['__points'] = data['__points'].apply(json.loads)
        data['__distances'] = data['__distances'].apply(json.loads)
        data['__distances'] = data['__distances'].apply(sum)
        data_sorted = data.sort_values(by="id")

        # Replace the 'id' column with increasing values ​​from 0 to len(df)-1
        self.edge_id_mapping = dict(zip(data_sorted['id'], range(len(data_sorted))))  # old: new
        data_sorted['id'] = range(len(data_sorted))

        # Original edges
        edges_df, self.nodes_df = build_edges_with_node_ids(data_sorted,
                                                            threshold=threshold,
                                                            distances_col=distances_col)

        # Creates a set of unique arcs (normalizing orientation)
        # Newyork dataset: src_id 73 and 75 originals are the same
        unique_edges = set()
        duplicates = []
        for _, row in edges_df.iterrows():
            u_id = int(row['src_id'])
            v_id = int(row['tgt_id'])

            # Normalize the orientation of the arcs
            if u_id > v_id:
                u_id, v_id = v_id, u_id
            edge = frozenset((u_id, v_id))
            if edge in unique_edges:
                duplicates.append(row)
            else:
                unique_edges.add(edge)
        id_duplicates = [elem['id'] for elem in duplicates]

        # I delete duplicate arcs and update the index column
        edges_df = edges_df[~edges_df['id'].isin(id_duplicates)]
        edges_df['id'] = np.arange(len(edges_df))

        # Update temporal tensor to integrate duplicate arcs deletion
        rows_to_keep = [i for i in range(self.data_tensor_traffic.shape[0]) if i not in id_duplicates]
        self.data_tensor_traffic = self.data_tensor_traffic[rows_to_keep]

        # Create original adjacency matrix (if needed)
        orig_adj_matrix, _ = create_adjacency_matrix_newyork(edges_df['src_id'],
                                                          edges_df['tgt_id'],
                                                          num_nodes=len(self.nodes_df),
                                                          distance=edges_df['distance'])

        # Modify adjacency matrix by adding fake edges between nodes.
        # You should add edges for make the graph connected first, by checking on node distance then for example
        # Mimimal number of edges to have a graph connected is num_nodes -1
        self.edges_df, self.added_edges_df = augment_graph_df_v3(edges_df=edges_df,
                                                                 nodes_df=self.nodes_df)  # TODO: add directed/edges diciture

        adj_matrix, double_nodes = create_adjacency_matrix_newyork(self.edges_df['src_id'],
                                                     self.edges_df['tgt_id'],
                                                     num_nodes=len(self.nodes_df),
                                                     distance=self.edges_df['distance'])

        # Create graph based on distance threshold (in Km)
        edge_index, edge_weights = dense_to_sparse(torch.tensor(adj_matrix))
        edge_index, edge_weights = directed_to_undirected(edge_index, edge_weights)
        self.edge_index_traffic, self.edge_weights_traffic = edge_index.to('cuda'), edge_weights.to('cuda')

        # Update temporal data with fake data (zeros features)
        self.data_tensor_traffic = append_along_N_torch(self.data_tensor_traffic,
                                                        len(self.added_edges_df),
                                                        fill='mean')

        print('[get_edges_traffic] Connected graph created!')

    def _load_ev_data(self,
                      pad_value=-1.0):
        """
        Function for loading EV temporal data and implementing some preprocessing activity:
            - Cut distant EV nodes
            - Check and substitute duplicate values
            - Check columns
            - Cut temporal data before and after start and end time values constraints
        """

        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (
                ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long)
        )

        filtered_ev_df = ev_metadata.loc[mask].copy()

        # values of a specific column for the EXCLUDED rows
        col = "LocID"
        excluded_vals = (ev_metadata.loc[~mask, col]).tolist()

        dfs = []  # DataFrame list per site
        sites = []  # site ID
        resolutions = []
        ev_columns = ["timestamp","Available","Total","Offline"]
        print('Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]
            print(site_id)
            if int(site_id) in excluded_vals:
                print(f'Skipping {site_id}')
                continue

            df = pd.read_csv(path, usecols=ev_columns)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            df = df.sort_values("timestamp").reset_index(drop=True)

            # Set ts index
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts

            # Get resolution
            diffs = df['timestamp'].diff().dropna().mean()
            resolutions.append(diffs.total_seconds() / 60)

            df = df.set_index("timestamp")

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
            feature_cols_ = [c for c in ev_columns if c != "timestamp"]
            feature_cols = [c for c in feature_cols_ if c in self.params.ev_columns_to_use]
            df = df[feature_cols]  # mantiene ordine voluto
            df = df.apply(pd.to_numeric, errors="coerce")

            dfs.append(df)
            sites.append(site_id)

        print(f'[_load_ev_data] Loaded {len(sites)} EV sites')
        # Get mean resolution in minutes
        self.ev_resolution = float(np.asarray(resolutions).mean())

        # Index outer join
        union_index = dfs[0].index
        for d in dfs[1:]:
            union_index = union_index.union(d.index)
        # Here apply cut to align EV data to traffic data given start and end time variables
        union_index = union_index.sort_values()
        union_index = union_index[(union_index >= self.start_time) & (union_index <= self.end_time)]

        # Realign and constant padding
        dfs_aligned_ = [d.reindex(union_index).fillna(pad_value) for d in dfs]

        #Order dfs in order to respect ID increasing order
        order = sorted(range(len(sites)), key=lambda i: sites[i])
        sites_sorted = [sites[i] for i in order]
        dfs_aligned = [dfs_aligned_[i] for i in order]

        # Concat
        df_all = pd.concat(dfs_aligned, axis=1, keys=sites_sorted, names=["site", "feature"])
        self.df_ev_all = df_all.sort_index()  # (T_max, N*M)

        # Build EV tensor
        self.ev_columns_used_in_data = dfs_aligned[0].columns
        data = np.stack([d.values.astype(np.float32) for d in dfs_aligned], axis=0)
        self.data_tensor_ev = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.N_ev, self.T_ev, self.M_ev = self.data_tensor_ev.shape
        print(f'Loaded EV data: {self.N_ev} nodes, {self.T_ev} timesteps, {self.M_ev} features')

    def assign_ev_node_to_traffic_node(self):
        # Load EV metadata
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)

        # Get EV coordinates
        ev_coordinates = list()
        for row in ev_metadata.iterrows():
            lat, lng = row[1]['Latitude'], row[1]['Longitude']
            ev_coordinates.append((lat, lng))

        # Map each EV node to nearest traffic node
        map_ev_node_traffic_node = {}
        print('Assigning EV to traffic nodes!')
        cont_ev = 0
        for ev_node_idx, ev_coord in enumerate(ev_coordinates):
            if cont_ev == self.params.num_of_ev_nodes_limit:
                break
            print(ev_node_idx)
            min_dist = float('inf')
            min_dist_traffic_node_idx = -1
            lat1, lon1 = ev_coord
            cont_traffic = 0
            for row in self.nodes_df.iterrows():
                if cont_traffic == self.params.num_of_traffic_nodes_limit:
                    break
                lat2, lon2, traffic_node_idx = row[1]['lat'], row[1]['lon'], row[1]['node_id']
                distance = haversine(lat1, lon1, lat2, lon2)
                if distance < min_dist:
                    min_dist = distance
                    min_dist_traffic_node_idx = traffic_node_idx
                cont_traffic += 1
            map_ev_node_traffic_node[ev_node_idx] = int(min_dist_traffic_node_idx)
            cont_ev += 1

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to map_ev_node_traffic_node: create a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(len(self.nodes_df))]
        for key in map_ev_node_traffic_node.keys():
            corrispective_traffic_node = map_ev_node_traffic_node[key]
            ev_values = self.data_tensor_ev[key]
            temp_list[corrispective_traffic_node].append(ev_values)

        lista_max = max(temp_list, key=len)
        tensor_ev_temp = torch.stack(lista_max)
        ev_timesteps = tensor_ev_temp.shape[1]

        new_temp_list = list()
        for elem in temp_list:
            if len(elem) == 0:
                new_temp_list.append(torch.zeros(ev_timesteps,len(self.params.ev_columns_to_use)).to(self.params.device))
            elif len(elem) == 1:
                new_temp_list.append(elem[0])
            else:
                new_temp_list.append(torch.stack(elem).sum(0).squeeze(0))

        # inner join con self.timestamp_final_traffic e check sincronicità
        self.ev_temporal_data_on_merged_nodes = torch.stack(new_temp_list)
        print('Merged EV!')

        assert self.ev_temporal_data_on_merged_nodes.shape[1] == self.data_tensor_traffic.shape[1]
        self.ev_edge_temporal_data = torch.zeros(self.data_tensor_traffic.shape[0],
                                                 self.data_tensor_traffic.shape[1],
                                                 self.ev_temporal_data_on_merged_nodes.shape[2], device='cpu')
        self.ev_temporal_data_on_merged_nodes.cpu()
        copy_edges_df = self.edges_df.copy()
        for i in range(self.ev_temporal_data_on_merged_nodes.shape[0]):
            print(i)
            ev_node_level_info = self.ev_temporal_data_on_merged_nodes[i,:,:]
            ev_node_level_info.cpu()
            # Filtra le righe che hanno src_id o tgt_id uguale a i
            filtered_edges_ids = copy_edges_df[(copy_edges_df['src_id'] == i) | (copy_edges_df['tgt_id'] == i)]['id'].tolist()
            # Usa gli indici di filtered_edges_ids per assegnare i valori direttamente
            self.ev_edge_temporal_data[filtered_edges_ids, :, :] = ev_node_level_info.cpu()/2

        # Project ev info on node_t to edge_t
        self.final_temporal_merged_data = torch.cat([self.data_tensor_traffic.cpu(), self.ev_edge_temporal_data.cpu()], dim=-1)
        self.traffic_features, self.ev_features = self.data_tensor_traffic.shape[-1], self.ev_edge_temporal_data.shape[-1]

        # edge_to_node_aggregation
        self.final_temporal_merged_data = edge_to_node_aggregation(self.edge_index_traffic, self.final_temporal_merged_data, len(self.nodes_df))
        print('Done!')


    def preprocess_and_assemble_data(self):
        # Prepare final data
        stacked_target = self.final_temporal_merged_data.to('cpu')
        self.number_of_station = self.final_temporal_merged_data.shape[0]

        # Calcola il Min e Max separato per ogni canale lungo le dimensioni (N, T)
        self.min_vals_normalization = stacked_target.min(dim=0)[0].min(dim=0)[0]  # Min lungo (N, T) per ogni canale
        self.max_vals_normalization = stacked_target.max(dim=0)[0].max(dim=0)[0]  # Max lungo (N, T) per ogni canale

        # Normalizza usando MinMax scaling
        standardized_target = ((stacked_target - self.min_vals_normalization) /
                               (self.max_vals_normalization - self.min_vals_normalization))

        # Input data
        self.features = [standardized_target[:,i: i + self.params.lags, :]
                         for i in range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                                        self.params.time_series_step)]

        # Output data
        N = standardized_target.shape[0]
        self.targets = [standardized_target[:, i:i + self.params.prediction_window, :].view(N, -1)
                        for i in range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                       self.params.time_series_step)]

        # Input time data
        self.time_input = [self.time_column[i: i + self.params.lags]
                         for i in range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                                        self.params.time_series_step)]

        # Output time data
        self.time_output = [self.time_column[i:i + self.params.prediction_window]
                        for i in range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                       self.params.time_series_step)]



        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=self.edge_index_traffic.long(),
                                          edge_attr=self.edge_weights_traffic.float(),
                                          y=torch.FloatTensor(self.targets[i]),
                                          time_input=self.time_input[i],
                                          time_output=self.time_output[i]),)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


class DatasetChicago(Dataset):
    _TRAFFIC_DATA_COLUMNS = ['speed', 'length', 'time']  # ["speed","travel_time","status","data_as_of","link_id"]
    _TRAFFIC_METADATA_COLUMNS = ['id', 'street', 'length', 'start_latitude', 'start_longitude', 'end_latitude',
                                 'end_longitude', 'max_speed']  # ["id","link_points","points","distances"]
    _EV_DATA_COLUMNS = ["location_id", "timestamp", "Available", "Total", "Offline"]
    _EV_METADATA_COLUMNS = ["LocID", "LocName", "Latitude", "Longitude"]

    def __init__(self, params, columns=None, dtype=torch.float32, device="cuda"):
        # Params
        self.params = params
        self.dtype = dtype
        self.device = torch.device(device)

        # Dataset spatial area (correlated to the specific dataset)
        self.min_lat, self.max_lat = 41.5, 42.1  # 41.6589702, 42.0128310
        self.min_long, self.max_long = -87.9, -87.4  # -87.8368335, -87.5350520

        # Common time between traffic and ev data
        self.start_time = None
        self.end_time = None

        # Other params
        self.traffic_resolution = None
        self.ev_resolution = None
        self.min_vals_normalization = None
        self.max_vals_normalization = None
        self.number_of_station = None
        self.features = None
        self.targets = None
        self.edge_index_traffic = None
        self.edge_weights_traffic = None
        self.edges_df = None
        self.nodes_df = None
        self.parsing_traffic_procedure = 'by_rows'
        self.added_edges_df = None
        self.coordinates_traffic = list()
        self.traffic_features, self.ev_features = None, None
        self.encoded_data = []

        # Get all traffic filepaths
        self.filepaths = sorted(glob.glob(os.path.join(self.params.traffic_temporal_data_folder, "*.csv")))
        if not self.filepaths:
            raise RuntimeError(f"No CSV founded in {self.params.traffic_temporal_data_folder}")

        # Check if preprocessed traffic data already exists. Be careful to use same creation parameters
        processed_traffic_temporal_data_exist = os.path.exists(os.path.join(self.params.preprocessed_dataset_path,
                                                                            f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))

        # If you don't want to use preprocessed data or preprocessed data doesn't exist
        if not (self.params.use_traffic_temporal_data_processed and processed_traffic_temporal_data_exist):

            # # Process metadata
            # process_traffic_metadata_chicago(self.params, self.params.traffic_metadata_file)

            # Find temporal intersection window
            self.check_traffic_ev_time()

            # Load traffic temporal data
            self._load_traffic_data()

            # Load ev temporal data
            self._load_ev_data()

            # Since resolution is not precise, resample both traffic and ev data to obtain the same timesteps
            self.data_tensor_traffic, self.data_tensor_ev = resample_to_common_time(self.data_tensor_traffic,
                                                                                    self.data_tensor_ev,
                                                                                    target="A",
                                                                                    method="linear")

            # Save processed results
            if self.params.save_preprocessed_data:
                # Save preprocessed data after temporal alignment and resolution resampling
                torch.save({"data_tensor_traffic": self.data_tensor_traffic, "data_tensor_ev": self.data_tensor_ev},
                           os.path.join(self.params.preprocessed_dataset_path,
                                        f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))

                # Visualize info
                self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
                N_e, T_e, M_e = self.data_tensor_ev.shape
                print(f'Saved Traffic data by rows: {self.N_t} nodes, {self.T_t} timesteps, {self.M_t} features')
                print(f'Saved EV data by rows: {N_e} nodes, {T_e} timesteps, {M_e} features')

        # If you want to use preprocessed data and preprocessed data doesn't exist
        elif self.params.use_traffic_temporal_data_processed and processed_traffic_temporal_data_exist:

            # Load temporal data after temporal alignment and resolution resampling
            self.data_tensor_traffic, self.data_tensor_ev = self.load_preprocessed_data()

            # Visualize info
            self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
            N_e, T_e, M_e = self.data_tensor_ev.shape
            print(f'Loaded Traffic data by rows: {self.N_t} nodes, {self.T_t} timesteps, {self.M_t} features')
            print(f'Loaded EV data by rows: {N_e} nodes, {T_e} timesteps, {M_e} features')

        else:
            raise Exception('Error in loading or saving data!')

        # Construct graph, with original nodes, original edges and add fake edges (if needed) to allow whole
        # graph communication.
        self.get_edges_traffic(threshold=self.params.graph_distance_threshold)

        # Assign EV temporal features to the nearest traffic nodes and then traffic edges
        self.assign_ev_node_to_traffic_node()

        # Normalize and stack data into fixed-size input/output windows objects
        self.preprocess_and_assemble_data()

    def load_preprocessed_data(self):
        """
        Load preprocessed data after temporal alignment and resolution resampling
        """
        load_tensor = torch.load(os.path.join(self.params.preprocessed_dataset_path,
                                              f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))
        data_tensor_traffic = load_tensor["data_tensor_traffic"]
        data_tensor_ev = load_tensor["data_tensor_ev"]
        return data_tensor_traffic, data_tensor_ev

    def check_traffic_ev_time(self):
        """
        This function define the start and end time of the synchronous traffic and time series. To decide it, it gets
        the overlapping time between traffic and ev timeseries.
        """
        # Parsing traffic time series
        dfs = []  # Collecting dataFrame list per site
        sites = []  # Collecting Site ID per site

        # 'by_time' procedure collect traffic temporal data between different sites by aligning them at same time.
        # However, with raw data alignment at same time can cause problems!
        if self.parsing_traffic_procedure == 'by_time':
            print('Loading Traffic data by_time...')
            seen = set()
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break
                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Ordina per tempo e normalizza al minuto (così 'fine' = righe più recenti)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing con AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Check different time index between files
                new_vals = set(df["time"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("time")

                # Cast
                feature_cols = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]
                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

                dfs.append(df)
                sites.append(site_id)

            # Index outer join
            union_index = dfs[0].index
            for d in dfs[1:]:
                union_index = union_index.union(d.index)

        # 'by_rows'  procedure collect traffic temporal data between different sites by aligning them at same rows
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[check_traffic_ev_time] Loading Traffic data by_rows...')

            # Collect traffic temporal data
            dfs = []
            sites = []

            # 1) First pass: Read and clean each traffic CSV, but do NOT set the index to time.
            cont = 0
            for path in self.filepaths:
                # Check number of traffic sites constraint (defined by user)
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} lacks columns {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing with AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]
                df = df[
                    ["time"] + feature_cols]  # mantieni ordine voluto, tenendo anche il tempo come colonna (non indice)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
                cont += 1
                print(f'[check_traffic_ev_time]: Traffic site #{cont}/{len(self.filepaths)}')

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and cut from the end
            min_len = min(len(d) for d in dfs)
            if min_len == 0:
                raise ValueError("After cleaning, at least one CSV has no useful rows.")

            dfs_trimmed = [d.tail(min_len).reset_index(drop=True) for d in dfs]

            # 3) Get min and max timestamp needed for EV rows selections
            self.start_time = dfs_trimmed[0]['time'][0]
            self.end_time = dfs_trimmed[0]['time'][len(dfs_trimmed[0]['time']) - 1]

        # Parsing ev time series
        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long))

        # Values of a specific column for the EXCLUDED rows
        excluded_vals = (ev_metadata.loc[~mask, "LocID"]).tolist()

        # Collect ev temporal data
        dfs = []  # DataFrame list per site
        sites = []  # site ID

        ev_columns = DatasetChicago._EV_DATA_COLUMNS
        # ev_columns = ["timestamp","Available","Total","Offline"]
        print('[check_traffic_ev_time] Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Check number of ev sites constraint (defined by user)
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break

            # Gather site id and do cut off distant nodes
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]
            if int(site_id) in excluded_vals:
                print(f'[check_traffic_ev_time] {site_id} EV site out of spatial range!')
                continue

            # Import EV data
            df = pd.read_csv(path, usecols=ev_columns)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            # Set timestamp to index
            df = df.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts
            df = df.set_index("timestamp")

            # Cast
            feature_cols = [c for c in ev_columns if c != "timestamp"]
            df = df[feature_cols]
            df = df.apply(pd.to_numeric, errors="coerce")
            dfs.append(df)
            sites.append(site_id)

        # Get low and high timestamp index
        union_index_ev = dfs[0].index
        for d in dfs[1:]:
            union_index_ev = union_index_ev.union(d.index)

        union_index_ev = union_index_ev.sort_values()
        low = union_index_ev.min()
        high = union_index_ev.max()

        # Finally, assign the highest timestamp values between traffic and ev starting time
        if low > self.start_time:
            self.start_time = low

        # Finally, assign the lowest timestamp values between traffic and ev finish time
        if high < self.end_time:
            self.end_time = high

    def _load_traffic_data(self,
                           pad_value=-1.0):
        """
        Load temporal traffic data collected for each edges
        """
        # Collect traffic data: dataFrame list per site and Site ID per site
        dfs = []
        sites = []

        # Select procedure ['by_rows', 'by_time'] of aligning data on .csv files rows (starting from the end) or by
        # time index (but unfeaseable since datetime differs among files of different amount of time)

        if self.parsing_traffic_procedure == 'by_time':
            seen = set()
            print('Loading Traffic data by_time...')
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # # Sort values by time
                # df = df.sort_values("time").reset_index(drop=True)

                # Ordina per tempo e normalizza al minuto (così 'fine' = righe più recenti)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing con AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Check different time index between files
                new_vals = set(df["time"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("time")

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
                feature_cols = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]  # old: data_as_of
                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

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
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[_load_traffic_data] Loading Traffic data by_rows...')
            dfs = []
            sites = []
            resolutions = []

            # 1) First pass: Read and clean each CSV, but do NOT set the index to time.
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing con AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Get resolution
                diffs = df['time'].diff().dropna().mean()
                resolutions.append(diffs.total_seconds() / 60)

                # (Optional) Handle time duplicates BEFORE cutting: average per minute
                if df["time"].duplicated().any():
                    print(f'Find duplicate timestamps in {os.path.basename(path)} (collapsed by mean)')
                    # calculates the average across rows with the same timestamp, then reorders by time
                    df = (df.groupby("time", as_index=False)
                          .mean(numeric_only=True)
                          .sort_values("time")
                          .reset_index(drop=True))

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]
                df = df[
                    ["time"] + feature_cols]  # maintain desired order, also keeping the time as a column (not index)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
            print(f'[_load_traffic_data] Loaded {len(sites)} traffic sites')

            # Get mean resolution in minutes
            self.traffic_resolution = float(np.asarray(resolutions).mean())

            # 1) Time window on each df (dfs is the list of DataFrames)
            dfs_windowed = []
            for d in dfs:
                mask = d["time"].between(self.start_time, self.end_time, inclusive="both")
                d_cut = d.loc[mask].reset_index(drop=True)
                dfs_windowed.append(d_cut)

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and trim from the end
            min_len = min(len(d) for d in dfs_windowed)
            if min_len == 0:
                raise ValueError("No rows fall within the time window for at least one CSV.")
            dfs_trimmed_ = [d.tail(min_len).reset_index(drop=True) for d in dfs_windowed]

            # Order dfs in order to respect ID increasing order
            sites = [int(s) for s in sites]
            order = sorted(range(len(sites)), key=lambda i: sites[i])
            self.sites_sorted = [sites[i] for i in order]
            dfs_trimmed = [dfs_trimmed_[i] for i in order]

            # 3) Align by POSITION (same rows for all), concatenating by columns
            #    Build a single multi-index DataFrame on the columns: (site, feature)
            #    The index remains a RangeIndex 0..min_len-1 (position).
            dfs_features_only = [d.drop(columns=["time"]) for d in dfs_trimmed]  # se non vuoi più il timestamp
            df_all = pd.concat(dfs_features_only, axis=1, keys=sites, names=["site", "feature"])
            self.df_all = df_all  # already aligned per line; no union/padding
            self.timestamp_final_traffic = pd.RangeIndex(start=0, stop=min_len, name="row")  # positional index
            self.traffic_columns_used_in_data = dfs_features_only[0].columns
            self.time_column = dfs_trimmed[0]["time"]

            # 4) Build the tensor: (N nodes, T timesteps (=min_len), M features)
            data = np.stack([d.values.astype(np.float32) for d in dfs_features_only], axis=0)
            self.data_tensor_traffic = torch.tensor(data, dtype=torch.float32, device=self.device)
            self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
            print(
                f'Loaded Traffic data from csv files by rows: {self.N_t} edges, {self.T_t} rows (tail), {self.M_t} features')

    def get_edges_traffic(self,
                          threshold: float = 0.001,
                          distances_col='length'):
        """
        Function for creating the graph. Staring by traffic sites, we first build the original graph.
        Then, if the graph is not connected, we add fake edges to guarantee the connectivity of the graph.
        TO each fake edges is correlated zeros features.
        """
        # Load traffic metadata
        data = pd.read_csv(self.params.traffic_metadata_file)
        data_sorted = data.sort_values(by="id")

        # Original edges
        edges_df, self.nodes_df = build_edges_with_node_ids_chicago(data_sorted,
                                                                    threshold=threshold,
                                                                    distances_col=distances_col)  # exact match

        # Creates a set of unique arcs (normalizing orientation)
        unique_edges = set()
        duplicates = []
        for _, row in edges_df.iterrows():
            u_id = int(row['src_id'])
            v_id = int(row['tgt_id'])

            # Normalize the orientation of the arcs
            if u_id > v_id:
                u_id, v_id = v_id, u_id
            edge = frozenset((u_id, v_id))
            if edge in unique_edges:
                duplicates.append(row)
            else:
                unique_edges.add(edge)
        id_duplicates = [elem['id'] for elem in duplicates]

        # I delete duplicate arcs and update the index column
        edges_df = edges_df[~edges_df['id'].isin(id_duplicates)]
        edges_df['id'] = np.arange(len(edges_df))

        # Update temporal tensor to integrate duplicate arcs deletion
        rows_to_keep = [i for i in range(self.data_tensor_traffic.shape[0]) if i not in id_duplicates]
        self.data_tensor_traffic = self.data_tensor_traffic[rows_to_keep]

        # Create original adjacency matrix (if needed)
        # orig_adj_matrix, _ = create_adjacency_matrix_newyork(edges_df['src_id'],
        #                                                   edges_df['tgt_id'],
        #                                                   num_nodes=len(self.nodes_df),
        #                                                   distance=edges_df['distance'])

        # Modify adjacency matrix by adding fake edges between nodes.
        # You should add edges for make the graph connected
        self.edges_df, self.added_edges_df = augment_graph_df_v3(edges_df=edges_df,
                                                                 nodes_df=self.nodes_df)  # TODO: add directed/edges diciture

        adj_matrix, double_nodes = create_adjacency_matrix_newyork(self.edges_df['src_id'],
                                                                   self.edges_df['tgt_id'],
                                                                   num_nodes=len(self.nodes_df),
                                                                   distance=self.edges_df['distance'])

        # Create graph based on distance threshold (in Km)
        edge_index, edge_weights = dense_to_sparse(torch.tensor(adj_matrix))
        edge_index, edge_weights = directed_to_undirected(edge_index, edge_weights)
        self.edge_index_traffic, self.edge_weights_traffic = edge_index.to('cuda'), edge_weights.to('cuda')

        # Update temporal data with fake data (zeros features)
        self.data_tensor_traffic = append_along_N_torch(self.data_tensor_traffic,
                                                        len(self.added_edges_df),
                                                        fill='mean')

        print('[get_edges_traffic] Connected graph created!')

    def _load_ev_data(self,
                      pad_value=-1.0,
                      col="LocID"):
        """
        Function for loading EV temporal data and implementing some preprocessing activity:
            - Cut distant EV nodes
            - Check and substitute duplicate values
            - Check columns
            - Cut temporal data before and after start and end time values constraints
        """

        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (
                ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long)
        )

        # filtered_ev_df = ev_metadata.loc[mask].copy()
        excluded_vals = (ev_metadata.loc[~mask, col]).tolist()  # Values of a specific column for the EXCLUDED rows

        # Collect data
        dfs = []  # DataFrame list per site
        sites = []  # site ID
        resolutions = []
        ev_columns = ["timestamp", "Available", "Total", "Offline"]  # without location ID
        print('Loading EV data ...')

        # Gathering EV files
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Constraints on EV sites number
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]

            # Cut off too distant EV sites
            if int(site_id) in excluded_vals:
                print(f'Skipping {site_id}')
                continue

            # Load EV data
            df = pd.read_csv(path, usecols=DatasetChicago._EV_DATA_COLUMNS)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            # Set ts index
            df = df.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts

            # Get resolution
            diffs = df['timestamp'].diff().dropna().mean()
            resolutions.append(diffs.total_seconds() / 60)

            df = df.set_index("timestamp")

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
            feature_cols = [c for c in ev_columns if c != "timestamp"]
            df = df[feature_cols]  # maintains the desired order
            df = df.apply(pd.to_numeric, errors="coerce")
            dfs.append(df)
            sites.append(site_id)

        print(f'[_load_ev_data] Loaded {len(sites)} EV sites')

        # Get mean resolution in minutes
        self.ev_resolution = float(np.asarray(resolutions).mean())

        # Index outer join
        # Here apply cut to align EV data to traffic data given start and end time variables
        union_index = dfs[0].index
        for d in dfs[1:]:
            union_index = union_index.union(d.index)
        union_index = union_index.sort_values()
        union_index = union_index[(union_index >= self.start_time) & (union_index <= self.end_time)]
        dfs_aligned_ = [d.reindex(union_index).fillna(pad_value) for d in dfs]  # Realign and constant padding

        # Order dfs in order to respect ID increasing order
        order = sorted(range(len(sites)), key=lambda i: sites[i])
        sites_sorted = [sites[i] for i in order]
        dfs_aligned = [dfs_aligned_[i] for i in order]

        # Concat
        df_all = pd.concat(dfs_aligned, axis=1, keys=sites_sorted, names=["site", "feature"])
        self.df_ev_all = df_all.sort_index()  # (T_max, N*M)

        # Build EV tensor
        self.ev_columns_used_in_data = dfs_aligned[0].columns
        data = np.stack([d.values.astype(np.float32) for d in dfs_aligned], axis=0)
        self.data_tensor_ev = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.N_ev, self.T_ev, self.M_ev = self.data_tensor_ev.shape
        print(f'Loaded EV data: {self.N_ev} nodes, {self.T_ev} timesteps, {self.M_ev} features')

    def assign_ev_node_to_traffic_node(self):
        """
        We project EV node features into traffic nodes for aggregating information
        """
        # Load EV metadata
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)

        # Get EV coordinates
        ev_coordinates = list()
        for row in ev_metadata.iterrows():
            lat, lng = row[1]['Latitude'], row[1]['Longitude']
            ev_coordinates.append((lat, lng))

        # Map each EV node to nearest traffic node
        map_ev_node_traffic_node = {}
        print('Assigning EV to traffic nodes!')
        cont_ev = 0
        for ev_node_idx, ev_coord in enumerate(ev_coordinates):
            if cont_ev == self.params.num_of_ev_nodes_limit:
                break
            min_dist = float('inf')
            min_dist_traffic_node_idx = -1
            lat1, lon1 = ev_coord
            cont_traffic = 0
            for row in self.nodes_df.iterrows():
                if cont_traffic == self.params.num_of_traffic_nodes_limit:
                    break
                lat2, lon2, traffic_node_idx = row[1]['lat'], row[1]['lon'], row[1]['node_id']
                distance = haversine(lat1, lon1, lat2, lon2)
                if distance < min_dist:
                    min_dist = distance
                    min_dist_traffic_node_idx = traffic_node_idx
                cont_traffic += 1
            map_ev_node_traffic_node[ev_node_idx] = int(min_dist_traffic_node_idx)
            cont_ev += 1

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to map_ev_node_traffic_node. So we're creating a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(len(self.nodes_df))]
        for key in map_ev_node_traffic_node.keys():
            corrispective_traffic_node = map_ev_node_traffic_node[key]
            ev_values = self.data_tensor_ev[key]
            temp_list[corrispective_traffic_node].append(ev_values)

        lista_max = max(temp_list, key=len)
        tensor_ev_temp = torch.stack(lista_max)
        ev_timesteps = tensor_ev_temp.shape[1]

        new_temp_list = list()
        for elem in temp_list:
            if len(elem) == 0:
                new_temp_list.append(torch.zeros(ev_timesteps, 3).to(self.params.device))
            elif len(elem) == 1:
                new_temp_list.append(elem[0])
            else:
                new_temp_list.append(torch.stack(elem).sum(0).squeeze(0))

        # Inner join con self.timestamp_final_traffic e check sincronicità
        self.ev_temporal_data_on_merged_nodes = torch.stack(new_temp_list)
        print('Merged EV temporal data into traffic temporal data!')

        # Check temporal consistency
        assert self.ev_temporal_data_on_merged_nodes.shape[1] == self.data_tensor_traffic.shape[1]

        # Now we have to:
        #  1) Firts, for consistency we aggregate EV node temporal data to traffic edges
        #  2) Then, once we have all edge temporal data,
        self.ev_edge_temporal_data = torch.zeros(self.data_tensor_traffic.shape[0],
                                                 self.data_tensor_traffic.shape[1],
                                                 self.ev_temporal_data_on_merged_nodes.shape[2], device='cpu')
        self.ev_temporal_data_on_merged_nodes.cpu()
        copy_edges_df = self.edges_df.copy()
        for i in range(self.ev_temporal_data_on_merged_nodes.shape[0]):
            print(i)
            ev_node_level_info = self.ev_temporal_data_on_merged_nodes[i, :, :]
            ev_node_level_info.cpu()

            # Filter rows that have src_id or tgt_id equal to i
            filtered_edges_ids = copy_edges_df[(copy_edges_df['src_id'] == i) | (copy_edges_df['tgt_id'] == i)][
                'id'].tolist()

            # Use the filtered_edges_ids indexes to assign values directly
            self.ev_edge_temporal_data[filtered_edges_ids, :, :] = ev_node_level_info.cpu() / 2

        # Project ev info on node_t to edge_t
        self.final_temporal_merged_data = torch.cat([self.data_tensor_traffic.cpu(), self.ev_edge_temporal_data.cpu()],
                                                    dim=-1)
        self.traffic_features, self.ev_features = self.data_tensor_traffic.shape[-1], self.ev_edge_temporal_data.shape[
            -1]

        # Final temporal data on nodes
        self.final_temporal_merged_data = edge_to_node_aggregation(self.edge_index_traffic,
                                                                   self.final_temporal_merged_data, len(self.nodes_df))
        print('Traffic and EV temporal data merging completed!')

    def preprocess_and_assemble_data(self):
        """
        Normalize data for each channel and finally create dataset self.encoded_data
        """
        # Prepare final data
        stacked_target = self.final_temporal_merged_data.to('cpu')
        self.number_of_station = self.final_temporal_merged_data.shape[0]

        # Calcola il Min e Max separato per ogni canale lungo le dimensioni (N, T)
        self.min_vals_normalization = stacked_target.min(dim=0)[0].min(dim=0)[0]  # Min lungo (N, T) per ogni canale
        self.max_vals_normalization = stacked_target.max(dim=0)[0].max(dim=0)[0]  # Max lungo (N, T) per ogni canale

        # Normalizza usando MinMax scaling
        standardized_target = ((stacked_target - self.min_vals_normalization) /
                               (self.max_vals_normalization - self.min_vals_normalization))

        # Input data
        self.features = [standardized_target[:, i: i + self.params.lags, :]
                         for i in
                         range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                               self.params.time_series_step)]

        # Output data
        N = standardized_target.shape[0]
        self.targets = [standardized_target[:, i:i + self.params.prediction_window, :].view(N, -1)
                        for i in range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                       self.params.time_series_step)]

        # Input time data
        self.time_input = [self.time_column[i: i + self.params.lags]
                           for i in
                           range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                                 self.params.time_series_step)]

        # Output time data
        self.time_output = [self.time_column[i:i + self.params.prediction_window]
                            for i in
                            range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                  self.params.time_series_step)]

        # Collect processed data on list
        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=self.edge_index_traffic.long(),
                                          edge_attr=self.edge_weights_traffic.float(),
                                          y=torch.FloatTensor(self.targets[i]),
                                          time_input=self.time_input[i],
                                          time_output=self.time_output[i]))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


def get_datamodule(params):
    # TSL-datasets style
    if params.dataset_name in ['METR-LA', 'electricity', 'solar']:
        if params.dataset_name == 'METR-LA':
            dataset = MetrLA(root='../data')
        elif params.dataset_name == 'electricity':
            dataset = Elergone(root='../data')
        else:
            raise ValueError(f'Dataset {params.dataset_name} not recognized')

        params.num_nodes = dataset.shape[1]
        connectivity = dataset.get_connectivity(threshold=0.1,
                                                include_self=True,
                                                normalize_axis=1,
                                                layout="edge_index")
        df_dataset = dataset.dataframe()
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Apply the scaler to the DataFrame
        df_dataset = pd.DataFrame(scaler.fit_transform(df_dataset), columns=df_dataset.columns)
        torch_dataset = SpatioTemporalDataset(target=df_dataset,
                                              connectivity=connectivity,  # edge_index
                                              horizon=params.prediction_window,
                                              window=params.lags,
                                              stride=1)
        # Normalize data using mean and std computed over time and node dimensions
        splitter = TemporalSplitter(val_len=0.2,
                                    test_len=0.1)
        data_module_instance = SpatioTemporalDataModule(
            dataset=torch_dataset,
            # scalers=scalers,
            splitter=splitter,
            batch_size=params.batch_size,
            workers=2
        )
    # Dataset from scratch
    elif params.dataset_name in ['denmark', 'newyork', 'chicago']:
        data_module_instance = EVDataModule(params)
        params = data_module_instance.run_params
    else:
        raise ValueError('Define dataset name correct!')

    return data_module_instance, params


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description="Experiments parameters!")
    parser.add_argument("--dataset_name", type=str, default='chicago',
                        help="['denmark', 'metr_la', 'newyork', 'chicago']")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size!")
    parser.add_argument("--model", type=str, default='GraphWavenet', help="Select model!")
    parser.add_argument("--verbose", "-v", action="store_false", help="Attiva output dettagliato")
    args = parser.parse_args()

    # Parameters
    instance_parameters = Parameters(args)

    # Datamodule
    dm = EVDataModule(instance_parameters)
    print('DM created!')
=======
import argparse
import glob
import json
import random
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as DataLoaderPyg
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from datetime import datetime, timedelta

from tsl.data import SpatioTemporalDataset, TemporalSplitter, SpatioTemporalDataModule
from tsl.data.preprocessing import MinMaxScaler
from tsl.datasets import MetrLA, Elergone
from torch_geometric.data import Data

from src.config import Parameters
from src.dataset.resamplling import resample_to_common_time
from src.dataset.utils import (create_adjacency_matrix, haversine, build_edges_with_node_ids, \
                               create_adjacency_matrix_newyork, augment_graph_df_v3, append_along_N_torch,
                               build_edges_with_node_ids_chicago,
                               process_traffic_metadata_newyork)
from src.utils.utils import directed_to_undirected, edge_to_node_aggregation


class EVDataModule(LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.run_params = params

        # Denmark dataset
        if self.run_params.dataset_name == 'denmark':
            self.run_params.traffic_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                        'data',
                                                                        self.run_params.dataset_name,
                                                                        'traffic/citypulse_traffic_raw_data_surrey_feb_jun_2014')
            self.run_params.traffic_metadata_file = os.path.join(self.run_params.project_path,
                                                                 'data',
                                                                 self.run_params.dataset_name,
                                                                 'traffic/trafficMetaData.csv')
            self.run_params.ev_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                   'data',
                                                                   self.run_params.dataset_name,
                                                                   'ev/available_connectors_counts')
            self.run_params.ev_metadata_file = os.path.join(self.run_params.project_path,
                                                            'data',
                                                            self.run_params.dataset_name,
                                                            'ev/charging_stations.csv')
            dataset = Dataset_Denmark(self.run_params)

        # Newyork dataset
        elif self.run_params.dataset_name == 'newyork':
            self.run_params.traffic_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                        'data',
                                                                        self.run_params.dataset_name,
                                                                        'traffic/traffic_data')
            self.run_params.traffic_metadata_file = os.path.join(self.run_params.project_path,
                                                                 'data',
                                                                 self.run_params.dataset_name,
                                                                 'traffic/processed_newyork_traffic_graph.csv')
            self.run_params.ev_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                   'data',
                                                                   self.run_params.dataset_name,
                                                                   'ev/stations_connectors_counts_data')
            self.run_params.ev_metadata_file = os.path.join(self.run_params.project_path,
                                                            'data',
                                                            self.run_params.dataset_name,
                                                            'ev/location_meta_data.csv')
            self.run_params.traffic_columns_to_use = ["speed", "travel_time"]
            self.run_params.ev_columns_to_use = ["Available"]
            dataset = DatasetNewyork(self.run_params)
            print(f'Dataset starting time: {dataset.start_time}')
            print(f'Dataset end time: {dataset.end_time}')
            print(f'Dataset traffic mean resolution: {dataset.traffic_resolution}')
            print(f'Dataset ev mean resolution: {dataset.ev_resolution}')


        # Chicago dataset
        elif self.run_params.dataset_name == 'chicago':
            self.run_params.traffic_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                        'data',
                                                                        self.run_params.dataset_name,
                                                                        'traffic/traffic')
            self.run_params.traffic_metadata_file = os.path.join(self.run_params.project_path,
                                                                 'data',
                                                                 self.run_params.dataset_name,
                                                                 'traffic/location_summary.csv')
            self.run_params.ev_temporal_data_folder = os.path.join(self.run_params.project_path,
                                                                   'data',
                                                                   self.run_params.dataset_name,
                                                                   'ev/ev_locations_availability')
            self.run_params.ev_metadata_file = os.path.join(self.run_params.project_path,
                                                            'data',
                                                            self.run_params.dataset_name,
                                                            'ev/ev_location_metadata.csv')
            self.run_params.traffic_columns_to_use = ["speed", "length"]
            self.run_params.ev_columns_to_use = ["Available"]
            dataset = DatasetChicago(self.run_params)
            print(f'Dataset starting time: {dataset.start_time}')
            print(f'Dataset end time: {dataset.end_time}')
            print(f'Dataset traffic mean resolution: {dataset.traffic_resolution}')
            print(f'Dataset ev mean resolution: {dataset.ev_resolution}')

        else:
            raise ValueError(f'Dataset {self.run_params.dataset_name} not recognized')

        self.num_station = dataset.number_of_station
        self.run_params.num_nodes = self.num_station
        self.run_params.traffic_features = dataset.traffic_features
        self.run_params.ev_features = dataset.ev_features
        self.run_params.traffic_features_names = dataset.traffic_columns_used_in_data.tolist()
        self.run_params.ev_features_names = dataset.ev_columns_used_in_data.tolist()
        self.run_params.min_vals_normalization = dataset.min_vals_normalization.tolist()
        self.run_params.max_vals_normalization = dataset.max_vals_normalization.tolist()
        self.run_params.start_time = dataset.start_time.isoformat()
        self.run_params.end_time = dataset.end_time.isoformat()
        self.run_params.traffic_resolution = dataset.traffic_resolution
        self.run_params.ev_resolution = dataset.ev_resolution

        # Split data
        split_data_modality = 'sequential'  # ['random', 'sequential']  # TODO: Metti in config
        if split_data_modality == 'random':
            len_dataset = len(dataset)
            train_snapshots = int(self.run_params.train_ratio * len_dataset)
            val_test_snapshots = len_dataset - train_snapshots
            val_snapshots = int(self.run_params.val_test_ratio * val_test_snapshots)
            test_snapshots = len_dataset - train_snapshots - val_snapshots
            self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset, [train_snapshots,
                                                                                                     val_snapshots,
                                                                                                     test_snapshots])  # N, T, F
        elif split_data_modality == 'sequential':
            len_dataset = len(dataset)
            train_snapshots = int(self.run_params.train_ratio * len_dataset)
            val_test_snapshots = len_dataset - train_snapshots
            val_snapshots = int(self.run_params.val_test_ratio * val_test_snapshots)
            test_snapshots = len_dataset - train_snapshots - val_snapshots
            start_point = random.randint(0, len_dataset - test_snapshots)
            test_start = start_point
            test_end = start_point + test_snapshots
            train_start = test_end
            train_end = train_start + train_snapshots
            val_start = train_end
            val_end = val_start + val_snapshots
            dataset_indices = np.arange(len_dataset).tolist() + np.arange(
                len_dataset).tolist()  # 2 times because restart from the beginning
            train_idx = dataset_indices[train_start:train_end]
            val_idx = dataset_indices[val_start:val_end]
            test_idx = dataset_indices[test_start:test_end]
            self.train_data = torch.utils.data.Subset(dataset, train_idx)
            self.val_data = torch.utils.data.Subset(dataset, val_idx)
            self.test_data = torch.utils.data.Subset(dataset, test_idx)

            # Here i'm sure by construction that test doesn't overlap the end of the dataset
            self.run_params.start_time_test = list(self.test_data[0].time_input)[0].isoformat()
            self.run_params.end_time_test = list(self.test_data[len(self.test_data) - 1].time_output)[-1].isoformat()

            # I don't need for train and val for now since i will utilize only test values
            # # If train idx don't restart from the beginning
            # if train_start < train_end:
            #     self.start_time_train = self.end_time_test
            #     self.end_time_train = self.start_time_train + timedelta(seconds=(train_end - train_start) * dataset.traffic_resolution * 60)
            # # If train idx restart from the beginning
            # else:
            #     self.start_time_train_0 = self.end_time_test
            #     self.end_time_train_0 = self.start_time_train + timedelta(seconds=(train_end - train_start) * dataset.traffic_resolution * 60)
            #
            # if val_start < val_end:
            #     self.start_time_val = self.end_time_train
            #     self.end_time_val = self.start_time_val + timedelta(seconds=(val_end - val_start) * dataset.traffic_resolution * 60)
            # else:
            #     pass
            # self.start_time_train = dataset.start_time + timedelta(seconds=train_start * dataset.traffic_resolution * 60)
            # self.end_time_train = dataset.start_time + timedelta(seconds=train_end * dataset.traffic_resolution * 60)
            # self.start_time_val = dataset.start_time + timedelta(seconds=val_start * dataset.traffic_resolution * 60)
            # self.end_time_val = dataset.start_time + timedelta(seconds=val_end * dataset.traffic_resolution * 60)


        else:
            raise ValueError(f'split_data_modality not recognized')

        # Check integrity
        if self.train_data is None:
            raise Exception("Dataset %s not supported" % self.run_params.dataset)

        # Get dataloaders
        self.train_loader = DataLoaderPyg(self.train_data,
                                          batch_size=self.run_params.batch_size,
                                          shuffle=True,
                                          drop_last=True)
        self.val_loader = DataLoaderPyg(self.val_data,
                                        batch_size=self.run_params.batch_size,
                                        drop_last=True)
        self.test_loader = DataLoaderPyg(self.test_data,
                                         batch_size=self.run_params.batch_size,
                                         drop_last=True)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


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
        map_ev_node_traffic_node = {}
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
            map_ev_node_traffic_node[ev_node_idx] = min_dist_traffic_node_idx

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to map_ev_node_traffic_node: create a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(self.N_t)]
        for key in map_ev_node_traffic_node.keys():
            corrispective_traffic_node = map_ev_node_traffic_node[key]
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


class DatasetNewyork(Dataset):
    _TRAFFIC_DATA_COLUMNS = ["speed", "travel_time", "status", "data_as_of", "link_id"]
    _TRAFFIC_METADATA_COLUMNS = ["id", "link_points", "points", "distances"]
    _EV_DATA_COLUMNS = ["location_id", "timestamp", "Available", "Total", "Offline"]
    _EV_METADATA_COLUMNS = ["LocID", "LocName", "Latitude", "Longitude"]

    def __init__(self, params, dtype=torch.float32, device="cuda"):
        # Params
        self.params = params
        self.dtype = dtype
        self.device = torch.device(device)

        # # Dataset spatial area (correlated to the specific dataset)
        self.min_lat, self.max_lat = 40.4, 40.95
        self.min_long, self.max_long = -74.5, -73.5

        # Common time between traffic and ev data
        self.start_time = None
        self.end_time = None

        # Other params
        self.traffic_resolution = None
        self.ev_resolution = None
        self.min_vals_normalization = None
        self.max_vals_normalization = None
        self.number_of_station = None
        self.features = None
        self.targets = None
        self.edge_index_traffic = None
        self.edge_weights_traffic = None
        self.edges_df = None
        self.nodes_df = None
        self.parsing_traffic_procedure = 'by_rows'
        self.added_edges_df = None
        self.coordinates_traffic = list()
        self.traffic_features, self.ev_features = None, None
        self.encoded_data = []

        preprocessed_filepath_merged_data = os.path.join(self.params.preprocessed_data_path,
                                                         'final_temporal_merged_data.pt')
        preprocessed_filepath_time_column = os.path.join(self.params.preprocessed_data_path,
                                                         'time_column.pt')
        preprocessed_edge_index_traffic = os.path.join(self.params.preprocessed_data_path,
                                                       'edge_index_traffic.pt')
        preprocessed_edge_weights_traffic = os.path.join(self.params.preprocessed_data_path,
                                                         'edge_weights_traffic.pt')
        preprocessed_dataset_config = os.path.join(self.params.preprocessed_data_path,
                                                   'dataset_config.pt')

        preprocessed_nodes_df = os.path.join(self.params.preprocessed_data_path,
                                                   'nodes_df.csv')
        preprocessed_edges_df = os.path.join(self.params.preprocessed_data_path,
                                                   'edges_df.csv')
        preprocessed_added_edges_df = os.path.join(self.params.preprocessed_data_path,
                                                   'added_edges_df.csv')

        if (os.path.exists(preprocessed_filepath_merged_data) and
                os.path.exists(preprocessed_filepath_time_column) and
                os.path.exists(preprocessed_edge_index_traffic) and
                os.path.exists(preprocessed_edge_weights_traffic) and
                os.path.exists(preprocessed_dataset_config) and
                os.path.exists(preprocessed_nodes_df) and
                os.path.exists(preprocessed_edges_df) and
                os.path.exists(preprocessed_added_edges_df)):
            self.final_temporal_merged_data = torch.load(preprocessed_filepath_merged_data, weights_only=False)
            self.time_column = torch.load(preprocessed_filepath_time_column, weights_only=False)
            self.edge_index_traffic = torch.load(preprocessed_edge_index_traffic, weights_only=False)
            self.edge_weights_traffic = torch.load(preprocessed_edge_weights_traffic, weights_only=False)

            self.dataset_config = torch.load(preprocessed_dataset_config, weights_only=False)
            self.number_of_station = self.dataset_config["number_of_station"]
            self.traffic_features = self.dataset_config["traffic_features"]
            self.ev_features = self.dataset_config["ev_features"]
            self.traffic_columns_used_in_data = self.dataset_config["traffic_columns_used_in_data"]
            self.ev_columns_used_in_data = self.dataset_config["ev_columns_used_in_data"]
            self.start_time = self.dataset_config["start_time"]
            self.end_time = self.dataset_config["end_time"]
            self.traffic_resolution = self.dataset_config["traffic_resolution"]
            self.ev_resolution = self.dataset_config["ev_resolution"]

            self.nodes_df = pd.read_csv(preprocessed_nodes_df)
            self.edges_df = pd.read_csv(preprocessed_edges_df)
            self.added__df = pd.read_csv(preprocessed_added_edges_df)

            self.preprocess_and_assemble_data()
        else:

            # Traffic filepaths
            self.filepaths = sorted(glob.glob(os.path.join(self.params.traffic_temporal_data_folder, "*.csv")))
            if not self.filepaths:
                raise RuntimeError(f"Nessun XLS trovato in {self.params.traffic_temporal_data_folder}")

            # Process traffic metadata in order to incorporate distances
            if not self.params.use_traffic_metadata_processed:
                process_traffic_metadata_newyork(self.params,
                                                 self.params.traffic_metadata_file)

            # Find temporal intersection window
            self.check_traffic_ev_time()

            # Load traffic temporal data
            self._load_traffic_data()

            # Load ev temporal data
            self._load_ev_data()

            # Since resolution is not precise, resample both traffic and ev data to the same timestep
            self.data_tensor_traffic, self.data_tensor_ev = resample_to_common_time(self.data_tensor_traffic,
                                                                                    self.data_tensor_ev,
                                                                                    target="A",
                                                                                    # resample to traffic resolution
                                                                                    method="linear")

            # Construct graph, with original nodes, original edges plus fake edges to allow whole graph communication
            self.get_edges_traffic(threshold=self.params.graph_distance_threshold)

            # Assign EV temporal features to the nearest traffic nodes and then traffic edges
            self.assign_ev_node_to_traffic_node()

            # Save or load
            torch.save(self.final_temporal_merged_data, preprocessed_filepath_merged_data)
            torch.save(self.time_column, preprocessed_filepath_time_column)
            torch.save(self.edge_index_traffic, preprocessed_edge_index_traffic)
            torch.save(self.edge_weights_traffic, preprocessed_edge_weights_traffic)

            dataset_config = {"number_of_station": self.number_of_station,
                              "traffic_features": self.traffic_features,
                              "ev_features": self.ev_features,
                              "traffic_columns_used_in_data": self.traffic_columns_used_in_data,
                              "ev_columns_used_in_data": self.ev_columns_used_in_data,
                              "start_time": self.start_time,
                              "end_time": self.end_time,
                              "traffic_resolution": self.traffic_resolution,
                              "ev_resolution": self.ev_resolution}
            torch.save(dataset_config, preprocessed_dataset_config)

            self.nodes_df.to_csv(preprocessed_nodes_df)
            self.edges_df.to_csv(preprocessed_edges_df)
            self.added_edges_df.to_csv(preprocessed_added_edges_df)

            # Normalize and stack data into fixed-size input/output windows objects
            self.preprocess_and_assemble_data()

    def load_preprocessed_data(self):
        """
        Load preprocessed data after temporal alignment and resolution resampling
        """
        load_tensor = torch.load(os.path.join(self.params.preprocessed_dataset_path,
                                              f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))
        data_tensor_traffic = load_tensor["data_tensor_traffic"]
        data_tensor_ev = load_tensor["data_tensor_ev"]
        return data_tensor_traffic, data_tensor_ev

    def check_traffic_ev_time(self):
        """
        This function define the start and end time of the synchronous traffic and time series. To decide it, it gets
        the overlapping time between traffic and ev timeseries.
        """
        ## Parsing traffic time series
        dfs = []  # Collecting dataFrame list per site
        sites = []  # Collecting Site ID per site

        # 'by_time' procedure collect traffic temporal data between different sites by aligning them at same time.
        # However, with raw data alignment at same time can cause problems!
        if self.parsing_traffic_procedure == 'by_time':
            seen = set()
            print('Loading Traffic data by_time...')
            for path in self.filepaths[:self.params.num_of_traffic_nodes_limit]:
                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort values by time
                df = df.sort_values("data_as_of").reset_index(drop=True)

                # Set ts index - convert la colonna TIMESTAMP da stringa ISO 8601 a datetime
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce")

                # Arrotonda al minuto (toglie secondi e millisecondi)
                df["data_as_of"] = df["data_as_of"].dt.floor("min")

                # Check different time index between files
                new_vals = set(df["data_as_of"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("data_as_of")

                # Cast
                feature_cols = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

                dfs.append(df)
                sites.append(site_id)

            # Index outer join
            union_index = dfs[0].index
            for d in dfs[1:]:
                union_index = union_index.union(d.index)

        # 'by_rows'  procedure collect traffic temporal data between different sites by aligning them at same rows
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[check_traffic_ev_time] Loading Traffic data by_rows...')

            # Collect traffic temporal data
            dfs = []
            sites = []

            # 1) First pass: Read and clean each traffic CSV, but do NOT set the index to time
            cont = 0
            for path in self.filepaths:
                # Check number of traffic sites constraint (defined by user)
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} lacks columns {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df = df.sort_values("data_as_of").reset_index(drop=True)
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce").dt.floor("min")

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols_ = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                feature_cols = [c for c in feature_cols_ if c in self.params.traffic_columns_to_use]
                df = df[["data_as_of"] + feature_cols]  # maintain desired order, keeping time as a column (not index)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
                cont += 1
                print(f'[check_traffic_ev_time]: Traffic site #{cont}/{len(self.filepaths)}')

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and cut from the end
            min_len = min(len(d) for d in dfs)
            if min_len == 0:
                raise ValueError("After cleaning, at least one CSV has no useful rows.")

            dfs_trimmed = [d.tail(min_len).reset_index(drop=True) for d in dfs]

            # 3) Get min and max timestamp needed for EV rows selections
            self.start_time = dfs_trimmed[0]['data_as_of'][0]
            self.end_time = dfs_trimmed[0]['data_as_of'][len(dfs_trimmed[0]['data_as_of']) - 1]

        ## Parsing ev time series
        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long))

        # Values of a specific column for the EXCLUDED rows
        excluded_vals = (ev_metadata.loc[~mask, "LocID"]).tolist()

        # Collect ev temporal data
        dfs = []  # DataFrame list per site
        sites = []  # site ID
        # ev_columns = ["timestamp","Available","Total","Offline"]
        ev_columns = DatasetNewyork._EV_DATA_COLUMNS  # TODO: choose the correct of features to use
        print('[check_traffic_ev_time] Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Check number of ev sites constraint (defined by user)
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break

            # Gather site id and do cut off distant nodes
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]
            if int(site_id) in excluded_vals:
                print(f'Skipping {site_id}')
                continue

            # Import EV data
            df = pd.read_csv(path, usecols=ev_columns)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            # Set timestamp to index
            df = df.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts
            df = df.set_index("timestamp")

            # Cast
            feature_cols_ = [c for c in ev_columns if c != "timestamp"]
            feature_cols = [c for c in feature_cols_ if c in self.params.ev_columns_to_use]
            df = df[feature_cols]  # mantiene ordine voluto
            df = df.apply(pd.to_numeric, errors="coerce")
            dfs.append(df)
            sites.append(site_id)

        # # Get low and high timestamp index
        union_index_ev = dfs[0].index
        for d in dfs[1:]:
            union_index_ev = union_index_ev.union(d.index)

        union_index_ev = union_index_ev.sort_values()
        low = union_index_ev.min()
        high = union_index_ev.max()

        # Finally, assign the highest timestamp values between traffic and ev starting time
        if low > self.start_time:
            self.start_time = low

        # Finally, assign the lowest timestamp values between traffic and ev finish time
        if high < self.end_time:
            self.end_time = high

    def _load_traffic_data(self,
                           pad_value=-1.0):
        """
        Load temporal traffic data collected for each edges
        """
        # Collect traffic data: dataFrame list per site and Site ID per site
        dfs = []  # Collecting dataFrame list per site
        sites = []  # Colelcting Site ID per site

        # Select procedure ['by_rows', 'by_time'] of aligning data on .csv files rows (starting from the end) or by time index (but unfiseable
        # since datetime differs among files of different amount of time)
        if self.parsing_traffic_procedure == 'by_time':
            seen = set()
            print('Loading Traffic data by_time...')
            for path in self.filepaths[:self.params.num_of_traffic_nodes_limit]:
                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort values by time
                df = df.sort_values("data_as_of").reset_index(drop=True)

                # Set ts index - convert la colonna TIMESTAMP da stringa ISO 8601 a datetime
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce")

                # Arrotonda al minuto (toglie secondi e millisecondi)
                df["data_as_of"] = df["data_as_of"].dt.floor("min")

                # Check different time index between files
                new_vals = set(df["data_as_of"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("data_as_of")

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
                feature_cols = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

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
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[_load_traffic_data] Loading Traffic data by_rows...')
            dfs = []
            sites = []
            resolutions = []

            # 1) First pass: Read and clean each CSV, but do NOT set the index to time.
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetNewyork._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetNewyork._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df = df.sort_values("data_as_of").reset_index(drop=True)
                df["data_as_of"] = pd.to_datetime(df["data_as_of"], format="ISO8601", errors="coerce").dt.floor("min")

                # Get resolution
                diffs = df['data_as_of'].diff().dropna().mean()
                resolutions.append(diffs.total_seconds() / 60)

                # (Optional) Handle time duplicates BEFORE cutting: average per minute
                if df["data_as_of"].duplicated().any():
                    # Calculates the average across rows with the same timestamp, then reorders by time
                    df = (df.groupby("data_as_of", as_index=False)
                          .mean(numeric_only=True)
                          .sort_values("data_as_of")
                          .reset_index(drop=True))
                    print(f'Find duplicate timestamps in {os.path.basename(path)} (collapsed by mean)')

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols_ = [c for c in DatasetNewyork._TRAFFIC_DATA_COLUMNS if c != "data_as_of"]
                feature_cols = [c for c in feature_cols_ if c in self.params.traffic_columns_to_use]
                df = df[["data_as_of"] + feature_cols]  # maintain desired order, keeping time as a column (not index)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
            print(f'[_load_traffic_data] Loaded {len(sites)} traffic sites')

            # Get mean resolution in minutes
            self.traffic_resolution = float(np.asarray(resolutions).mean())

            # 1) Time window on each df (dfs is the list of DataFrames)
            dfs_windowed = []
            for d in dfs:
                mask = d["data_as_of"].between(self.start_time, self.end_time, inclusive="both")
                d_cut = d.loc[mask].reset_index(drop=True)
                dfs_windowed.append(d_cut)

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and trim from the end
            min_len = min(len(d) for d in dfs_windowed)
            if min_len == 0:
                raise ValueError("No rows fall within the time window for at least one CSV.")
            dfs_trimmed_ = [d.tail(min_len).reset_index(drop=True) for d in dfs_windowed]

            # Order dfs in order to respect ID increasing order
            sites = [int(s) for s in sites]
            order = sorted(range(len(sites)), key=lambda i: sites[i])
            self.sites_sorted = [sites[i] for i in order]
            dfs_trimmed = [dfs_trimmed_[i] for i in order]

            # 3) Align by POSITION (same rows for all), concatenating by columns
            #    Build a single multi-index DataFrame on the columns: (site, feature)
            #    The index remains a RangeIndex 0..min_len-1 (position)
            dfs_features_only = [d.drop(columns=["data_as_of"]) for d in dfs_trimmed]  # if you no longer want timestamp
            df_all = pd.concat(dfs_features_only, axis=1, keys=sites, names=["site", "feature"])
            self.df_all = df_all  # already aligned per line; no union/padding
            self.timestamp_final_traffic = pd.RangeIndex(start=0, stop=min_len, name="row")  # indice posizionale
            self.traffic_columns_used_in_data = dfs_features_only[0].columns
            self.time_column = dfs_trimmed[0]["data_as_of"]

            # 4) Build the tensor: (N nodes, T timesteps (=min_len), M features)
            data = np.stack([d.values.astype(np.float32) for d in dfs_features_only], axis=0)
            self.data_tensor_traffic = torch.tensor(data, dtype=torch.float32, device=self.device)
            self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
            print(f'Loaded Traffic data by rows: {self.N_t} edges, {self.T_t} rows (tail), {self.M_t} features')

    def get_edges_traffic(self,
                          threshold=0.001,
                          distances_col='__distances'):
        """
        Function for creating the graph. Staring by traffic sites, we first build the original graph.
        Then, if the graph is not connected, we add fake edges to guarantee the connectivity of the graph.
        TO each fake edges is correlated zeros features.
        """
        # Load traffic metadata
        data = pd.read_csv(self.params.traffic_metadata_file, sep=';')

        # Deserialize i dati JSON, se necessario
        data['__points'] = data['__points'].apply(json.loads)
        data['__distances'] = data['__distances'].apply(json.loads)
        data['__distances'] = data['__distances'].apply(sum)
        data_sorted = data.sort_values(by="id")

        # Replace the 'id' column with increasing values from 0 to len(df)-1
        self.edge_id_mapping = dict(zip(data_sorted['id'], range(len(data_sorted))))  # old: new
        data_sorted['id'] = range(len(data_sorted))

        # Original edges
        edges_df, self.nodes_df = build_edges_with_node_ids(data_sorted,
                                                            threshold=threshold,
                                                            distances_col=distances_col)

        # Creates a set of unique arcs (normalizing orientation)
        # Newyork dataset: src_id 73 and 75 originals are the same
        unique_edges = set()
        duplicates = []
        for _, row in edges_df.iterrows():
            u_id = int(row['src_id'])
            v_id = int(row['tgt_id'])

            # Normalize the orientation of the arcs
            if u_id > v_id:
                u_id, v_id = v_id, u_id
            edge = frozenset((u_id, v_id))
            if edge in unique_edges:
                duplicates.append(row)
            else:
                unique_edges.add(edge)
        id_duplicates = [elem['id'] for elem in duplicates]

        # I delete duplicate arcs and update the index column
        edges_df = edges_df[~edges_df['id'].isin(id_duplicates)]
        edges_df['id'] = np.arange(len(edges_df))

        # Update temporal tensor to integrate duplicate arcs deletion
        rows_to_keep = [i for i in range(self.data_tensor_traffic.shape[0]) if i not in id_duplicates]
        self.data_tensor_traffic = self.data_tensor_traffic[rows_to_keep]

        # Create original adjacency matrix (if needed)
        orig_adj_matrix, _ = create_adjacency_matrix_newyork(edges_df['src_id'],
                                                             edges_df['tgt_id'],
                                                             num_nodes=len(self.nodes_df),
                                                             distance=edges_df['distance'])

        # Modify adjacency matrix by adding fake edges between nodes.
        # You should add edges for make the graph connected first, by checking on node distance then for example
        # Mimimal number of edges to have a graph connected is num_nodes -1
        self.edges_df, self.added_edges_df = augment_graph_df_v3(edges_df=edges_df,
                                                                 nodes_df=self.nodes_df)  # TODO: add directed/edges diciture

        adj_matrix, double_nodes = create_adjacency_matrix_newyork(self.edges_df['src_id'],
                                                                   self.edges_df['tgt_id'],
                                                                   num_nodes=len(self.nodes_df),
                                                                   distance=self.edges_df['distance'])

        # Create graph based on distance threshold (in Km)
        edge_index, edge_weights = dense_to_sparse(torch.tensor(adj_matrix))
        edge_index, edge_weights = directed_to_undirected(edge_index, edge_weights)
        self.edge_index_traffic, self.edge_weights_traffic = edge_index.to('cuda'), edge_weights.to('cuda')

        # Update temporal data with fake data (zeros features)
        self.data_tensor_traffic = append_along_N_torch(self.data_tensor_traffic,
                                                        len(self.added_edges_df),
                                                        fill='mean')

        print('[get_edges_traffic] Connected graph created!')

    def _load_ev_data(self,
                      pad_value=-1.0):
        """
        Function for loading EV temporal data and implementing some preprocessing activity:
            - Cut distant EV nodes
            - Check and substitute duplicate values
            - Check columns
            - Cut temporal data before and after start and end time values constraints
        """

        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (
                ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long)
        )

        filtered_ev_df = ev_metadata.loc[mask].copy()

        # values of a specific column for the EXCLUDED rows
        col = "LocID"
        excluded_vals = (ev_metadata.loc[~mask, col]).tolist()

        dfs = []  # DataFrame list per site
        sites = []  # site ID
        resolutions = []
        ev_columns = ["timestamp", "Available", "Total", "Offline"]
        print('Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]
            print(site_id)
            if int(site_id) in excluded_vals:
                print(f'Skipping {site_id}')
                continue

            df = pd.read_csv(path, usecols=ev_columns)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            df = df.sort_values("timestamp").reset_index(drop=True)

            # Set ts index
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts

            # Get resolution
            diffs = df['timestamp'].diff().dropna().mean()
            resolutions.append(diffs.total_seconds() / 60)

            df = df.set_index("timestamp")

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
            feature_cols_ = [c for c in ev_columns if c != "timestamp"]
            feature_cols = [c for c in feature_cols_ if c in self.params.ev_columns_to_use]
            df = df[feature_cols]  # mantiene ordine voluto
            df = df.apply(pd.to_numeric, errors="coerce")

            dfs.append(df)
            sites.append(site_id)

        print(f'[_load_ev_data] Loaded {len(sites)} EV sites')
        # Get mean resolution in minutes
        self.ev_resolution = float(np.asarray(resolutions).mean())

        # Index outer join
        union_index = dfs[0].index
        for d in dfs[1:]:
            union_index = union_index.union(d.index)
        # Here apply cut to align EV data to traffic data given start and end time variables
        union_index = union_index.sort_values()
        union_index = union_index[(union_index >= self.start_time) & (union_index <= self.end_time)]

        # Realign and constant padding
        dfs_aligned_ = [d.reindex(union_index).fillna(pad_value) for d in dfs]

        # Order dfs in order to respect ID increasing order
        order = sorted(range(len(sites)), key=lambda i: sites[i])
        sites_sorted = [sites[i] for i in order]
        dfs_aligned = [dfs_aligned_[i] for i in order]

        # Concat
        df_all = pd.concat(dfs_aligned, axis=1, keys=sites_sorted, names=["site", "feature"])
        self.df_ev_all = df_all.sort_index()  # (T_max, N*M)

        # Build EV tensor
        self.ev_columns_used_in_data = dfs_aligned[0].columns
        data = np.stack([d.values.astype(np.float32) for d in dfs_aligned], axis=0)
        self.data_tensor_ev = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.N_ev, self.T_ev, self.M_ev = self.data_tensor_ev.shape
        print(f'Loaded EV data: {self.N_ev} nodes, {self.T_ev} timesteps, {self.M_ev} features')

    def assign_ev_node_to_traffic_node(self):
        """
         We project EV node features into traffic nodes for aggregating information
         """
        # Load EV metadata
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)

        # Get EV coordinates
        ev_coordinates = list()
        for row in ev_metadata.iterrows():
            lat, lng = row[1]['Latitude'], row[1]['Longitude']
            ev_coordinates.append((lat, lng))

        # Map each EV node to nearest traffic node
        map_ev_node_traffic_node = {}
        print('Assigning EV to traffic nodes!')
        cont_ev = 0
        for ev_node_idx, ev_coord in enumerate(ev_coordinates):
            if cont_ev == self.params.num_of_ev_nodes_limit:
                break
            print(ev_node_idx)
            min_dist = float('inf')
            min_dist_traffic_node_idx = -1
            lat1, lon1 = ev_coord
            cont_traffic = 0
            for row in self.nodes_df.iterrows():
                if cont_traffic == self.params.num_of_traffic_nodes_limit:
                    break
                lat2, lon2, traffic_node_idx = row[1]['lat'], row[1]['lon'], row[1]['node_id']
                distance = haversine(lat1, lon1, lat2, lon2)
                if distance < min_dist:
                    min_dist = distance
                    min_dist_traffic_node_idx = traffic_node_idx
                cont_traffic += 1
            map_ev_node_traffic_node[ev_node_idx] = int(min_dist_traffic_node_idx)
            cont_ev += 1

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to map_ev_node_traffic_node: create a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(len(self.nodes_df))]
        for key in map_ev_node_traffic_node.keys():
            corrispective_traffic_node = map_ev_node_traffic_node[key]
            ev_values = self.data_tensor_ev[key]
            temp_list[corrispective_traffic_node].append(ev_values)

        lista_max = max(temp_list, key=len)
        tensor_ev_temp = torch.stack(lista_max)
        ev_timesteps = tensor_ev_temp.shape[1]

        new_temp_list = list()
        for elem in temp_list:
            if len(elem) == 0:
                new_temp_list.append(
                    torch.zeros(ev_timesteps, len(self.params.ev_columns_to_use)).to(self.params.device))
            elif len(elem) == 1:
                new_temp_list.append(elem[0])
            else:
                new_temp_list.append(torch.stack(elem).sum(0).squeeze(0))

        # inner join con self.timestamp_final_traffic e check sincronicità
        self.ev_temporal_data_on_merged_nodes = torch.stack(new_temp_list)
        print('Merged EV temporal data into traffic temporal data!')

        assert self.ev_temporal_data_on_merged_nodes.shape[1] == self.data_tensor_traffic.shape[1]

        # Now we have to:
        #  1) Firts, for consistency we aggregate EV node temporal data to traffic edges
        #  2) Then, once we have all edge temporal data, we collapse it on traffic nodes
        self.ev_edge_temporal_data = torch.zeros(self.data_tensor_traffic.shape[0],
                                                 self.data_tensor_traffic.shape[1],
                                                 self.ev_temporal_data_on_merged_nodes.shape[2], device='cpu')

        # Assumiamo:
        # self.ev_temporal_data_on_merged_nodes: [N, T, F_ev]  (EV su nodi traffic)
        # self.data_tensor_traffic:               [E, T, F_tr]
        # self.edges_df ha colonne 'src_id','tgt_id' ed è allineato riga-per-edge con gli indici [0..E-1]

        device = self.data_tensor_traffic.device
        nodes_ev = self.ev_temporal_data_on_merged_nodes.to(device)  # [N,T,F_ev]

        src = torch.as_tensor(self.edges_df['src_id'].values, device=device)  # [E]
        tgt = torch.as_tensor(self.edges_df['tgt_id'].values, device=device)  # [E]
        N = nodes_ev.shape[0]

        # gradi per nodo (evita div/0 con clamp)
        deg = torch.bincount(torch.cat([src, tgt]), minlength=N).clamp(min=1)  # [N]

        # pick EV dei capi e normalizza per grado
        ev_u = nodes_ev[src] / deg[src].view(-1, 1, 1)  # [E,T,F_ev]
        ev_v = nodes_ev[tgt] / deg[tgt].view(-1, 1, 1)  # [E,T,F_ev]

        # contributo EV edge = somma dei due capi normalizzati
        self.ev_edge_temporal_data = ev_u + ev_v  # [E,T,F_ev]

        # concateni con le feature traffic sulle edge
        self.final_temporal_merged_data = torch.cat([self.data_tensor_traffic.to(device),
                                                     self.ev_edge_temporal_data],
                                                    dim=-1)
        self.traffic_features = self.data_tensor_traffic.shape[-1]
        self.ev_features = self.ev_edge_temporal_data.shape[-1]

        # poi fai l'aggregazione edge->node (somma) come già fai
        self.final_temporal_merged_data = edge_to_node_aggregation(self.edge_index_traffic,
                                                                   self.final_temporal_merged_data,
                                                                   len(self.nodes_df))
        print('Traffic and EV temporal data merging completed!')

    def preprocess_and_assemble_data(self):
        # Prepare final data
        stacked_target = self.final_temporal_merged_data.to('cpu')
        self.number_of_station = self.final_temporal_merged_data.shape[0]

        # Calcola il Min e Max separato per ogni canale lungo le dimensioni (N, T)
        self.min_vals_normalization = stacked_target.min(dim=0)[0].min(dim=0)[0]  # Min lungo (N, T) per ogni canale
        self.max_vals_normalization = stacked_target.max(dim=0)[0].max(dim=0)[0]  # Max lungo (N, T) per ogni canale

        # Normalizza usando MinMax scaling
        standardized_target = ((stacked_target - self.min_vals_normalization) /
                               (self.max_vals_normalization - self.min_vals_normalization))

        # Input data
        self.features = [standardized_target[:, i: i + self.params.lags, :]
                         for i in
                         range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                               self.params.time_series_step)]

        # Output data
        N = standardized_target.shape[0]
        self.targets = [standardized_target[:, i:i + self.params.prediction_window, :].view(N, -1)
                        for i in range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                       self.params.time_series_step)]

        # Input time data
        self.time_input = [self.time_column[i: i + self.params.lags]
                           for i in
                           range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                                 self.params.time_series_step)]

        # Output time data
        self.time_output = [self.time_column[i:i + self.params.prediction_window]
                            for i in
                            range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                  self.params.time_series_step)]

        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=self.edge_index_traffic.long(),
                                          edge_attr=self.edge_weights_traffic.float(),
                                          y=torch.FloatTensor(self.targets[i]),
                                          time_input=self.time_input[i],
                                          time_output=self.time_output[i]), )

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


class DatasetChicago(Dataset):
    _TRAFFIC_DATA_COLUMNS = ['speed', 'length', 'time']
    _TRAFFIC_METADATA_COLUMNS = ['id', 'street', 'length', 'start_latitude', 'start_longitude', 'end_latitude',
                                 'end_longitude', 'max_speed']
    _EV_DATA_COLUMNS = ["location_id", "timestamp", "Available", "Total", "Offline"]
    _EV_METADATA_COLUMNS = ["LocID", "LocName", "Latitude", "Longitude"]

    def __init__(self, params, columns=None, dtype=torch.float32, device="cuda"):
        # Params
        self.params = params
        self.dtype = dtype
        self.device = torch.device(device)

        # Dataset spatial area (correlated to the specific dataset)
        self.min_lat, self.max_lat = 41.5, 42.1  # 41.6589702, 42.0128310
        self.min_long, self.max_long = -87.9, -87.4  # -87.8368335, -87.5350520

        # Common time between traffic and ev data
        self.start_time = None
        self.end_time = None

        # Other params
        self.traffic_resolution = None
        self.ev_resolution = None
        self.min_vals_normalization = None
        self.max_vals_normalization = None
        self.number_of_station = None
        self.features = None
        self.targets = None
        self.edge_index_traffic = None
        self.edge_weights_traffic = None
        self.edges_df = None
        self.nodes_df = None
        self.parsing_traffic_procedure = 'by_rows'
        self.added_edges_df = None
        self.coordinates_traffic = list()
        self.traffic_features, self.ev_features = None, None
        self.encoded_data = []

        preprocessed_filepath_merged_data = os.path.join(self.params.preprocessed_data_path,
                                                         'final_temporal_merged_data.pt')
        preprocessed_filepath_time_column = os.path.join(self.params.preprocessed_data_path,
                                                         'time_column.pt')
        preprocessed_edge_index_traffic = os.path.join(self.params.preprocessed_data_path,
                                                       'edge_index_traffic.pt')
        preprocessed_edge_weights_traffic = os.path.join(self.params.preprocessed_data_path,
                                                         'edge_weights_traffic.pt')
        preprocessed_dataset_config = os.path.join(self.params.preprocessed_data_path,
                                                   'dataset_config.pt')
        preprocessed_nodes_df = os.path.join(self.params.preprocessed_data_path,
                                                   'nodes_df.csv')
        preprocessed_edges_df = os.path.join(self.params.preprocessed_data_path,
                                                   'edges_df.csv')
        preprocessed_added_edges_df = os.path.join(self.params.preprocessed_data_path,
                                                   'added_edges_df.csv')

        if (os.path.exists(preprocessed_filepath_merged_data) and
                os.path.exists(preprocessed_filepath_time_column) and
                os.path.exists(preprocessed_edge_index_traffic) and
                os.path.exists(preprocessed_edge_weights_traffic) and
                os.path.exists(preprocessed_dataset_config) and
                os.path.exists(preprocessed_nodes_df) and
                os.path.exists(preprocessed_edges_df) and
                os.path.exists(preprocessed_added_edges_df)):
            self.final_temporal_merged_data = torch.load(preprocessed_filepath_merged_data, weights_only=False)
            self.time_column = torch.load(preprocessed_filepath_time_column, weights_only=False)
            self.edge_index_traffic = torch.load(preprocessed_edge_index_traffic, weights_only=False)
            self.edge_weights_traffic = torch.load(preprocessed_edge_weights_traffic, weights_only=False)

            self.dataset_config = torch.load(preprocessed_dataset_config, weights_only=False)
            self.number_of_station = self.dataset_config["number_of_station"]
            self.traffic_features = self.dataset_config["traffic_features"]
            self.ev_features = self.dataset_config["ev_features"]
            self.traffic_columns_used_in_data = self.dataset_config["traffic_columns_used_in_data"]
            self.ev_columns_used_in_data = self.dataset_config["ev_columns_used_in_data"]
            self.start_time = self.dataset_config["start_time"]
            self.end_time = self.dataset_config["end_time"]
            self.traffic_resolution = self.dataset_config["traffic_resolution"]
            self.ev_resolution = self.dataset_config["ev_resolution"]

            self.nodes_df = pd.read_csv(preprocessed_nodes_df)
            self.edges_df = pd.read_csv(preprocessed_edges_df)
            self.added__df = pd.read_csv(preprocessed_added_edges_df)

            self.preprocess_and_assemble_data()
        else:

            # Get all traffic filepaths
            self.filepaths = sorted(glob.glob(os.path.join(self.params.traffic_temporal_data_folder, "*.csv")))
            if not self.filepaths:
                raise RuntimeError(f"No CSV founded in {self.params.traffic_temporal_data_folder}")

            # # Check if preprocessed traffic data already exists. Be careful to use same creation parameters
            # processed_traffic_temporal_data_exist = os.path.exists(os.path.join(self.params.preprocessed_dataset_path,
            #                                                                     f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))
            #
            # # If you don't want to use preprocessed data or preprocessed data doesn't exist
            # if not (self.params.use_traffic_temporal_data_processed and processed_traffic_temporal_data_exist):

            # # Process metadata
            # process_traffic_metadata_chicago(self.params, self.params.traffic_metadata_file)

            # Find temporal intersection window
            self.check_traffic_ev_time()

            # Load traffic temporal data
            self._load_traffic_data()

            # Load ev temporal data
            self._load_ev_data()

            # Since resolution is not precise, resample both traffic and ev data to obtain the same timesteps
            self.data_tensor_traffic, self.data_tensor_ev = resample_to_common_time(self.data_tensor_traffic,
                                                                                    self.data_tensor_ev,
                                                                                    target="A",
                                                                                    method="linear")

            # Construct graph, with original nodes, original edges and add fake edges (if needed) to allow whole
            # graph communication.
            self.get_edges_traffic(threshold=self.params.graph_distance_threshold)

            # Assign EV temporal features to the nearest traffic nodes and then traffic edges
            self.assign_ev_node_to_traffic_node()

            # Save or load
            torch.save(self.final_temporal_merged_data, preprocessed_filepath_merged_data)
            torch.save(self.time_column, preprocessed_filepath_time_column)
            torch.save(self.edge_index_traffic, preprocessed_edge_index_traffic)
            torch.save(self.edge_weights_traffic, preprocessed_edge_weights_traffic)
            self.nodes_df.to_csv(preprocessed_nodes_df)
            self.edges_df.to_csv(preprocessed_edges_df)
            self.added_edges_df.to_csv(preprocessed_added_edges_df)

            dataset_config = {"number_of_station": self.number_of_station,
                              "traffic_features": self.traffic_features,
                              "ev_features": self.ev_features,
                              "traffic_columns_used_in_data": self.traffic_columns_used_in_data,
                              "ev_columns_used_in_data": self.ev_columns_used_in_data,
                              "start_time": self.start_time,
                              "end_time": self.end_time,
                              "traffic_resolution": self.traffic_resolution,
                              "ev_resolution": self.ev_resolution}
            torch.save(dataset_config, preprocessed_dataset_config)

            # Normalize and stack data into fixed-size input/output windows objects
            self.preprocess_and_assemble_data()

    def load_preprocessed_data(self):
        """
        Load preprocessed data after temporal alignment and resolution resampling
        """
        load_tensor = torch.load(os.path.join(self.params.preprocessed_dataset_path,
                                              f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))
        data_tensor_traffic = load_tensor["data_tensor_traffic"]
        data_tensor_ev = load_tensor["data_tensor_ev"]
        return data_tensor_traffic, data_tensor_ev

    def check_traffic_ev_time(self):
        """
        This function define the start and end time of the synchronous traffic and time series. To decide it, it gets
        the overlapping time between traffic and ev timeseries.
        """
        # Parsing traffic time series
        dfs = []  # Collecting dataFrame list per site
        sites = []  # Collecting Site ID per site

        # 'by_time' procedure collect traffic temporal data between different sites by aligning them at same time.
        # However, with raw data alignment at same time can cause problems!
        if self.parsing_traffic_procedure == 'by_time':
            print('Loading Traffic data by_time...')
            seen = set()
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break
                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Ordina per tempo e normalizza al minuto (così 'fine' = righe più recenti)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing con AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Check different time index between files
                new_vals = set(df["time"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("time")

                # Cast
                feature_cols = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]

                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

                dfs.append(df)
                sites.append(site_id)

            # Index outer join
            union_index = dfs[0].index
            for d in dfs[1:]:
                union_index = union_index.union(d.index)

        # 'by_rows'  procedure collect traffic temporal data between different sites by aligning them at same rows
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[check_traffic_ev_time] Loading Traffic data by_rows...')

            # Collect traffic temporal data
            dfs = []
            sites = []

            # 1) First pass: Read and clean each traffic CSV, but do NOT set the index to time.
            cont = 0
            for path in self.filepaths:
                # Check number of traffic sites constraint (defined by user)
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} lacks columns {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing with AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols_ = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]
                feature_cols = [c for c in feature_cols_ if c in self.params.traffic_columns_to_use]
                df = df[
                    ["time"] + feature_cols]  # mantieni ordine voluto, tenendo anche il tempo come colonna (non indice)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
                cont += 1
                print(f'[check_traffic_ev_time]: Traffic site #{cont}/{len(self.filepaths)}')

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and cut from the end
            min_len = min(len(d) for d in dfs)
            if min_len == 0:
                raise ValueError("After cleaning, at least one CSV has no useful rows.")

            dfs_trimmed = [d.tail(min_len).reset_index(drop=True) for d in dfs]

            # 3) Get min and max timestamp needed for EV rows selections
            self.start_time = dfs_trimmed[0]['time'][0]
            self.end_time = dfs_trimmed[0]['time'][len(dfs_trimmed[0]['time']) - 1]

        # Parsing ev time series
        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long))

        # Values of a specific column for the EXCLUDED rows
        excluded_vals = (ev_metadata.loc[~mask, "LocID"]).tolist()

        # Collect ev temporal data
        dfs = []  # DataFrame list per site
        sites = []  # site ID

        ev_columns = DatasetChicago._EV_DATA_COLUMNS
        # ev_columns = ["timestamp","Available","Total","Offline"]
        print('[check_traffic_ev_time] Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Check number of ev sites constraint (defined by user)
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break

            # Gather site id and do cut off distant nodes
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]
            if int(site_id) in excluded_vals:
                print(f'[check_traffic_ev_time] {site_id} EV site out of spatial range!')
                continue

            # Import EV data
            df = pd.read_csv(path, usecols=ev_columns)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            # Set timestamp to index
            df = df.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts
            df = df.set_index("timestamp")

            # Cast
            feature_cols_ = [c for c in ev_columns if c != "timestamp"]
            feature_cols = [c for c in feature_cols_ if c in self.params.ev_columns_to_use]

            df = df[feature_cols]
            df = df.apply(pd.to_numeric, errors="coerce")
            dfs.append(df)
            sites.append(site_id)

        # Get low and high timestamp index
        union_index_ev = dfs[0].index
        for d in dfs[1:]:
            union_index_ev = union_index_ev.union(d.index)

        union_index_ev = union_index_ev.sort_values()
        low = union_index_ev.min()
        high = union_index_ev.max()

        # Finally, assign the highest timestamp values between traffic and ev starting time
        if low > self.start_time:
            self.start_time = low

        # Finally, assign the lowest timestamp values between traffic and ev finish time
        if high < self.end_time:
            self.end_time = high

    def _load_traffic_data(self,
                           pad_value=-1.0):
        """
        Load temporal traffic data collected for each edges
        """
        # Collect traffic data: dataFrame list per site and Site ID per site
        dfs = []
        sites = []

        # Select procedure ['by_rows', 'by_time'] of aligning data on .csv files rows (starting from the end) or by
        # time index (but unfeaseable since datetime differs among files of different amount of time)

        if self.parsing_traffic_procedure == 'by_time':
            seen = set()
            print('Loading Traffic data by_time...')
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read XLS data per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # # Sort values by time
                # df = df.sort_values("time").reset_index(drop=True)

                # Ordina per tempo e normalizza al minuto (così 'fine' = righe più recenti)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing con AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Check different time index between files
                new_vals = set(df["time"]) - seen
                print(f"[{os.path.basename(path).split('.')[0]}] Nuovi:", len(new_vals))
                seen.update(new_vals)  # aggiorni il set

                # Imposta la colonna TIMESTAMP come indice
                df = df.set_index("time")

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
                feature_cols = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]  # old: data_as_of
                df = df[feature_cols]  # mantiene ordine voluto
                df = df.apply(pd.to_numeric, errors="coerce")
                site_id = os.path.basename(path).split(".")[0]

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
        elif self.parsing_traffic_procedure == 'by_rows':
            print('[_load_traffic_data] Loading Traffic data by_rows...')
            dfs = []
            sites = []
            resolutions = []

            # 1) First pass: Read and clean each CSV, but do NOT set the index to time.
            for path in self.filepaths:
                if len(sites) == self.params.num_of_traffic_nodes_limit:
                    break

                # Read CSV per site
                df = pd.read_csv(path, usecols=DatasetChicago._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetChicago._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%m/%d/%Y %I:%M:%S %p",  # parsing con AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Get resolution
                diffs = df['time'].diff().dropna().mean()
                resolutions.append(diffs.total_seconds() / 60)

                # (Optional) Handle time duplicates BEFORE cutting: average per minute
                if df["time"].duplicated().any():
                    print(f'Find duplicate timestamps in {os.path.basename(path)} (collapsed by mean)')
                    # calculates the average across rows with the same timestamp, then reorders by time
                    df = (df.groupby("time", as_index=False)
                          .mean(numeric_only=True)
                          .sort_values("time")
                          .reset_index(drop=True))

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols_ = [c for c in DatasetChicago._TRAFFIC_DATA_COLUMNS if c != "time"]
                feature_cols = [c for c in feature_cols_ if c in self.params.traffic_columns_to_use]
                df = df[
                    ["time"] + feature_cols]  # maintain desired order, also keeping the time as a column (not index)
                df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

                dfs.append(df)
                sites.append(os.path.basename(path).split(".")[0])
            print(f'[_load_traffic_data] Loaded {len(sites)} traffic sites')

            # Get mean resolution in minutes
            self.traffic_resolution = float(np.asarray(resolutions).mean())

            # 1) Time window on each df (dfs is the list of DataFrames)
            dfs_windowed = []
            for d in dfs:
                mask = d["time"].between(self.start_time, self.end_time, inclusive="both")
                d_cut = d.loc[mask].reset_index(drop=True)
                dfs_windowed.append(d_cut)

            # 2) Calculate the minimum number of rows between CSVs (after cleaning) and trim from the end
            min_len = min(len(d) for d in dfs_windowed)
            if min_len == 0:
                raise ValueError("No rows fall within the time window for at least one CSV.")
            dfs_trimmed_ = [d.tail(min_len).reset_index(drop=True) for d in dfs_windowed]

            # Order dfs in order to respect ID increasing order
            sites = [int(s) for s in sites]
            order = sorted(range(len(sites)), key=lambda i: sites[i])
            self.sites_sorted = [sites[i] for i in order]
            dfs_trimmed = [dfs_trimmed_[i] for i in order]

            # 3) Align by POSITION (same rows for all), concatenating by columns
            #    Build a single multi-index DataFrame on the columns: (site, feature)
            #    The index remains a RangeIndex 0..min_len-1 (position).
            dfs_features_only = [d.drop(columns=["time"]) for d in dfs_trimmed]  # se non vuoi più il timestamp
            df_all = pd.concat(dfs_features_only, axis=1, keys=sites, names=["site", "feature"])
            self.df_all = df_all  # already aligned per line; no union/padding
            self.timestamp_final_traffic = pd.RangeIndex(start=0, stop=min_len, name="row")  # positional index
            self.traffic_columns_used_in_data = dfs_features_only[0].columns
            self.time_column = dfs_trimmed[0]["time"]

            # 4) Build the tensor: (N nodes, T timesteps (=min_len), M features)
            data = np.stack([d.values.astype(np.float32) for d in dfs_features_only], axis=0)
            self.data_tensor_traffic = torch.tensor(data, dtype=torch.float32, device=self.device)
            self.N_t, self.T_t, self.M_t = self.data_tensor_traffic.shape
            print(
                f'Loaded Traffic data from csv files by rows: {self.N_t} edges, {self.T_t} rows (tail), {self.M_t} features')

    def get_edges_traffic(self,
                          threshold: float = 0.001,
                          distances_col='length'):
        """
        Function for creating the graph. Staring by traffic sites, we first build the original graph.
        Then, if the graph is not connected, we add fake edges to guarantee the connectivity of the graph.
        TO each fake edges is correlated zeros features.
        """
        # Load traffic metadata
        data = pd.read_csv(self.params.traffic_metadata_file)
        data_sorted = data.sort_values(by="id")

        # Original edges
        edges_df, self.nodes_df = build_edges_with_node_ids_chicago(data_sorted,
                                                                    threshold=threshold,
                                                                    distances_col=distances_col)  # exact match

        # Creates a set of unique arcs (normalizing orientation)
        unique_edges = set()
        duplicates = []
        for _, row in edges_df.iterrows():
            u_id = int(row['src_id'])
            v_id = int(row['tgt_id'])

            # Normalize the orientation of the arcs
            if u_id > v_id:
                u_id, v_id = v_id, u_id
            edge = frozenset((u_id, v_id))
            if edge in unique_edges:
                duplicates.append(row)
            else:
                unique_edges.add(edge)
        id_duplicates = [elem['id'] for elem in duplicates]

        # I delete duplicate arcs and update the index column
        edges_df = edges_df[~edges_df['id'].isin(id_duplicates)]
        edges_df['id'] = np.arange(len(edges_df))

        # Update temporal tensor to integrate duplicate arcs deletion
        rows_to_keep = [i for i in range(self.data_tensor_traffic.shape[0]) if i not in id_duplicates]
        self.data_tensor_traffic = self.data_tensor_traffic[rows_to_keep]

        # Create original adjacency matrix (if needed)
        # orig_adj_matrix, _ = create_adjacency_matrix_newyork(edges_df['src_id'],
        #                                                   edges_df['tgt_id'],
        #                                                   num_nodes=len(self.nodes_df),
        #                                                   distance=edges_df['distance'])

        # Modify adjacency matrix by adding fake edges between nodes.
        # You should add edges for make the graph connected
        self.edges_df, self.added_edges_df = augment_graph_df_v3(edges_df=edges_df,
                                                                 nodes_df=self.nodes_df)  # TODO: add directed/edges diciture

        adj_matrix, double_nodes = create_adjacency_matrix_newyork(self.edges_df['src_id'],
                                                                   self.edges_df['tgt_id'],
                                                                   num_nodes=len(self.nodes_df),
                                                                   distance=self.edges_df['distance'])

        # Create graph based on distance threshold (in Km) (DELETE)
        edge_index, edge_weights = dense_to_sparse(torch.tensor(adj_matrix))
        edge_index, edge_weights = directed_to_undirected(edge_index, edge_weights)
        self.edge_index_traffic, self.edge_weights_traffic = edge_index.to('cuda'), edge_weights.to('cuda')

        # Update temporal data with fake data (zeros features)
        self.data_tensor_traffic = append_along_N_torch(self.data_tensor_traffic,
                                                        len(self.added_edges_df),
                                                        fill='mean')

        print('[get_edges_traffic] Connected graph created!')

    def _load_ev_data(self,
                      pad_value=-1.0,
                      col="LocID"):
        """
        Function for loading EV temporal data and implementing some preprocessing activity:
            - Cut distant EV nodes
            - Check and substitute duplicate values
            - Check columns
            - Cut temporal data before and after start and end time values constraints
        """

        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (
                ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long)
        )

        # filtered_ev_df = ev_metadata.loc[mask].copy()
        excluded_vals = (ev_metadata.loc[~mask, col]).tolist()  # Values of a specific column for the EXCLUDED rows

        # Collect data
        dfs = []  # DataFrame list per site
        sites = []  # site ID
        resolutions = []
        ev_columns = ["timestamp", "Available", "Total", "Offline"]  # without location ID
        print('Loading EV data ...')

        # Gathering EV files
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Constraints on EV sites number
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]

            # Cut off too distant EV sites
            if int(site_id) in excluded_vals:
                print(f'Skipping {site_id} since it is too distant!')
                continue

            # Load EV data
            df = pd.read_csv(path, usecols=DatasetChicago._EV_DATA_COLUMNS)

            # Control missing columns
            missing = set(ev_columns) - set(df.columns)
            if missing:
                raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

            # Set ts index
            df = df.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(df["timestamp"])
            df["timestamp"] = ts

            # Get resolution
            diffs = df['timestamp'].diff().dropna().mean()
            resolutions.append(diffs.total_seconds() / 60)

            df = df.set_index("timestamp")

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
            feature_cols_ = [c for c in ev_columns if c != "timestamp"]
            feature_cols = [c for c in feature_cols_ if c in self.params.ev_columns_to_use]

            df = df[feature_cols]  # maintains the desired order
            df = df.apply(pd.to_numeric, errors="coerce")
            dfs.append(df)
            sites.append(site_id)

        print(f'[_load_ev_data] Loaded {len(sites)} EV sites')

        # Get mean resolution in minutes
        self.ev_resolution = float(np.asarray(resolutions).mean())

        # Index outer join
        # Here apply cut to align EV data to traffic data given start and end time variables
        union_index = dfs[0].index
        for d in dfs[1:]:
            union_index = union_index.union(d.index)
        union_index = union_index.sort_values()
        union_index = union_index[(union_index >= self.start_time) & (union_index <= self.end_time)]
        dfs_aligned_ = [d.reindex(union_index).fillna(pad_value) for d in dfs]  # Realign and constant padding

        # Order dfs in order to respect ID increasing order
        order = sorted(range(len(sites)), key=lambda i: sites[i])
        sites_sorted = [sites[i] for i in order]
        dfs_aligned = [dfs_aligned_[i] for i in order]

        # Concat
        df_all = pd.concat(dfs_aligned, axis=1, keys=sites_sorted, names=["site", "feature"])
        self.df_ev_all = df_all.sort_index()  # (T_max, N*M)

        # Build EV tensor
        self.ev_columns_used_in_data = dfs_aligned[0].columns
        data = np.stack([d.values.astype(np.float32) for d in dfs_aligned], axis=0)
        self.data_tensor_ev = torch.tensor(data, dtype=torch.float32, device=self.device)
        self.N_ev, self.T_ev, self.M_ev = self.data_tensor_ev.shape
        print(f'Loaded EV data: {self.N_ev} nodes, {self.T_ev} timesteps, {self.M_ev} features')

    def assign_ev_node_to_traffic_node(self):
        """
        We project EV node features into traffic nodes for aggregating information
        """
        # Load EV metadata
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)

        # Get EV coordinates
        ev_coordinates = list()
        for row in ev_metadata.iterrows():
            lat, lng = row[1]['Latitude'], row[1]['Longitude']
            ev_coordinates.append((lat, lng))

        # Map each EV node to nearest traffic node
        map_ev_node_traffic_node = {}
        print('Assigning EV to traffic nodes!')
        cont_ev = 0
        for ev_node_idx, ev_coord in enumerate(ev_coordinates):
            if cont_ev == self.params.num_of_ev_nodes_limit:
                break
            min_dist = float('inf')
            min_dist_traffic_node_idx = -1
            lat1, lon1 = ev_coord
            cont_traffic = 0
            for row in self.nodes_df.iterrows():
                if cont_traffic == self.params.num_of_traffic_nodes_limit:
                    break
                lat2, lon2, traffic_node_idx = row[1]['lat'], row[1]['lon'], row[1]['node_id']
                distance = haversine(lat1, lon1, lat2, lon2)
                if distance < min_dist:
                    min_dist = distance
                    min_dist_traffic_node_idx = traffic_node_idx
                cont_traffic += 1
            map_ev_node_traffic_node[ev_node_idx] = int(min_dist_traffic_node_idx)
            cont_ev += 1

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to map_ev_node_traffic_node. So we're creating a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(len(self.nodes_df))]
        for key in map_ev_node_traffic_node.keys():
            corrispective_traffic_node = map_ev_node_traffic_node[key]
            ev_values = self.data_tensor_ev[key]
            temp_list[corrispective_traffic_node].append(ev_values)

        lista_max = max(temp_list, key=len)
        tensor_ev_temp = torch.stack(lista_max)
        ev_timesteps = tensor_ev_temp.shape[1]

        new_temp_list = list()
        for elem in temp_list:
            if len(elem) == 0:
                new_temp_list.append(
                    torch.zeros(ev_timesteps, len(self.params.ev_columns_to_use)).to(self.params.device))  # TODO: change 3 to selected features
            elif len(elem) == 1:
                new_temp_list.append(elem[0])
            else:
                new_temp_list.append(torch.stack(elem).sum(0).squeeze(0))

        # Inner join con self.timestamp_final_traffic e check sincronicità
        self.ev_temporal_data_on_merged_nodes = torch.stack(new_temp_list)
        print('Merged EV temporal data into traffic temporal data!')

        # Check temporal consistency
        assert self.ev_temporal_data_on_merged_nodes.shape[1] == self.data_tensor_traffic.shape[1]

        # Now we have to:
        #  1) Firts, for consistency we aggregate EV node temporal data to traffic edges
        #  2) Then, once we have all edge temporal data, we collapse it on traffic nodes
        self.ev_edge_temporal_data = torch.zeros(self.data_tensor_traffic.shape[0],
                                                 self.data_tensor_traffic.shape[1],
                                                 self.ev_temporal_data_on_merged_nodes.shape[2], device='cpu')

        device = self.data_tensor_traffic.device
        nodes_ev = self.ev_temporal_data_on_merged_nodes.to(device)  # [N,T,F_ev]

        src = torch.as_tensor(self.edges_df['src_id'].values, device=device)  # [E]
        tgt = torch.as_tensor(self.edges_df['tgt_id'].values, device=device)  # [E]
        N = nodes_ev.shape[0]

        # gradi per nodo (evita div/0 con clamp)
        deg = torch.bincount(torch.cat([src, tgt]), minlength=N).clamp(min=1)  # [N]

        # pick EV dei capi e normalizza per grado
        ev_u = nodes_ev[src] / deg[src].view(-1, 1, 1)  # [E,T,F_ev]
        ev_v = nodes_ev[tgt] / deg[tgt].view(-1, 1, 1)  # [E,T,F_ev]

        # contributo EV edge = somma dei due capi normalizzati
        self.ev_edge_temporal_data = ev_u + ev_v  # [E,T,F_ev]

        # concateni con le feature traffic sulle edge
        self.final_temporal_merged_data = torch.cat([self.data_tensor_traffic.to(device),
                                                     self.ev_edge_temporal_data],
                                                    dim=-1)
        self.traffic_features = self.data_tensor_traffic.shape[-1]
        self.ev_features = self.ev_edge_temporal_data.shape[-1]

        # poi fai l'aggregazione edge->node (somma) come già fai
        self.final_temporal_merged_data = edge_to_node_aggregation(self.edge_index_traffic,
                                                                   self.final_temporal_merged_data,
                                                                   len(self.nodes_df))
        ############################################

        print('Traffic and EV temporal data merging completed!')

    def preprocess_and_assemble_data(self):
        """
        Normalize data for each channel and finally create dataset self.encoded_data
        """
        # Prepare final data
        stacked_target = self.final_temporal_merged_data.to('cpu')
        self.number_of_station = self.final_temporal_merged_data.shape[0]

        # Calcola il Min e Max separato per ogni canale lungo le dimensioni (N, T)
        self.min_vals_normalization = stacked_target.min(dim=0)[0].min(dim=0)[0]  # Min lungo (N, T) per ogni canale
        self.max_vals_normalization = stacked_target.max(dim=0)[0].max(dim=0)[0]  # Max lungo (N, T) per ogni canale

        # Normalizza usando MinMax scaling
        standardized_target = ((stacked_target - self.min_vals_normalization) /
                               (self.max_vals_normalization - self.min_vals_normalization))

        # Input data
        self.features = [standardized_target[:, i: i + self.params.lags, :]
                         for i in
                         range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                               self.params.time_series_step)]

        # Output data
        N = standardized_target.shape[0]
        self.targets = [standardized_target[:, i:i + self.params.prediction_window, :].view(N, -1)
                        for i in range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                       self.params.time_series_step)]

        # Input time data
        self.time_input = [self.time_column[i: i + self.params.lags]
                           for i in
                           range(0, standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                                 self.params.time_series_step)]

        # Output time data
        self.time_output = [self.time_column[i:i + self.params.prediction_window]
                            for i in
                            range(self.params.lags, standardized_target.shape[1] - self.params.prediction_window,
                                  self.params.time_series_step)]

        # Collect processed data on list
        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=self.edge_index_traffic.long(),
                                          edge_attr=self.edge_weights_traffic.float(),
                                          y=torch.FloatTensor(self.targets[i]),
                                          time_input=self.time_input[i],
                                          time_output=self.time_output[i]))

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


def get_datamodule(params):
    # TSL-datasets style
    if params.dataset_name in ['METR-LA', 'electricity', 'solar']:
        if params.dataset_name == 'METR-LA':
            dataset = MetrLA(root='../data')
        elif params.dataset_name == 'electricity':
            dataset = Elergone(root='../data')
        else:
            raise ValueError(f'Dataset {params.dataset_name} not recognized')

        params.num_nodes = dataset.shape[1]
        connectivity = dataset.get_connectivity(threshold=0.1,
                                                include_self=True,
                                                normalize_axis=1,
                                                layout="edge_index")
        df_dataset = dataset.dataframe()
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Apply the scaler to the DataFrame
        df_dataset = pd.DataFrame(scaler.fit_transform(df_dataset), columns=df_dataset.columns)
        torch_dataset = SpatioTemporalDataset(target=df_dataset,
                                              connectivity=connectivity,  # edge_index
                                              horizon=params.prediction_window,
                                              window=params.lags,
                                              stride=1)
        # Normalize data using mean and std computed over time and node dimensions
        splitter = TemporalSplitter(val_len=0.2,
                                    test_len=0.1)
        data_module_instance = SpatioTemporalDataModule(
            dataset=torch_dataset,
            # scalers=scalers,
            splitter=splitter,
            batch_size=params.batch_size,
            workers=2
        )
    # Dataset from scratch
    elif params.dataset_name in ['denmark', 'newyork', 'chicago']:
        data_module_instance = EVDataModule(params)
        params = data_module_instance.run_params
    else:
        raise ValueError('Define dataset name correct!')

    return data_module_instance, params


if __name__ == '__main__':
    # Args
    parser = argparse.ArgumentParser(description="Experiments parameters!")
    parser.add_argument("--dataset_name", type=str, default='chicago',
                        help="['denmark', 'metr_la', 'newyork', 'chicago']")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size!")
    parser.add_argument("--model", type=str, default='GraphWavenet', help="Select model!")
    parser.add_argument("--verbose", "-v", action="store_false", help="Attiva output dettagliato")
    args = parser.parse_args()

    # Parameters
    instance_parameters = Parameters(args)

    # Datamodule
    dm = EVDataModule(instance_parameters)
    print('DM created!')
>>>>>>> 50dc9fda781f8e9df4fc6cce2a2ff266f6b1e2a0

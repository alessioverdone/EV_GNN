import glob
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from src.dataset.resamplling import resample_to_common_time
from src.dataset.old.visualize_raw_data import visualize_processed_graph
from src.dataset.utils import (haversine,
                               create_adjacency_matrix_newyork,
                               augment_graph_df_v3,
                               append_along_N_torch,
                               build_edges_with_node_ids_chicago,
                               clean_tensor, is_good_ev_file_v2,
                               select_ev_features)
from src.utils.utils import directed_to_undirected, edge_to_node_aggregation


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
        preprocessed_map_ev_node_traffic_node = os.path.join(self.params.preprocessed_data_path,
                                                         'map_ev_node_traffic_node.pt')
        preprocessed_map_real_ev_node_traffic_node = os.path.join(self.params.preprocessed_data_path,
                                                         'map_real_ev_node_traffic_node.pt')
        preprocessed_dataset_config = os.path.join(self.params.preprocessed_data_path,
                                                   'dataset_config.pt')
        preprocessed_merged_traffic_nodes_map = os.path.join(self.params.preprocessed_data_path,
                                                             'merged_traffic_nodes_map.pt')
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
                os.path.exists(preprocessed_added_edges_df) and
                os.path.exists(preprocessed_map_ev_node_traffic_node) and
                os.path.exists(preprocessed_map_real_ev_node_traffic_node)and
                os.path.exists(preprocessed_merged_traffic_nodes_map)):
            self.final_temporal_merged_data = torch.load(preprocessed_filepath_merged_data, weights_only=False)
            self.time_column = torch.load(preprocessed_filepath_time_column, weights_only=False)
            self.edge_index_traffic = torch.load(preprocessed_edge_index_traffic, weights_only=False)
            self.edge_weights_traffic = torch.load(preprocessed_edge_weights_traffic, weights_only=False)
            self.map_ev_node_traffic_node = torch.load(preprocessed_map_ev_node_traffic_node, weights_only=False)
            self.map_real_ev_node_traffic_node = torch.load(preprocessed_map_real_ev_node_traffic_node, weights_only=False)
            self.merged_traffic_nodes_map = torch.load(preprocessed_merged_traffic_nodes_map, weights_only=False)

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
            self.added_edges_df = pd.read_csv(preprocessed_added_edges_df)

            self.preprocess_and_assemble_data()
        else:

            # Get all traffic filepaths
            self.filepaths = sorted(glob.glob(os.path.join(self.params.traffic_temporal_data_folder, "*.csv")))
            if not self.filepaths:
                raise RuntimeError(f"No CSV founded in {self.params.traffic_temporal_data_folder}")

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

            # # Clean data of -1 values by substituting channel and node mean values
            self.final_temporal_merged_data = clean_tensor(self.final_temporal_merged_data)

            # Save or load
            torch.save(self.final_temporal_merged_data, preprocessed_filepath_merged_data)
            torch.save(self.time_column, preprocessed_filepath_time_column)
            torch.save(self.edge_index_traffic, preprocessed_edge_index_traffic)
            torch.save(self.edge_weights_traffic, preprocessed_edge_weights_traffic)
            torch.save(self.map_ev_node_traffic_node, preprocessed_map_ev_node_traffic_node)
            torch.save(self.map_real_ev_node_traffic_node, preprocessed_map_real_ev_node_traffic_node)
            torch.save(self.merged_traffic_nodes_map, preprocessed_merged_traffic_nodes_map)

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

            # # >>> NUOVO: scarta i file "non buoni" (check stringente)
            # if not is_good_ev_file_v2(path):
            #     print(f'Skipping {site_id} since it does not pass the quality check!')
            #     continue

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

            # Select requested EV features (raw and/or derived, e.g. AvailabilityRate)
            df = select_ev_features(df, self.params.ev_columns_to_use)
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
                          threshold: float = 0.001,  # 1mm
                          distances_col='length'):
        """
        Function for creating the graph. Staring by traffic sites, we first build the original graph.
        Then, if the graph is not connected, we add fake edges to guarantee the connectivity of the graph.
        To each fake edges is correlated zeros features.
        We merge also too close nodes in order to avoid road crossing with 4 nodes instead of 1
        """
        # Load traffic metadata
        data = pd.read_csv(self.params.traffic_metadata_file)
        data_sorted = data.sort_values(by="id")

        # Original edges, with  self.nodes_df: (id, lat, long) and edges_df: (id,src, tgt, distance, src_id, tgt_id)
        edges_df, self.nodes_df, self.merged_traffic_nodes_map = build_edges_with_node_ids_chicago(data_sorted,
                                                                    threshold=threshold,
                                                                    distances_col=distances_col)  # exact match

        # Creates a set of unique arcs (normalizing orientation)
        unique_edges = set()
        unique_edges_info = dict()
        duplicate_temporal_data_info = dict()
        duplicates = []
        ordinal_id_duplicates = []
        row_id = -1
        for _, row in edges_df.iterrows():
            row_id += 1
            u_id = int(row['src_id'])
            v_id = int(row['tgt_id'])

            # Normalize the orientation of the arcs
            if u_id > v_id:
                u_id, v_id = v_id, u_id
            edge = frozenset((u_id, v_id))

            # Duplicate info saving
            if edge in unique_edges:
                duplicates.append(row)
                ordinal_id_duplicates.append(row_id)

                # Collect info to merge temporal data
                if edge not in duplicate_temporal_data_info.keys():
                    duplicate_temporal_data_info[edge] = [[unique_edges_info[edge], (row['id'], row_id)]]
                else:
                    duplicate_temporal_data_info[edge].append([unique_edges_info[edge], (row['id'], row_id)])
            else:
                unique_edges.add(edge)
                unique_edges_info[edge] = (row['id'], row_id)
        id_duplicates = [elem['id'] for elem in duplicates]

        # First, sum/mean up duplicated information
        for list_of_duplicates_per_edge_key in duplicate_temporal_data_info.keys():
            temporal_edge_info = list()
            temporal_edge_info.append(self.data_tensor_traffic[list(list_of_duplicates_per_edge_key)[1],:,:].unsqueeze(dim=0))
            for edges_ids in duplicate_temporal_data_info[list_of_duplicates_per_edge_key]:
                for id in edges_ids:
                    temporal_edge_info.append(self.data_tensor_traffic[id[1],:,:].unsqueeze(dim=0))

            temporal_edge_info = torch.stack(temporal_edge_info, dim=0)
            self.data_tensor_traffic[list(list_of_duplicates_per_edge_key)[1], :, :] = temporal_edge_info.sum(dim=0)

        # I delete duplicate arcs and update the index column
        edges_df = edges_df[~edges_df['id'].isin(id_duplicates)]
        edges_df['id'] = np.arange(len(edges_df))

        # Update temporal tensor to integrate duplicate arcs deletion
        rows_to_keep = [i for i in range(self.data_tensor_traffic.shape[0]) if i not in ordinal_id_duplicates]  # ERRORE: indici aggiornati e non cancella niente
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

        # Create graph based on distance threshold (in Km) (DELETE/UPDATE)
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

            # # >>> NUOVO: scarta i file "non buoni" (check stringente)
            # if not is_good_ev_file_v2(path):
            #     print(f'Skipping {site_id} since it does not pass the quality check!')
            #     continue

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

            # Select requested EV features (raw and/or derived, e.g. AvailabilityRate)
            df = select_ev_features(df, self.params.ev_columns_to_use)
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

    def assign_ev_node_to_traffic_node(self,
                                       col="LocID"):
        """
        We project EV node features into traffic nodes for aggregating information
        """
        # Load EV metadata
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (
                ev_metadata['Latitude'].between(self.min_lat, self.max_lat) &
                ev_metadata['Longitude'].between(self.min_long, self.max_long)
        )

        # filtered_ev_df = ev_metadata.loc[mask].copy()
        excluded_vals = (ev_metadata.loc[~mask, col]).tolist()  # Values of a specific column for the EXCLUDED rows

        # Get EV coordinates
        ev_coordinates = list()
        for row in ev_metadata.iterrows():
            if row[1]['LocID'] in excluded_vals:
                continue
            else:
                lat, lng = row[1]['Latitude'], row[1]['Longitude']
                loc_id = row[1]['LocID']
                ev_coordinates.append((lat, lng, loc_id))

        # Map each EV node to nearest traffic node
        self.map_ev_node_traffic_node = {}
        self.map_real_ev_node_traffic_node = {}
        print('Assigning EV to traffic nodes!')
        cont_ev = 0
        for ev_node_idx, ev_coord_and_id in enumerate(ev_coordinates):
            if cont_ev == self.params.num_of_ev_nodes_limit:
                break
            min_dist = float('inf')
            min_dist_traffic_node_idx = -1
            lat1, lon1, loc_id = ev_coord_and_id
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
            self.map_ev_node_traffic_node[ev_node_idx] = int(min_dist_traffic_node_idx)
            self.map_real_ev_node_traffic_node[loc_id] = int(min_dist_traffic_node_idx)
            cont_ev += 1

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to self.map_ev_node_traffic_node. So we're creating a list of list with len = num_of_traffic_nodes
        temp_list = [[] for _ in range(len(self.nodes_df))]
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
                new_temp_list.append(
                    torch.zeros(ev_timesteps, len(self.params.ev_columns_to_use)).to(self.params.device))  # TODO: change 3 to selected features
            elif len(elem) == 1:
                new_temp_list.append(elem[0])
            else:
                new_temp_list.append(torch.stack(elem).sum(0).squeeze(0))  # mean?

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

        # Degrees per node (avoid div/0 with clamp)
        deg = torch.bincount(torch.cat([src, tgt]), minlength=N).clamp(min=1)  # [N]

        # Pick EV of the extremes and normalize by degree
        ev_u = nodes_ev[src] / deg[src].view(-1, 1, 1)  # [E,T,F_ev]
        ev_v = nodes_ev[tgt] / deg[tgt].view(-1, 1, 1)  # [E,T,F_ev]

        # EV edge contribution = sum of the two normalized ends
        self.ev_edge_temporal_data = ev_u + ev_v  # [E,T,F_ev]

        # concatenate with feature traffic on the edges
        self.final_temporal_merged_data = torch.cat([self.data_tensor_traffic.to(device),
                                                     self.ev_edge_temporal_data],
                                                    dim=-1)
        self.traffic_features = self.data_tensor_traffic.shape[-1]
        self.ev_features = self.ev_edge_temporal_data.shape[-1]

        # then do the edge->node aggregation (sum) as you already do
        self.final_temporal_merged_data = edge_to_node_aggregation(self.edge_index_traffic,
                                                                   self.final_temporal_merged_data,
                                                                   len(self.nodes_df))
        print('Traffic and EV temporal data merging completed!')

    def preprocess_and_assemble_data(self):
        """
        Normalize data for each channel and finally create dataset self.encoded_data
        """
        # # Clean data of -1 values by substituting channel and node mean values
        # self.final_temporal_merged_data = clean_tensor(self.final_temporal_merged_data)

        # Prepare final data
        self.stacked_target = self.final_temporal_merged_data.to('cpu')
        self.number_of_station = self.final_temporal_merged_data.shape[0]

        # Calcola il Min e Max separato per ogni canale lungo le dimensioni (N, T)
        self.min_vals_normalization = self.stacked_target.min(dim=0)[0].min(dim=0)[0]  # Min lungo (N, T) per ogni canale
        self.max_vals_normalization = self.stacked_target.max(dim=0)[0].max(dim=0)[0]  # Max lungo (N, T) per ogni canale

        # Normalizza usando MinMax scaling
        self.standardized_target = ((self.stacked_target - self.min_vals_normalization) /
                               (self.max_vals_normalization - self.min_vals_normalization))

        # Input data
        self.features = [self.standardized_target[:, i: i + self.params.lags, :]
                         for i in
                         range(0, self.standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                               self.params.time_series_step)]

        # Output data
        N = self.standardized_target.shape[0]
        self.targets = [self.standardized_target[:, i:i + self.params.prediction_window, :].view(N, -1)
                        for i in range(self.params.lags, self.standardized_target.shape[1] - self.params.prediction_window,
                                       self.params.time_series_step)]

        # Input time data
        self.time_input = [self.time_column[i: i + self.params.lags]
                           for i in
                           range(0, self.standardized_target.shape[1] - self.params.lags - self.params.prediction_window,
                                 self.params.time_series_step)]

        # Output time data
        self.time_output = [self.time_column[i:i + self.params.prediction_window]
                            for i in
                            range(self.params.lags, self.standardized_target.shape[1] - self.params.prediction_window,
                                  self.params.time_series_step)]

        # Collect processed data on list
        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=self.edge_index_traffic.long(),
                                          edge_attr=self.edge_weights_traffic.float(),
                                          y=torch.FloatTensor(self.targets[i]),
                                          time_input=self.time_input[i],
                                          time_output=self.time_output[i]))

        if getattr(self.params, 'visualize_data', True):
            visualize_processed_graph(
                edges_df=self.edges_df,
                added_edges_df=self.added_edges_df,
                nodes_df=self.nodes_df,
                highlight_added=True,
                file_html=os.path.join(self.params.preprocessed_data_path, 'processed_graph.html'),
            )

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx]


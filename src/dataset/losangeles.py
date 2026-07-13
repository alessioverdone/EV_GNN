import glob
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from src.dataset.losangeles_raw_preprocessing.losangeles_check_ev_files import is_good_ev_file
from src.dataset.resamplling import resample_to_common_time
from src.dataset.old.visualize_raw_data import visualize_processed_graph
from src.dataset.utils import (haversine, create_adjacency_matrix_newyork, augment_graph_df_v3, clean_tensor,
                               select_ev_features)
from src.utils.utils import directed_to_undirected


class DatasetLosangeles(Dataset):
    _TRAFFIC_DATA_COLUMNS = ['avg_speed', 'total_flow', 'avg_occupancy', 'samples', 'observed', 'time']
    _TRAFFIC_METADATA_COLUMNS = ['id', 'street', 'length', 'latitude', 'longitude']
    _EV_DATA_COLUMNS = ['location_id', 'timestamp', 'Available', 'Total', 'Offline', 'In_use']
    _EV_METADATA_COLUMNS = ['station_id', 'title', 'supplier_name', 'lat', 'lng', 'num_units',
                            'connector_types', 'max_power_kw', 'access', 'house_number', 'street',
                            'city', 'district', 'county', 'state_code', 'postal_code',
                            'country_code', 'here_station_id', 'first_seen_utc', 'last_seen_utc']

    def __init__(self, params, columns=None, dtype=torch.float32, device="cuda"):
        # Params
        self.params = params
        self.dtype = dtype
        self.device = torch.device(device)

        # Dataset spatial area (correlated to the specific dataset)
        self.min_lat, self.max_lat = 33., 35.
        self.min_long, self.max_long = -120, -117
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
                df = pd.read_csv(path, usecols=DatasetLosangeles._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetLosangeles._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Ordina per tempo e normalizza al minuto (così 'fine' = righe più recenti)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%Y-%m-%d %H:%M:%S",  # parsing con AM/PM
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
                feature_cols = [c for c in DatasetLosangeles._TRAFFIC_DATA_COLUMNS if c != "time"]

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
                df = pd.read_csv(path, usecols=DatasetLosangeles._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetLosangeles._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} lacks columns {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%Y-%m-%d %H:%M:%S",  # parsing with AM/PM
                    errors="coerce"
                ).dt.floor("min")
                df = df.sort_values("time").reset_index(drop=True)

                # Cast: Keep only the features (without the time column) and convert to numeric
                feature_cols_ = [c for c in DatasetLosangeles._TRAFFIC_DATA_COLUMNS if c != "time"]
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
        mask = (ev_metadata['lat'].between(self.min_lat, self.max_lat) &
                ev_metadata['lng'].between(self.min_long, self.max_long))

        # Values of a specific column for the EXCLUDED rows
        excluded_vals = (ev_metadata.loc[~mask, "station_id"]).tolist()

        # Collect ev temporal data
        dfs = []  # DataFrame list per site
        sites = []  # site ID

        ev_columns = DatasetLosangeles._EV_DATA_COLUMNS
        # ev_columns = ["timestamp","Available","Total","Offline"]
        print('[check_traffic_ev_time] Loading EV data ...')
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Check number of ev sites constraint (defined by user)
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break

            # Gather site id and do cut off distant nodes
            site_id = str(os.path.basename(path)).split('.')[0]  # o usa os.path.basename(path).split(".")[0]
            # if int(site_id) in excluded_vals:
            if site_id in excluded_vals:
                print(f'[check_traffic_ev_time] {site_id} EV site out of spatial range!')
                continue

            # >>> NUOVO: scarta i file "non buoni" (check stringente)
            if not is_good_ev_file(path):
                print(f'Skipping {site_id} since it does not pass the quality check!')
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
                df = pd.read_csv(path, usecols=DatasetLosangeles._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetLosangeles._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # # Sort values by time
                # df = df.sort_values("time").reset_index(drop=True)

                # Ordina per tempo e normalizza al minuto (così 'fine' = righe più recenti)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%Y-%m-%d %H:%M:%S",  # parsing con AM/PM
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
                feature_cols = [c for c in DatasetLosangeles._TRAFFIC_DATA_COLUMNS if c != "time"]  # old: data_as_of
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
                df = pd.read_csv(path, usecols=DatasetLosangeles._TRAFFIC_DATA_COLUMNS)

                # Control missing columns
                missing = set(DatasetLosangeles._TRAFFIC_DATA_COLUMNS) - set(df.columns)
                if missing:
                    raise ValueError(f"{os.path.basename(path)} manca di colonne {missing}")

                # Sort by time and normalize to the minute (so 'end' = newest rows)
                df["time"] = pd.to_datetime(
                    df["time"],
                    format="%Y-%m-%d %H:%M:%S",  # parsing con AM/PM
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
                feature_cols_ = [c for c in DatasetLosangeles._TRAFFIC_DATA_COLUMNS if c != "time"]
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
                          threshold: float = 300.0,  # interchange distance threshold (meters)
                          distances_col='length'):
        """
        Build the Los Angeles road graph (NODE-centric).

        Unlike Chicago/New York -- where traffic measurements live on the EDGES of
        the graph -- in Los Angeles the PeMS stations ARE the nodes and the temporal
        traffic tensor is already node-aligned. So here we:
          1) build the road topology from the PeMS postmile metadata (backbone per
             freeway+direction + interchange edges) via ``los_angeles_graph.build_graph``;
          2) keep only the stations that actually have a temporal CSV (intersection
             with ``self.sites_sorted``) and re-index the nodes 0..N-1 so that
             ``self.data_tensor_traffic[i]`` corresponds to ``node_id == i``;
          3) guarantee connectivity by adding fake edges (``augment_graph_df_v3``),
             which here is "free": data is on the nodes, so no temporal feature
             padding is needed (no ``append_along_N_torch``, no edge dedup/merge).

        ``threshold`` is used as the interchange distance threshold (meters).
        """
        from src.dataset.losangeles_raw_preprocessing import losangeles_build_traffic_graph as lag

        # --- 1) Build the road topology from PeMS postmile metadata --------------
        # NOTE: this needs the RAW d07 metadata (with Fwy/Dir/Abs_PM), NOT the
        # chicago-style location_summary.csv (which has no postmile information).
        meta_file = getattr(self.params, 'traffic_raw_metadata_file', lag.DEFAULT_META_FILE)
        graph = lag.build_graph(meta_file=meta_file,
                                station_types=("ML",),
                                directed=False,            # undirected, coherent with the downstream graph
                                add_interchanges=True,
                                interchange_threshold_m=float(threshold),
                                weight="distance",         # weights are rebuilt below anyway
                                save=False)
        nodes_full = graph["nodes"]          # node_id, id(station), fwy, dir, abs_pm, lat, lon, ...
        edges_full = graph["edges"]          # src_station, dst_station, distance_m, kind, src_node, dst_node, weight

        # --- 2) Align graph stations with stations that actually have data -------
        data_stations = [int(s) for s in self.sites_sorted]            # aligned with data_tensor_traffic rows
        graph_stations = set(int(s) for s in nodes_full["id"].tolist())
        common = sorted(set(data_stations) & graph_stations)
        if len(common) == 0:
            raise RuntimeError("[get_edges_traffic] No station in common between traffic CSVs and PeMS metadata.")

        n_dropped_data = len(data_stations) - len(common)
        n_dropped_graph = len(graph_stations) - len(common)
        print(f'[get_edges_traffic] {len(common)} stations in common '
              f'(dropped {n_dropped_data} without metadata, {n_dropped_graph} without temporal data).')

        # Reorder/select the traffic tensor rows to follow `common` (sorted by station id)
        data_row_of_station = {sid: i for i, sid in enumerate(data_stations)}
        keep_rows = [data_row_of_station[s] for s in common]
        self.data_tensor_traffic = self.data_tensor_traffic[keep_rows]
        self.sites_sorted = common

        # --- 3) Re-index nodes 0..N-1 over the common stations (sorted by id) ----
        nodes_df = nodes_full[nodes_full["id"].astype(int).isin(common)].copy()
        nodes_df = nodes_df.sort_values("id").reset_index(drop=True)
        nodes_df["node_id"] = np.arange(len(nodes_df))
        station_to_node = dict(zip(nodes_df["id"].astype(int), nodes_df["node_id"].astype(int)))
        # nodes_df must expose at least node_id/lat/lon (used by assign and augment)
        self.nodes_df = nodes_df[["node_id", "lat", "lon", "id", "fwy", "dir"]].copy()

        # LA does not need the 4-way-crossing node merging of Chicago -> identity map
        self.merged_traffic_nodes_map = {int(n): int(n) for n in self.nodes_df["node_id"].tolist()}

        # --- 4) Build the original edge list, restricted/remapped to kept nodes --
        e = edges_full.copy()
        e = e[e["src_station"].astype(int).isin(common) & e["dst_station"].astype(int).isin(common)]
        edges_df = pd.DataFrame({
            "src_id": e["src_station"].astype(int).map(station_to_node).values,
            "tgt_id": e["dst_station"].astype(int).map(station_to_node).values,
            "distance": e["distance_m"].astype(float).values,   # meters, consistent with augment_graph_df_v3
        })
        # drop self-loops and duplicate undirected pairs
        edges_df = edges_df[edges_df["src_id"] != edges_df["tgt_id"]]
        norm_pair = edges_df.apply(lambda r: frozenset((int(r["src_id"]), int(r["tgt_id"]))), axis=1)
        edges_df = edges_df[~norm_pair.duplicated()].reset_index(drop=True)
        edges_df.insert(0, "id", np.arange(len(edges_df)))

        # --- 5) Make the graph connected with fake edges (classic procedure) -----
        self.edges_df, self.added_edges_df = augment_graph_df_v3(
            edges_df=edges_df,
            nodes_df=self.nodes_df[["node_id", "lat", "lon"]])

        # --- 6) Adjacency -> (undirected) edge_index / edge_weights --------------
        adj_matrix, double_nodes = create_adjacency_matrix_newyork(self.edges_df['src_id'],
                                                                   self.edges_df['tgt_id'],
                                                                   num_nodes=len(self.nodes_df),
                                                                   distance=self.edges_df['distance'])
        edge_index, edge_weights = dense_to_sparse(torch.tensor(adj_matrix))
        edge_index, edge_weights = directed_to_undirected(edge_index, edge_weights)
        self.edge_index_traffic = edge_index.to(self.device)
        self.edge_weights_traffic = edge_weights.to(self.device)

        # NOTE: traffic data stays node-aligned [N, T, F]; no edge dedup/merge and
        # no append_along_N_torch (fake edges carry no temporal features here).
        print('[get_edges_traffic] Connected graph created!')

    def _load_ev_data(self,
                      pad_value=-1.0,
                      col="station_id"):
        """
        Function for loading EV temporal data and implementing some preprocessing activity:
            - Cut distant EV nodes
            - Check and substitute duplicate values
            - Check columns
            - Cut temporal data before and after start and end time values constraints

        NOTE (Los Angeles): EV metadata columns are 'station_id' (string id, e.g.
        'S01395'), 'lat', 'lng'. Station ids are kept as strings everywhere in the
        class for coherence.
        """

        # Cut off too distant EV sites
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (
                ev_metadata['lat'].between(self.min_lat, self.max_lat) &
                ev_metadata['lng'].between(self.min_long, self.max_long)
        )

        # filtered_ev_df = ev_metadata.loc[mask].copy()
        excluded_vals = set(ev_metadata.loc[~mask, col].astype(str).tolist())  # EXCLUDED station ids (strings)

        # Collect data
        dfs = []  # DataFrame list per site
        sites = []  # site ID
        resolutions = []
        # ev_columns = ["timestamp", "Available", "Total", "Offline"]  # without location ID
        # _EV_DATA_COLUMNS = ['location_id', 'timestamp', 'Available', 'Total', 'Offline', 'In_use']
        ev_columns = DatasetLosangeles._EV_DATA_COLUMNS.copy()
        ev_columns.remove("location_id")
        print('Loading EV data ...')

        # Gathering EV files
        for path in sorted(glob.glob(os.path.join(self.params.ev_temporal_data_folder, "*.csv"))):
            # Constraints on EV sites number
            if len(sites) == self.params.num_of_ev_nodes_limit:
                break
            site_id = str(os.path.basename(path)).split('.')[0]  # station id as string, e.g. 'S01395'

            # Cut off too distant EV sites
            if site_id in excluded_vals:
                print(f'Skipping {site_id} since it is too distant!')
                continue

            # >>> NUOVO: scarta i file "non buoni" (check stringente)
            if not is_good_ev_file(path):
                print(f'Skipping {site_id} since it does not pass the quality check!')
                continue

            # Load EV data
            df = pd.read_csv(path, usecols=DatasetLosangeles._EV_DATA_COLUMNS)

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

        # Keep the EV station order so that data_tensor_ev row i <-> ev_sites_sorted[i].
        # assign_ev_node_to_traffic_node() matches EV rows by station id (not position).
        self.ev_sites_sorted = sites_sorted

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
                                       col="station_id"):
        """
        Node-centric EV -> traffic merge for Los Angeles.

        Traffic data is already node-aligned, so (differently from Chicago/New York)
        we do NOT project EV onto edges and we do NOT run edge->node aggregation:
        we simply map each EV station to its nearest traffic node, aggregate the EV
        temporal series on those nodes, and concatenate it channel-wise with the
        node-aligned traffic tensor.
        """
        # --- EV metadata (filter by bounding box) --------------------------------
        ev_metadata = pd.read_csv(self.params.ev_metadata_file)
        mask = (
                ev_metadata['lat'].between(self.min_lat, self.max_lat) &
                ev_metadata['lng'].between(self.min_long, self.max_long)
        )
        excluded_vals = set(ev_metadata.loc[~mask, col].astype(str).tolist())

        # Row of each EV station inside data_tensor_ev (set in _load_ev_data).
        # Matching by station_id (not by positional index) keeps EV temporal rows
        # aligned even when _load_ev_data skipped files (quality check / limits).
        ev_id_to_row = {str(sid): i for i, sid in enumerate(self.ev_sites_sorted)}

        # --- Map each (loaded) EV station to its nearest traffic node ------------
        self.map_ev_node_traffic_node = {}        # ev row index in data_tensor_ev -> traffic node_id
        self.map_real_ev_node_traffic_node = {}   # real EV station_id            -> traffic node_id
        print('Assigning EV to traffic nodes!')

        # Precompute traffic node coordinates once (respecting the node limit)
        traffic_nodes = self.nodes_df
        if self.params.num_of_traffic_nodes_limit != -1:
            traffic_nodes = traffic_nodes.iloc[:self.params.num_of_traffic_nodes_limit]
        traffic_coords = list(zip(traffic_nodes['lat'].values,
                                  traffic_nodes['lon'].values,
                                  traffic_nodes['node_id'].astype(int).values))

        for _, row in ev_metadata.iterrows():
            station_id = str(row[col])
            if station_id in excluded_vals:
                continue
            if station_id not in ev_id_to_row:        # no temporal data loaded for this station
                continue
            ev_row = ev_id_to_row[station_id]
            lat1, lon1 = row['lat'], row['lng']

            min_dist = float('inf')
            best_node = -1
            for lat2, lon2, node_id in traffic_coords:
                d = haversine(lat1, lon1, lat2, lon2)
                if d < min_dist:
                    min_dist = d
                    best_node = node_id
            self.map_ev_node_traffic_node[ev_row] = int(best_node)
            self.map_real_ev_node_traffic_node[station_id] = int(best_node)

        # --- Aggregate EV temporal series onto traffic nodes --------------------
        device = self.data_tensor_traffic.device
        ev_timesteps = self.data_tensor_ev.shape[1]
        n_ev_features = self.data_tensor_ev.shape[2]

        temp_list = [[] for _ in range(len(self.nodes_df))]
        for ev_row, traffic_node in self.map_ev_node_traffic_node.items():
            temp_list[traffic_node].append(self.data_tensor_ev[ev_row])

        new_temp_list = []
        for elem in temp_list:
            if len(elem) == 0:
                new_temp_list.append(torch.zeros(ev_timesteps, n_ev_features, device=device))
            elif len(elem) == 1:
                new_temp_list.append(elem[0].to(device))
            else:
                new_temp_list.append(torch.stack(elem).sum(0).to(device))   # sum of EV stations on the same node
        self.ev_temporal_data_on_merged_nodes = torch.stack(new_temp_list)   # [N, T, F_ev]
        print('Merged EV temporal data into traffic nodes!')

        # Temporal consistency between traffic and EV
        assert self.ev_temporal_data_on_merged_nodes.shape[1] == self.data_tensor_traffic.shape[1]

        # --- Node-centric concat: traffic(nodes) || ev(nodes) -------------------
        self.final_temporal_merged_data = torch.cat(
            [self.data_tensor_traffic.to(device),
             self.ev_temporal_data_on_merged_nodes.to(device)],
            dim=-1)                                                          # [N, T, F_traffic + F_ev]
        self.traffic_features = self.data_tensor_traffic.shape[-1]
        self.ev_features = self.ev_temporal_data_on_merged_nodes.shape[-1]
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


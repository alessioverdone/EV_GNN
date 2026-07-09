import csv
import glob
import json
import random
import os
import re
from pathlib import Path
from typing import Optional, List, Tuple
import folium
import numpy as np
from folium import DivIcon
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

from src.dataset.resamplling import resample_to_common_time
from src.dataset.old.visualize_raw_data import (visualize_processed_graph,
                                                parse_link_points)
from src.dataset.utils import (haversine,
                               build_edges_with_node_ids,
                               create_adjacency_matrix_newyork,
                               augment_graph_df_v3,
                               append_along_N_torch,
                               clean_tensor)
from src.utils.utils import (directed_to_undirected,
                             edge_to_node_aggregation)


def process_newyork_ev_stations(ev_csv: str) -> List[Tuple[float, float, str]]:
    """
    Legge un CSV con colonne ['LocID','LocName','Latitude','Longitude']
    e ritorna una lista di (lat, lon, id).
    """
    df_ev = pd.read_csv(ev_csv)
    ev_list: List[Tuple[float, float, str]] = []
    for _, row in df_ev.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        ev_id = str(row.get('LocID', row.get('LocName', '')))
        ev_list.append((lat, lon, ev_id))
    return ev_list


def _colore_random(rng: random.Random) -> str:
    return "#{:06x}".format(rng.randint(0, 0xFFFFFF))


def parse_link_points(
        seq: str,
        len_decimali_considered: int = 1,
        stampa_distanze: bool = True
) -> List[Tuple[float, float]]:
    """
    Estrae solo token lat,lon con precisione minima in lon e,
    se richiesto, stampa la distanza in metri tra ogni coppia consecutiva.

    :param seq: stringa dei link_points
    :param len_decimali_considered: numero minimo di decimali per la lon
    :param stampa_distanze: se True, stamperà le distanze tra i punti
    :return: lista di (lat, lon) valide
    """
    # Compiliamo una regexp per catturare lat e lon decimali
    _TOKEN_RE = re.compile(r'^(-?\d+\.\d+),(-?\d+\.\d+)$')

    pts: List[Tuple[float, float]] = []
    if not isinstance(seq, str):
        return pts
    list_dist = list()
    # parsing token validi
    for token in seq.split():
        m = _TOKEN_RE.match(token)
        if not m:
            continue
        lat_s, lon_s = m.group(1), m.group(2)
        if len(lon_s.split('.', 1)[1]) < len_decimali_considered:
            continue
        pts.append((float(lat_s), float(lon_s)))

    # calcolo e stampa distanze
    if stampa_distanze and len(pts) >= 2:
        for i in range(1, len(pts)):
            p_prev, p_cur = pts[i - 1], pts[i]
            d = haversine(p_prev, p_cur)
            print(f"Distanza tra punto {i - 1} {p_prev} e punto {i} {p_cur}: {d:.2f} m")
            list_dist.append(d)

    return pts, list_dist



def process_traffic_metadata_newyork(params,
                             percorso_csv_non_processed: str,
                             file_html: str = "grafo_stradale.html",
                            zoom_start: int = 12,
                            visualize_map: bool = True,
                            save_map: bool = True,
                            usa_satellite: bool = True,
                            mostra_nodi: bool = False,
                            mostra_popup_id: bool = True,
                            mostra_label: bool = False,
                            show_ev: bool = False,  # ← flag per EV
                            ev_csv: Optional[str] = None,  # ← percorso al CSV EV
                            seed_colori: Optional[int] = 42,
                            weight: int = 3,
                            opacity: float = 0.8,
                            outlier_value_selector: int = 60):

    #  Caricamento e parsing
    df = pd.read_csv(percorso_csv_non_processed)
    if "link_points" not in df or "id" not in df:
        raise ValueError("Manca colonna 'id' o 'link_points' nel CSV.")

    # Construct __points and  __distances columns
    df["__points"] = None
    df["__distances"] = None
    all_seq_distances = list()
    for idx, row in df.iterrows():
        pts, list_dist = parse_link_points(row["link_points"], stampa_distanze=True)
        all_seq_distances.append(list_dist)
        if len(list_dist) == 1:
            df.at[idx, "__points"] = pts
            df.at[idx, "__distances"] = list_dist
        else:
            x = np.array(list_dist)  # la tua sequenza
            M = np.median(x)
            mad = np.median(np.abs(x - M))
            r = np.abs(x - M) / mad
            mask = r <= outlier_value_selector  # seleziona gli indici senza outlier
            if False in mask:
                print(r)
            list_dist = np.array(list_dist)[mask]
            cont = 0
            pass_step = False
            for i in range(len(mask)):
                # I need this since I've 2 distance over the range but only one point is the responsible
                if pass_step:
                    pass_step = False
                    continue
                if not mask[i]:
                    _ = pts.pop(i + 1 + cont)
                    cont -= 1
                    pass_step = True

            # pts = np.array(pts)[mask]
            df.at[idx, "__points"] = pts
            df.at[idx, "__distances"] = list_dist

    # print(all_seq_distances)
    # Check archi diversi da formato standard
    df_valid = df[df["__points"].map(len) >= 2].copy()
    if df_valid.empty:
        raise ValueError("Nessun link valido (>=2 punti) trovato nel CSV.")

    df['__points'] = df['__points'].apply(lambda x: json.dumps(x))
    df['__distances'] = df['__distances'].apply(lambda x: np.array(x))
    df['__distances'] = df['__distances'].apply(lambda x: json.dumps(x.tolist()))
    path_to_save = os.path.join(params.project_path, 'data', params.dataset_name, f'traffic/processed_{params.dataset_name}_traffic_graph.csv')
    df.to_csv(str(path_to_save), sep=';', quoting=csv.QUOTE_NONNUMERIC, index=False)

    # Visualize original network
    if visualize_map or save_map:
        # --- Creazione mappa -------------------------------------------------------
        # Centro della mappa
        all_pts = [pt for pts in df_valid["__points"] for pt in pts]
        center_lat = sum(p[0] for p in all_pts) / len(all_pts)
        center_lon = sum(p[1] for p in all_pts) / len(all_pts)
        m = folium.Map(location=[center_lat, center_lon],
                       zoom_start=zoom_start,
                       tiles=None)
        if usa_satellite:
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                      "World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles © Esri — Source: Esri, i‑cubed, USDA, USGS, AEX, GeoEye, "
                     "Getmapping, Aerogrid, IGN, IGP, UPR‑EGP, GIS User Community",
                name="Esri World Imagery", overlay=False, control=True
            ).add_to(m)
        else:
            folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

        # --- Plot dei link ---------------------------------------------------------
        rng = random.Random(seed_colori)
        for _, row in df_valid.iterrows():
            pts = row["__points"]
            lid = row["id"]
            colore = _colore_random(rng)
            popup_txt = f"id: {lid}" if mostra_popup_id else None

            folium.PolyLine(
                locations=pts,
                color=colore,
                weight=weight,
                opacity=opacity,
                popup=popup_txt
            ).add_to(m)

            if mostra_nodi:
                for (lat, lon), tag in [(pts[0], f"start {lid}"), (pts[-1], f"end {lid}")]:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4, color=colore, fill=True, fill_opacity=1,
                        popup=tag if mostra_popup_id else None
                    ).add_to(m)

            if mostra_label:
                mid = pts[len(pts) // 2]
                folium.map.Marker(
                    location=[mid[0], mid[1]],
                    icon=DivIcon(
                        icon_size=(0, 0), icon_anchor=(0, 0),
                        html=f'<div style="font-size:10pt;color:{colore};'
                             f'text-shadow:1px 1px 2px white;">{lid}</div>'
                    )
                ).add_to(m)

        # --- Plot delle EV stations (opzionale) ------------------------------------
        if show_ev:
            if not ev_csv:
                raise ValueError("Per mostrare le EV stations devi passare `ev_csv`.")
            ev_list = process_newyork_ev_stations(ev_csv)
            ev_group = folium.FeatureGroup(name="EV Stations").add_to(m)
            for lat, lon, ev_id in ev_list:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="#0000FF",
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"EV station {ev_id}"
                ).add_to(ev_group)

        folium.LayerControl().add_to(m)
        out_path = Path(file_html)
        m.save(out_path)
        print(f"Mappa salvata in '{out_path}'")
    return m




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
        preprocessed_map_ev_node_traffic_node = os.path.join(self.params.preprocessed_data_path,
                                                         'map_ev_node_traffic_node.pt')
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
                os.path.exists(preprocessed_added_edges_df) and
                os.path.exists(preprocessed_map_ev_node_traffic_node)):
            self.final_temporal_merged_data = torch.load(preprocessed_filepath_merged_data, weights_only=False)
            self.time_column = torch.load(preprocessed_filepath_time_column, weights_only=False)
            self.edge_index_traffic = torch.load(preprocessed_edge_index_traffic, weights_only=False)
            self.edge_weights_traffic = torch.load(preprocessed_edge_weights_traffic, weights_only=False)
            self.map_ev_node_traffic_node = torch.load(preprocessed_map_ev_node_traffic_node, weights_only=False)

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

            # # Clean data of -1 values by substituting channel and node mean values
            # self.final_temporal_merged_data = clean_tensor(self.final_temporal_merged_data)

            # Save or load
            torch.save(self.final_temporal_merged_data, preprocessed_filepath_merged_data)
            torch.save(self.time_column, preprocessed_filepath_time_column)
            torch.save(self.edge_index_traffic, preprocessed_edge_index_traffic)
            torch.save(self.edge_weights_traffic, preprocessed_edge_weights_traffic)
            torch.save(self.map_ev_node_traffic_node, preprocessed_map_ev_node_traffic_node)

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

    # def load_preprocessed_data(self):
    #     """
    #     Load preprocessed data after temporal alignment and resolution resampling
    #     """
    #     load_tensor = torch.load(os.path.join(self.params.preprocessed_dataset_path,
    #                                           f"{self.params.dataset_name}{self.params.default_save_tensor_name}.pt"))
    #     data_tensor_traffic = load_tensor["data_tensor_traffic"]
    #     data_tensor_ev = load_tensor["data_tensor_ev"]
    #     return data_tensor_traffic, data_tensor_ev

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

    def assign_ev_node_to_traffic_node(self,
                                       col = "LocID"):
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
                ev_coordinates.append((lat, lng))


        # Map each EV node to nearest traffic node
        self.map_ev_node_traffic_node = {}
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
            self.map_ev_node_traffic_node[ev_node_idx] = int(min_dist_traffic_node_idx)
            cont_ev += 1

        # Assign the combined temporal ev data (self.data_tensor_ev) to temporal traffic data (self.data_tensor_traffic)
        # according to self.map_ev_node_traffic_node: create a list of list with len = num_of_traffic_nodes
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
        # Clean data of -1 values by substituting channel and node mean values
        self.final_temporal_merged_data = clean_tensor(self.final_temporal_merged_data)

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

        if  self.params.visualize_data:
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


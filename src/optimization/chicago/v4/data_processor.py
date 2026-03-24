import os
import pickle
from collections import defaultdict
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from config import ChargingStation


class ChicagoDataProcessor:

    def __init__(self, data_path: str = './'):
        self.data_path = data_path
        self.graph = nx.DiGraph()
        self.stations = {}
        self.speed_data = {}
        self.availability_data = {}
        self.speed_timestamps = []
        self.availability_timestamps = []
        self.node_coords = {}
        self.edge_to_site = {}
        self.site_to_edge = {}
        self.added_edge_ids = set()

    def _cache_path(self) -> str:
        return os.path.join(self.data_path, 'processor_cache.pkl')

    def _save_cache(self):
        with open(self._cache_path(), 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"  Cache saved to {self._cache_path()}")

    def _load_cache(self) -> bool:
        path = self._cache_path()
        if not os.path.exists(path):
            return False
        print(f"\nLoading from cache: {path}")
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        self._print_stats("Cached")
        return True

    def load(self, filter_largest_component=True, use_cache=True):
        if use_cache and self._load_cache():
            return (self.graph, self.stations, self.speed_data, self.availability_data,
                    self.speed_timestamps, self.availability_timestamps)

        print("\nLoading Chicago road network...")
        self._load_network()

        print("\nLoading predictions...")
        self._load_speed_predictions()
        self._load_availability_predictions()

        self._print_stats("Initial")

        if filter_largest_component:
            self._filter_to_largest_component()

        if use_cache:
            self._save_cache()

        print("\nData loading complete!")
        return (self.graph, self.stations, self.speed_data, self.availability_data,
                self.speed_timestamps, self.availability_timestamps)

    def _load_network(self, snap_tolerance_m: float = 35.0):
        edges_path = f'{self.data_path}/edges_df.csv'
        nodes_path = f'{self.data_path}/nodes_df.csv'
        added_edges_path = f'{self.data_path}/added_edges_df.csv'

        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"Cannot find {edges_path}")

        nodes_df = pd.read_csv(nodes_path) if os.path.exists(nodes_path) else None
        edges_df = self._load_edges(edges_path, added_edges_path)

        if nodes_df is not None and snap_tolerance_m > 0:
            nodes_df, edges_df = self._snap_nearby_nodes(nodes_df, edges_df, snap_tolerance_m)

        if nodes_df is not None:
            self._load_nodes_from_df(nodes_df)
        else:
            self._extract_nodes_from_edges(edges_df)

        self._add_edges_to_graph(edges_df)

    def _load_nodes_from_df(self, nodes_df: pd.DataFrame):
        print(f"  Nodes: {len(nodes_df)}")
        for _, row in nodes_df.iterrows():
            node_id = int(row['node_id'])
            lat, lon = float(row['lat']), float(row['lon'])
            self.graph.add_node(node_id, node_id=node_id, lat=lat, lon=lon)
            self.node_coords[node_id] = (lat, lon)

    def _snap_nearby_nodes(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                           tolerance_m: float = 25.0) -> tuple:
        coords = nodes_df[['lat', 'lon']].values
        node_ids = nodes_df['node_id'].values
        n = len(node_ids)

        coords_m = np.column_stack([coords[:, 0] * 111_000.0, coords[:, 1] * 82_600.0])

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                dlat = coords_m[i, 0] - coords_m[j, 0]
                dlon = coords_m[i, 1] - coords_m[j, 1]
                if abs(dlat) < tolerance_m and abs(dlon) < tolerance_m:
                    if dlat * dlat + dlon * dlon < tolerance_m * tolerance_m:
                        union(i, j)

        clusters = defaultdict(list)
        for idx in range(n):
            clusters[find(idx)].append(idx)

        print(f"  Node snapping (tol={tolerance_m}m): {n} → {len(clusters)} nodes "
              f"({n - len(clusters)} merged)")

        node_id_to_rep = {}
        new_node_rows = []
        for members in clusters.values():
            member_ids = node_ids[members]
            rep_id = int(member_ids.min())
            new_node_rows.append({
                'node_id': rep_id,
                'lat': float(coords[members, 0].mean()),
                'lon': float(coords[members, 1].mean()),
            })
            for mid in member_ids:
                node_id_to_rep[int(mid)] = rep_id

        new_edges_df = edges_df.copy()
        new_edges_df['src_id'] = new_edges_df['src_id'].map(node_id_to_rep)
        new_edges_df['tgt_id'] = new_edges_df['tgt_id'].map(node_id_to_rep)

        self_loops = (new_edges_df['src_id'] == new_edges_df['tgt_id']).sum()
        new_edges_df = new_edges_df[new_edges_df['src_id'] != new_edges_df['tgt_id']]

        new_edges_df = (new_edges_df
                        .sort_values(['src_id', 'tgt_id', 'id'])
                        .drop_duplicates(subset=['src_id', 'tgt_id'], keep='first')
                        .reset_index(drop=True))

        print(f"  Edges after snapping: {len(edges_df)} → {len(new_edges_df)} "
              f"({self_loops} self-loops removed)")

        return pd.DataFrame(new_node_rows), new_edges_df

    def _load_edges(self, edges_path: str, added_edges_path: str) -> pd.DataFrame:
        edges_df = pd.read_csv(edges_path)
        edges_df['is_added'] = False
        print(f"  Main edges: {len(edges_df)}")

        if os.path.exists(added_edges_path):
            added_df = pd.read_csv(added_edges_path)
            added_df['is_added'] = True
            self.added_edge_ids = set(added_df['id'].tolist())
            edges_df = pd.concat([edges_df, added_df], ignore_index=True)
            print(f"  Added edges: {len(added_df)}")

        return edges_df

    def _extract_nodes_from_edges(self, edges_df: pd.DataFrame):
        unique_nodes = set(edges_df['src_id'].tolist() + edges_df['tgt_id'].tolist())
        for node_id in unique_nodes:
            self.graph.add_node(int(node_id), node_id=int(node_id), lat=None, lon=None)
        print(f"  Extracted {len(unique_nodes)} nodes from edges")

    def _add_edges_to_graph(self, edges_df: pd.DataFrame):
        for _, row in edges_df.iterrows():
            src = int(row['src_id'])
            tgt = int(row['tgt_id'])
            edge_id = int(row['id'])
            distance = float(row['distance'])
            is_added = row.get('is_added', False)
            site_id = f"site_{edge_id + 1}"

            edge_attrs = {
                'site_id': site_id,
                'edge_id': edge_id,
                'distance': distance,
                'distance_km': distance,
                'is_added': is_added,
            }

            self.graph.add_edge(src, tgt, **edge_attrs)
            self.graph.add_edge(tgt, src, **edge_attrs)

            self.edge_to_site[(src, tgt)] = site_id
            self.edge_to_site[(tgt, src)] = site_id
            self.site_to_edge[site_id] = (src, tgt)

            self._create_station(site_id, src, tgt)

    def _create_station(self, site_id: str, src: int, tgt: int):
        src_coords = self.node_coords.get(src, (0, 0))
        self.stations[site_id] = ChargingStation(
            station_id=site_id,
            name=f"Station {site_id}",
            latitude=src_coords[0],
            longitude=src_coords[1],
            src_node=src,
            tgt_node=tgt,
        )

    def _load_speed_predictions(self):
        path = f'{self.data_path}/predictions_speed.csv'
        if not os.path.exists(path):
            print("  Speed predictions not found")
            return

        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        site_cols = [c for c in df.columns if c.startswith('site_')]
        print(f"  Speed data: {len(df)} timestamps, {len(site_cols)} sites")

        self.speed_timestamps = df['timestamp'].tolist()
        for site in site_cols:
            self.speed_data[site] = dict(zip(df['timestamp'], df[site].fillna(30.0)))

    def _load_availability_predictions(self):
        path = f'{self.data_path}/predictions_Available.csv'
        if not os.path.exists(path):
            print("  Availability predictions not found")
            return

        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        site_cols = [c for c in df.columns if c.startswith('site_')]
        print(f"  Availability data: {len(df)} timestamps, {len(site_cols)} sites")

        self.availability_timestamps = df['timestamp'].tolist()
        for site in site_cols:
            self.availability_data[site] = dict(zip(df['timestamp'], df[site].fillna(5.0)))

    def _filter_to_largest_component(self):
        print("\nFiltering to largest component...")
        components = list(nx.weakly_connected_components(self.graph))

        if not components:
            return

        largest = max(components, key=len)
        self.graph = self.graph.subgraph(largest).copy()

        self.node_coords = {n: c for n, c in self.node_coords.items() if n in largest}
        self.stations = {
            sid: s for sid, s in self.stations.items()
            if s.src_node in largest and s.tgt_node in largest
        }
        self.speed_data = {k: v for k, v in self.speed_data.items() if k in self.stations}
        self.availability_data = {k: v for k, v in self.availability_data.items() if k in self.stations}
        self.edge_to_site = {e: s for e, s in self.edge_to_site.items() if e[0] in largest and e[1] in largest}
        self.site_to_edge = {s: e for s, e in self.site_to_edge.items() if s in self.stations}

        self._print_stats("Filtered")

    def _print_stats(self, prefix=""):
        print(f"{prefix} stats:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Stations: {len(self.stations)}")
        print(f"  Speed data: {len(self.speed_data)} sites")
        print(f"  Availability data: {len(self.availability_data)} sites")

    def get_station(self, site_id: str) -> Optional[ChargingStation]:
        return self.stations.get(site_id)

    def get_speed(self, site_id: str, timestamp) -> float:
        if site_id in self.speed_data and timestamp in self.speed_data[site_id]:
            return self.speed_data[site_id][timestamp]
        return 30.0

    def get_availability(self, site_id: str, timestamp) -> float:
        if site_id in self.availability_data and timestamp in self.availability_data[site_id]:
            return self.availability_data[site_id][timestamp]
        return 5.0

    def get_nearest_timestamp(self, target_time, data_type='speed'):
        timestamps = self.speed_timestamps if data_type == 'speed' else self.availability_timestamps
        if not timestamps:
            return None
        return min(timestamps, key=lambda t: abs((t - target_time).total_seconds()))

    def get_statistics(self) -> Dict:
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'stations': len(self.stations),
            'speed_sites': len(self.speed_data),
            'availability_sites': len(self.availability_data),
        }

        if self.speed_data:
            all_speeds = [v for site in self.speed_data.values() for v in site.values()]
            if all_speeds:
                stats['avg_speed'] = np.mean(all_speeds)

        if self.availability_data:
            all_avails = [v for site in self.availability_data.values() for v in site.values()]
            if all_avails:
                stats['avg_availability'] = np.mean(all_avails)

        return stats

import os
import pickle
import pandas as pd
import networkx as nx
from typing import List, Tuple

from config import ChargingStation


class NewYorkDataProcessor:

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

    def _cache_path(self):
        return os.path.join(self.data_path, 'processor_cache.pkl')

    def _save_cache(self):
        cache = {
            'graph': self.graph,
            'stations': self.stations,
            'speed_data': self.speed_data,
            'availability_data': self.availability_data,
            'speed_timestamps': self.speed_timestamps,
            'availability_timestamps': self.availability_timestamps,
            'node_coords': self.node_coords,
            'edge_to_site': self.edge_to_site,
            'site_to_edge': self.site_to_edge,
        }
        with open(self._cache_path(), 'wb') as f:
            pickle.dump(cache, f)

    def _load_cache(self):
        with open(self._cache_path(), 'rb') as f:
            cache = pickle.load(f)
        self.graph = cache['graph']
        self.stations = cache['stations']
        self.speed_data = cache['speed_data']
        self.availability_data = cache['availability_data']
        self.speed_timestamps = cache['speed_timestamps']
        self.availability_timestamps = cache['availability_timestamps']
        self.node_coords = cache.get('node_coords', {})
        self.edge_to_site = cache.get('edge_to_site', {})
        self.site_to_edge = cache.get('site_to_edge', {})
        print(f"Cached stats:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Stations: {len(self.stations)}")
        print(f"  Speed data: {len(self.speed_data)} sites")
        print(f"  Availability data: {len(self.availability_data)} sites")

    def load(self, use_cache=True, filter_largest_component=True):
        cache_file = self._cache_path()
        if use_cache and os.path.exists(cache_file):
            print(f"\nLoading from cache: {cache_file}")
            self._load_cache()
            return (self.graph, self.stations, self.speed_data, self.availability_data,
                    self.speed_timestamps, self.availability_timestamps)

        print("\nLoading New York road network...")
        self._load_network()

        print("\nLoading predictions...")
        self._load_speed_predictions()
        self._load_availability_predictions()

        self._print_stats("Initial")

        if filter_largest_component:
            self._filter_to_largest_component()

        print("\nData loading complete!")
        if use_cache:
            self._save_cache()
        return (self.graph, self.stations, self.speed_data, self.availability_data,
                self.speed_timestamps, self.availability_timestamps)

    def _load_network(self):
        edges_path = f'{self.data_path}/edges_df.csv'
        nodes_path = f'{self.data_path}/nodes_df.csv'
        added_edges_path = f'{self.data_path}/added_edges_df.csv'

        if not os.path.exists(edges_path):
            raise FileNotFoundError(f"Cannot find {edges_path}")

        if os.path.exists(nodes_path):
            self._load_nodes(nodes_path)

        edges_df = self._load_edges(edges_path, added_edges_path)

        if self.graph.number_of_nodes() == 0:
            self._extract_nodes_from_edges(edges_df)

        self._add_edges_to_graph(edges_df)

    def _load_nodes(self, path):
        nodes_df = pd.read_csv(path)
        print(f"  Nodes: {len(nodes_df)}")

        for _, row in nodes_df.iterrows():
            node_id = int(row['node_id'])
            lat, lon = float(row['lat']), float(row['lon'])

            self.graph.add_node(node_id, node_id=node_id, lat=lat, lon=lon)
            self.node_coords[node_id] = (lat, lon)

    def _load_edges(self, edges_path, added_edges_path):
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

    def _extract_nodes_from_edges(self, edges_df):
        unique_nodes = set(edges_df['src_id'].tolist() + edges_df['tgt_id'].tolist())
        for node_id in unique_nodes:
            self.graph.add_node(int(node_id), node_id=int(node_id), lat=None, lon=None)
        print(f"  Extracted {len(unique_nodes)} nodes from edges")

    def _add_edges_to_graph(self, edges_df):
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
                'is_added': is_added
            }

            self.graph.add_edge(src, tgt, **edge_attrs)
            self.graph.add_edge(tgt, src, **edge_attrs)

            self.edge_to_site[(src, tgt)] = site_id
            self.edge_to_site[(tgt, src)] = site_id
            self.site_to_edge[site_id] = (src, tgt)

            self._create_station(site_id, src, tgt)

    def _create_station(self, site_id, src, tgt):
        src_coords = self.node_coords.get(src, (0, 0))

        station = ChargingStation(
            station_id=site_id,
            name=f"Station {site_id}",
            latitude=src_coords[0],
            longitude=src_coords[1],
            src_node=src,
            tgt_node=tgt
        )
        self.stations[site_id] = station

    def _load_speed_predictions(self):
        path = f'{self.data_path}/predictions_speed.csv'
        if not os.path.exists(path):
            print(f"  Speed predictions not found")
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
            print(f"  Availability predictions not found")
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


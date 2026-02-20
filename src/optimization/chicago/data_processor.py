import os
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import ChargingStation


class ChicagoDataProcessor:

    def __init__(self, data_path: str = './'):
        self.data_path = data_path
        self.graph = nx.DiGraph()
        self.stations = {}
        self.speed_data = {}
        self.availability_data = {}
        self.node_coords = {}
        self.edge_to_site = {}
        self.site_to_edge = {}
        self.added_edge_ids = set()

    def load(self, filter_largest_component=True):
        print("\nLoading Chicago road network...")
        self._load_network()

        print("\nLoading predictions...")
        self._load_speed_predictions()
        self._load_availability_predictions()

        self._print_stats("Initial")

        if filter_largest_component:
            self._filter_to_largest_component()

        print("\nData loading complete!")
        return self.graph, self.stations, self.speed_data, self.availability_data

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

            self.edge_to_site[(src, tgt)] = site_id
            self.site_to_edge[site_id] = (src, tgt)

            self._create_station(site_id, src, tgt, edge_id, distance)

    def _create_station(self, site_id, src, tgt, edge_id, distance):
        src_coords = self.node_coords.get(src, (0, 0))
        tgt_coords = self.node_coords.get(tgt, (0, 0))

        lat = (src_coords[0] + tgt_coords[0]) / 2
        lon = (src_coords[1] + tgt_coords[1]) / 2

        station = ChargingStation(
            station_id=site_id,
            name=f"Station {site_id}",
            latitude=lat,
            longitude=lon,
            src_node=src,
            tgt_node=tgt,
            edge_id=edge_id,
            edge_distance_km=distance,
            nearest_road_node=src
        )
        self.stations[site_id] = station

    def _load_speed_predictions(self):
        path = f'{self.data_path}/predictions_speed.csv'
        if not os.path.exists(path):
            print(f"  Speed predictions not found")
            return

        df = pd.read_csv(path)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

        site_cols = [c for c in df.columns if c.startswith('site_')]
        print(f"  Speed data: {len(df)} timestamps, {len(site_cols)} sites")

        for site in site_cols:
            hourly = {}
            for hour in range(24):
                values = df[df['hour'] == hour][site].dropna()
                hourly[hour] = float(values.mean()) if len(values) > 0 else 30.0

            self.speed_data[site] = hourly
            if site in self.stations:
                self.stations[site].hourly_speeds = hourly

    def _load_availability_predictions(self):
        path = f'{self.data_path}/predictions_Available.csv'
        if not os.path.exists(path):
            print(f"  Availability predictions not found")
            return

        df = pd.read_csv(path)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

        site_cols = [c for c in df.columns if c.startswith('site_')]
        print(f"  Availability data: {len(df)} timestamps, {len(site_cols)} sites")

        for site in site_cols:
            hourly = {}
            for hour in range(24):
                values = df[df['hour'] == hour][site].dropna()
                hourly[hour] = float(values.mean()) if len(values) > 0 else 5.0

            self.availability_data[site] = hourly
            if site in self.stations:
                self.stations[site].hourly_availability = hourly

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

    def get_speed(self, site_id: str, hour: int) -> float:
        if site_id in self.speed_data:
            return self.speed_data[site_id].get(hour, 30.0)
        return 30.0

    def get_availability(self, site_id: str, hour: int) -> float:
        if site_id in self.availability_data:
            return self.availability_data[site_id].get(hour, 5.0)
        return 5.0

    def get_statistics(self) -> Dict:
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'stations': len(self.stations),
            'speed_sites': len(self.speed_data),
            'availability_sites': len(self.availability_data),
        }

        if self.stations:
            speeds = [s.get_average_speed() for s in self.stations.values()]
            avails = [s.get_average_availability() for s in self.stations.values()]

            stats['avg_speed'] = np.mean(speeds)
            stats['avg_availability'] = np.mean(avails)

        return stats

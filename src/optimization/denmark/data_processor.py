import os
import math
import random
import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import ChargingStation


class DenmarkDataProcessor:

    def __init__(self, data_path: str = './data/', traffic_path: str = None):
        self.data_path = data_path
        self.traffic_path = traffic_path or os.path.join(data_path, 'traffic_feb_june')
        self.graph = nx.DiGraph()
        self.stations: Dict[str, ChargingStation] = {}
        self.speed_data: Dict[str, Dict] = {}
        self.availability_data: Dict = {}
        self.speed_timestamps: List = []
        self.availability_timestamps: List = []
        self.node_coords: Dict[str, Tuple[float, float]] = {}
        self.edge_to_site: Dict[Tuple, str] = {}
        self.site_to_edge: Dict[str, Tuple] = {}

    def load(self, filter_largest_component: bool = True):
        print("\nLoading Denmark road network...")
        self._load_network()

        print("\nLoading EV charging stations...")
        self._load_ev_stations()

        print("\nLoading traffic speed data...")
        self._load_speed_data()

        self._print_stats("Initial")

        if filter_largest_component:
            self._filter_to_largest_component()

        print("\nData loading complete!")
        return (self.graph, self.stations, self.speed_data,
                self.availability_data, self.speed_timestamps,
                self.availability_timestamps)

    def get_station(self, site_id: str) -> Optional[ChargingStation]:
        return self.stations.get(site_id)

    def get_speed(self, site_id: str, timestamp) -> float:
        if site_id in self.speed_data and timestamp in self.speed_data[site_id]:
            return self.speed_data[site_id][timestamp]
        return 50.0

    def get_availability(self, site_id: str, timestamp=None) -> float:
        return float(random.randint(0, 5))

    def get_nearest_timestamp(self, target_time, data_type: str = 'speed'):
        if not self.speed_timestamps:
            return None
        return min(self.speed_timestamps,
                   key=lambda t: abs((t - target_time).total_seconds()))

    def get_statistics(self) -> Dict:
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'stations': len(self.stations),
            'speed_sites': len(self.speed_data),
        }
        if self.speed_data:
            all_speeds = [v for site in self.speed_data.values() for v in site.values()]
            if all_speeds:
                stats['avg_speed_kmh'] = float(np.mean(all_speeds))
        return stats

    def _load_network(self):
        metadata_path = os.path.join(self.data_path, 'trafficMetaData.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Cannot find {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        print(f"  Metadata rows: {len(metadata)}")

        nodes: Dict[str, Tuple[float, float]] = {}
        for i in range(len(metadata)):
            p1 = str(metadata['POINT_1_NAME'][i])
            p2 = str(metadata['POINT_2_NAME'][i])
            nodes[p1] = (float(metadata['POINT_1_LAT'][i]), float(metadata['POINT_1_LNG'][i]))
            nodes[p2] = (float(metadata['POINT_2_LAT'][i]), float(metadata['POINT_2_LNG'][i]))

        for name, (lat, lng) in nodes.items():
            self.graph.add_node(name, lat=lat, lon=lng, pos=(lat, lng))
            self.node_coords[name] = (lat, lng)

        for i in range(len(metadata)):
            src = str(metadata['POINT_1_NAME'][i])
            tgt = str(metadata['POINT_2_NAME'][i])
            length_m = int(metadata['DISTANCE_IN_METERS'][i])
            speed_limit_kmh = int(metadata['NDT_IN_KMH'][i])
            report_id = str(metadata['REPORT_ID'][i])

            src_street = str(metadata['POINT_1_STREET'][i]).strip()
            tgt_street = str(metadata['POINT_2_STREET'][i]).strip()
            is_turn = 1 if src_street != tgt_street else 0

            distance_km = length_m / 1000.0
            site_id = f"site_{report_id}"

            self.graph.add_edge(src, tgt,
                                site_id=site_id,
                                report_id=report_id,
                                length=length_m,
                                distance_km=distance_km,
                                distance=distance_km,
                                speed_limit_kmh=speed_limit_kmh,
                                isTurn=is_turn)
            self.edge_to_site[(src, tgt)] = site_id
            self.site_to_edge[site_id] = (src, tgt)

        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")

    def _load_ev_stations(self):
        stations_path = os.path.join(self.data_path, 'charging_stations.csv')
        if not os.path.exists(stations_path):
            print("  charging_stations.csv not found - skipping")
            return

        df = pd.read_csv(stations_path)
        graph_nodes = list(self.node_coords.keys())

        power_cols = [
            'CEE_RED_power_max', 'CHADEMO_power_max', 'COMBO_TYPE_2_power_max',
            'DOMESTIC_TYPE_F_power_max', 'MENNEKES_TYPE_2_power_max',
            'MENNEKES_TYPE_2_CABLE_ATTACHED_power_max', 'TESLA_COMBO_CCS_power_max',
            'TESLA_SUPERCHARGER_EU_power_max'
        ]

        for _, row in df.iterrows():
            lat = row['lat']
            lon = row['lng']

            nearest_node = self._find_nearest_node(lat, lon, graph_nodes)
            if nearest_node is None:
                continue

            station_id = f"dk_station_{int(row['id'])}"
            name = f"{row['street_name']}, {row['city']}"

            powers = [row[c] for c in power_cols if c in df.columns and pd.notna(row[c])]
            power_kw = float(max(powers)) if powers else 50.0

            self.stations[station_id] = ChargingStation(
                station_id=station_id,
                name=name,
                latitude=lat,
                longitude=lon,
                src_node=nearest_node,
                tgt_node=nearest_node,
                charging_power_kw=power_kw,
            )

        print(f"  Loaded {len(self.stations)} EV stations")

    def _find_nearest_node(self, lat: float, lon: float, nodes: List[str]) -> Optional[str]:
        min_dist = float('inf')
        nearest = None
        for node in nodes:
            coords = self.node_coords.get(node)
            if coords is None:
                continue
            dist = math.sqrt((coords[0] - lat) ** 2 + (coords[1] - lon) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def _load_speed_data(self, max_timestamps_per_file: int = 50):
        traffic_dir = self.traffic_path
        if not os.path.exists(traffic_dir):
            print("  Traffic data directory not found")
            return

        all_timestamps: set = set()

        for filename in sorted(os.listdir(traffic_dir)):
            if not filename.endswith('.csv'):
                continue

            report_id = filename.replace('trafficData', '').replace('.csv', '')
            site_id = f"site_{report_id}"

            if site_id not in self.site_to_edge:
                continue

            src, tgt = self.site_to_edge[site_id]
            speed_limit_kmh = self.graph.edges[src, tgt].get('speed_limit_kmh', 50)

            try:
                df = pd.read_csv(os.path.join(traffic_dir, filename),
                                 nrows=max_timestamps_per_file)
            except Exception:
                continue

            df['timestamp'] = pd.to_datetime(df['TIMESTAMP'])

            site_speeds: Dict = {}
            for _, row in df.iterrows():
                ts = row['timestamp']
                avg_speed_kmh = float(row['avgSpeed']) * 1.60934
                site_speeds[ts] = avg_speed_kmh if avg_speed_kmh >= 5.0 else float(speed_limit_kmh)
                all_timestamps.add(ts)

            self.speed_data[site_id] = site_speeds

        self.speed_timestamps = sorted(list(all_timestamps))
        print(f"  Speed data: {len(self.speed_data)} sites, {len(self.speed_timestamps)} timestamps")

    def _filter_to_largest_component(self):
        print("\nFiltering to largest weakly connected component...")
        components = list(nx.weakly_connected_components(self.graph))
        if not components:
            return

        largest = max(components, key=len)
        self.graph = self.graph.subgraph(largest).copy()

        self.node_coords = {n: c for n, c in self.node_coords.items() if n in largest}
        self.stations = {sid: s for sid, s in self.stations.items() if s.src_node in largest}

        active_site_ids = {self.edge_to_site[e] for e in self.graph.edges() if e in self.edge_to_site}
        self.speed_data = {k: v for k, v in self.speed_data.items() if k in active_site_ids}

        self.edge_to_site = {e: s for e, s in self.edge_to_site.items()
                             if e[0] in largest and e[1] in largest}
        self.site_to_edge = {s: e for s, e in self.site_to_edge.items()
                             if e[0] in largest and e[1] in largest}

        self._print_stats("Filtered")

    def _print_stats(self, prefix: str = ""):
        print(f"{prefix} stats:")
        print(f"  Nodes: {self.graph.number_of_nodes()}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        print(f"  Stations: {len(self.stations)}")
        print(f"  Speed data: {len(self.speed_data)} sites")
        print(f"  Availability: random (0-5) generated on demand")

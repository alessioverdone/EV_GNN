import bisect
import networkx as nx
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config import EVConfig, ChargingStation, PathObjectives


class EVRoutePlanner:

    def __init__(self, graph: nx.DiGraph, stations: Dict, speed_data: Dict,
                 availability_data: Dict, ev_config: EVConfig,
                 speed_timestamps: List = None, availability_timestamps: List = None):
        self.graph = graph
        self.ev_stations = stations
        self.traffic_data = speed_data
        self.availability_data = availability_data
        self.ev_config = ev_config

        self.speed_timestamps = sorted(speed_timestamps) if speed_timestamps else []
        self.availability_timestamps = sorted(availability_timestamps) if availability_timestamps else []

        self.node_to_stations = {}
        for station in stations.values():
            if station.src_node:
                self.node_to_stations.setdefault(station.src_node, []).append(station)

    def has_charging(self, node) -> bool:
        return node in self.node_to_stations

    def get_stations(self, node) -> List[ChargingStation]:
        return self.node_to_stations.get(node, [])

    def is_charging_location(self, node) -> bool:
        return self.has_charging(node)

    def get_stations_at_node(self, node) -> List[ChargingStation]:
        return self.get_stations(node)

    def get_energy_consumption(self, path: List) -> float:
        total = 0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i + 1]):
                dist = self.graph.edges[path[i], path[i + 1]].get('distance_km', 0)
                total += dist * self.ev_config.consumption_kwh_per_km
        return total

    def get_charging_stops(self, path: List, departure_time: datetime = None) -> List:
        if len(path) < 2:
            return []

        if departure_time is None:
            departure_time = datetime.now().replace(hour=8, minute=0, second=0)

        stops = []
        soc = self.ev_config.current_soc
        current_time = departure_time
        cfg = self.ev_config

        for i in range(len(path) - 1):
            curr, next_node = path[i], path[i + 1]

            if not self.graph.has_edge(curr, next_node):
                continue

            edge = self.graph.edges[curr, next_node]
            dist = edge.get('distance_km', 0)
            site_id = edge.get('site_id', '')

            speed = self._get_speed(site_id, current_time)
            travel_hours = dist / speed if speed > 0 else dist / 30.0

            soc -= (dist * cfg.consumption_kwh_per_km) / cfg.battery_capacity_kwh

            if self.has_charging(next_node) and soc < cfg.charge_trigger_soc:
                stations = self.get_stations(next_node)
                if stations:
                    energy_needed = (cfg.target_charge_soc - soc) * cfg.battery_capacity_kwh

                    if energy_needed >= cfg.min_charge_amount_kwh:
                        stops.append(next_node)
                        charge_hours = energy_needed / (stations[0].charging_power_kw * cfg.charging_efficiency)
                        soc = cfg.target_charge_soc
                        current_time += timedelta(hours=travel_hours + charge_hours)
                        continue

            current_time += timedelta(hours=travel_hours)

        return stops

    def get_charging_stops_in_path(self, path: List, departure_time: datetime = None) -> List:
        return self.get_charging_stops(path, departure_time)

    def evaluate(self, path: List, departure_time: datetime = None) -> PathObjectives:
        if len(path) < 2:
            return self._invalid_result()

        if departure_time is None:
            departure_time = datetime.now().replace(hour=8, minute=0, second=0)

        total_dist = 0
        total_travel = 0
        total_charge = 0
        num_stops = 0

        soc = self.ev_config.current_soc
        current_time = departure_time
        cfg = self.ev_config

        for i in range(len(path) - 1):
            curr, next_node = path[i], path[i + 1]

            if not self.graph.has_edge(curr, next_node):
                return self._invalid_result()

            edge = self.graph.edges[curr, next_node]
            dist = edge.get('distance_km', 0)
            site_id = edge.get('site_id', '')

            speed = self._get_speed(site_id, current_time)
            travel_hours = dist / speed if speed > 0 else dist / 30.0
            travel_min = travel_hours * 60

            total_dist += dist
            total_travel += travel_min

            soc -= (dist * cfg.consumption_kwh_per_km) / cfg.battery_capacity_kwh

            if self.has_charging(next_node) and soc < cfg.charge_trigger_soc:
                stations = self.get_stations(next_node)
                if stations:
                    energy_needed = (cfg.target_charge_soc - soc) * cfg.battery_capacity_kwh

                    if energy_needed >= cfg.min_charge_amount_kwh:
                        charge_hours = energy_needed / (stations[0].charging_power_kw * cfg.charging_efficiency)
                        charge_min = charge_hours * 60

                        total_charge += charge_min
                        num_stops += 1
                        soc = cfg.target_charge_soc
                        current_time += timedelta(hours=travel_hours + charge_hours)
                        continue

            current_time += timedelta(hours=travel_hours)

        return PathObjectives(
            distance_km=total_dist,
            travel_time_min=total_travel,
            charging_time_min=total_charge,
            num_charging_stops=num_stops,
            final_soc=max(0, soc)
        )

    def find_path(self, start, end, max_detour: float = 1.3) -> List:
        try:
            direct = nx.shortest_path(self.graph, start, end, weight='distance_km')
            direct_dist = nx.shortest_path_length(self.graph, start, end, weight='distance_km')
        except nx.NetworkXNoPath:
            return []

        energy = self.get_energy_consumption(direct)
        available = self.ev_config.current_soc * self.ev_config.battery_capacity_kwh
        reserve = self.ev_config.min_soc_threshold * self.ev_config.battery_capacity_kwh

        if energy <= (available - reserve):
            return direct

        best_path = direct
        best_score = float('inf')

        for node in self.node_to_stations.keys():
            try:
                path1 = nx.shortest_path(self.graph, start, node, weight='distance_km')
                path2 = nx.shortest_path(self.graph, node, end, weight='distance_km')
                full = path1 + path2[1:]

                dist = sum(
                    self.graph.edges[full[i], full[i + 1]]['distance_km']
                    for i in range(len(full) - 1)
                    if self.graph.has_edge(full[i], full[i + 1])
                )

                if dist <= direct_dist * max_detour:
                    obj = self.evaluate(full)
                    score = obj.total_time_min

                    if score < best_score:
                        best_score = score
                        best_path = full

            except nx.NetworkXNoPath:
                continue

        return best_path

    def _get_speed(self, site_id: str, current_time: datetime) -> float:
        if site_id not in self.traffic_data:
            return 30.0

        site_speeds = self.traffic_data[site_id]
        if not site_speeds:
            return 30.0

        nearest_ts = self._find_nearest_timestamp(current_time, self.speed_timestamps)
        if nearest_ts is None or nearest_ts not in site_speeds:
            return 30.0

        speed_mph = site_speeds[nearest_ts]
        if speed_mph < 10:
            return 30.0

        return speed_mph * 1.60934

    def _get_availability(self, site_id: str, current_time: datetime) -> float:
        if site_id not in self.availability_data:
            return 5.0

        site_avail = self.availability_data[site_id]
        if not site_avail:
            return 5.0

        nearest_ts = self._find_nearest_timestamp(current_time, self.availability_timestamps)
        if nearest_ts is None or nearest_ts not in site_avail:
            return 5.0

        return site_avail[nearest_ts]

    def _find_nearest_timestamp(self, target: datetime, timestamps: List) -> Optional[datetime]:
        if not timestamps:
            return None

        idx = bisect.bisect_left(timestamps, target)

        if idx == 0:
            return timestamps[0]
        if idx == len(timestamps):
            return timestamps[-1]

        before = timestamps[idx - 1]
        after = timestamps[idx]

        if (target - before).total_seconds() <= (after - target).total_seconds():
            return before
        return after

    def _invalid_result(self) -> PathObjectives:
        return PathObjectives(
            distance_km=float('inf'),
            travel_time_min=float('inf'),
            charging_time_min=0,
            num_charging_stops=0,
            final_soc=0
        )

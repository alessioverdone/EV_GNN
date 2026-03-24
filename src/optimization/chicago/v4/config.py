import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class EVConfig:
    battery_capacity_kwh: float = 75.0
    current_soc: float = 0.4
    consumption_kwh_per_km: float = 0.15
    min_soc_threshold: float = 0.2
    charging_efficiency: float = 0.95
    charge_trigger_soc: float = 0.35
    target_charge_soc: float = 0.8
    min_charge_amount_kwh: float = 5.0


@dataclass
class ChargingStation:
    station_id: str
    name: str
    latitude: float
    longitude: float
    src_node: int = None
    tgt_node: int = None
    charging_power_kw: float = 50.0


SELECTED_ROUTE_INDEX = 2

PREDEFINED_ROUTES = [
    {"start": 854, "end": 893, "distance_km": 52.8893,
     "start_coords": (42.011599, -87.728599), "end_coords": (41.659189, -87.559286)},
    {"start": 855, "end": 893, "distance_km": 51.2800,
     "start_coords": (42.011857, -87.709055), "end_coords": (41.659189, -87.559286)},
    {"start": 858, "end": 893, "distance_km": 50.5236,
     "start_coords": (42.012114, -87.699745), "end_coords": (41.659189, -87.559286)},
    {"start": 854, "end": 890, "distance_km": 50.4271,
     "start_coords": (42.011599, -87.728599), "end_coords": (41.659956, -87.588804)},
    {"start": 855, "end": 890, "distance_km": 48.8177,
     "start_coords": (42.011857, -87.709055), "end_coords": (41.659956, -87.588804)},
]


@dataclass
class PathObjectives:
    distance_km: float
    travel_time_min: float
    charging_time_min: float
    num_charging_stops: int
    final_soc: float

    def as_array(self) -> np.ndarray:
        return np.array([
            self.distance_km,
            self.travel_time_min + self.charging_time_min,
        ])

    @property
    def total_time_min(self) -> float:
        return self.travel_time_min + self.charging_time_min

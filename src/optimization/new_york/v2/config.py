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


SELECTED_ROUTE_INDEX = 1

PREDEFINED_ROUTES = [
    {"start": 58,  "end": 181, "distance_km": 126.00,
     "start_coords": (40.744221, -73.771661), "end_coords": (40.621191, -74.100000)},
    {"start": 159, "end": 181, "distance_km": 125.36,
     "start_coords": (40.763800, -73.726090), "end_coords": (40.621191, -74.100000)},
    {"start": 158, "end": 181, "distance_km": 124.04,
     "start_coords": (40.750840, -73.750230), "end_coords": (40.621191, -74.100000)},
    {"start": 59,  "end": 181, "distance_km": 123.45,
     "start_coords": (40.753750, -73.744391), "end_coords": (40.621191, -74.100000)},
    {"start": 38,  "end": 181, "distance_km": 122.73,
     "start_coords": (40.748770, -73.738950), "end_coords": (40.621191, -74.100000)},
]

import numpy as np
from dataclasses import dataclass, field
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
    nearest_road_node: int = None
    distance_to_road_km: float = 0.0
    src_node: int = None
    tgt_node: int = None
    edge_id: int = None
    edge_distance_km: float = 0.0
    charging_power_kw: float = 50.0
    hourly_speeds: Dict[int, float] = field(default_factory=dict)
    hourly_availability: Dict[int, float] = field(default_factory=dict)

    def get_speed_at_hour(self, hour: int) -> float:
        return self.hourly_speeds.get(hour, 30.0)

    def get_availability_at_hour(self, hour: int) -> float:
        return self.hourly_availability.get(hour, 0.0)

    def get_average_speed(self) -> float:
        if not self.hourly_speeds:
            return 30.0
        return np.mean(list(self.hourly_speeds.values()))

    def get_average_availability(self) -> float:
        if not self.hourly_availability:
            return 5.0
        return np.mean(list(self.hourly_availability.values()))


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

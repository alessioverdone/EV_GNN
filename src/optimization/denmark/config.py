import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class EVConfig:
    battery_capacity_kwh: float = 75.0
    current_soc: float = 0.4
    consumption_kwh_per_km: float = 0.25
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

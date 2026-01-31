from __future__ import annotations


class KDTemperatureSchedule:
    def __init__(self, config, total_updates: int):
        self.config = config
        self._total_updates = total_updates

    def kd_T_at(self, update_idx: int) -> float:
        total_updates = self._total_updates
        T0 = float(getattr(self.config, "kd_temperature_start", 2.0))
        T1 = float(getattr(self.config, "kd_temperature_end", 1.0))
        hold_frac = float(getattr(self.config, "kd_hold_frac", 0.6))
        hold_u = int(hold_frac * total_updates)
        if update_idx <= hold_u:
            return T0
        tail = max(1, (total_updates - hold_u))
        pos = (update_idx - hold_u) / tail
        return T0 + (T1 - T0) * pos

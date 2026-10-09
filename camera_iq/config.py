"""Every tunable threshold lives here."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    mtf50_drop: float = 0.10    # flag if MTF50 falls by more than this fraction of the baseline's
    delta_e_rise: float = 1.5   # flag if mean CIEDE2000 rises by more than this (absolute)


DEFAULT = Thresholds()

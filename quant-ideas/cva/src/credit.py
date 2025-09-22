from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class CreditCurve:
    """Flat hazard curve for unilateral CVA."""
    hazard_rate: float  # continuous intensity

    def survival(self, t: float) -> float:
        return math.exp(-self.hazard_rate * t)

    def dPD(self, times: List[float]) -> np.ndarray:
        """Default probability increments on [t_{i-1}, t_i]."""
        if not times:
            return np.array([])
        out = []
        S_prev = self.survival(0.0)
        for ti in times:
            S_i = self.survival(ti)
            out.append(S_prev - S_i)
            S_prev = S_i
        return np.array(out, dtype=float)


def build_credit_curve(flat_hazard: float = 0.015) -> CreditCurve:
    return CreditCurve(hazard_rate=flat_hazard)

## person 2 code

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal


import numpy as np

OptionType = Literal["call, put"]


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    n_sims: int
    antithetic: bool


def _validate_inputs(S0: float, K: float, r: float, sigma: float, T: float, n_sims: int) -> None:
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be > 0.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if T <= 0:
        raise ValueError("T must be > 0.")
    if n_sims < 10:
        raise ValueError("n_sims must be >= 10.")

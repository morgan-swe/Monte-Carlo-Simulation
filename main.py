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
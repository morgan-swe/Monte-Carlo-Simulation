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


def discounted_payoffs_gbm(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int = 200_000,
    option_type: OptionType = "call",
    seed: Optional[int] = 42,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Returns an array of discounted payoffs under risk-neutral GBM.

    Risk-neutral terminal stock:
        S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z), Z~N(0,1)

    If antithetic=True, we generate n_half = ceil(n_sims/2) normals Z and pair with -Z.
    The returned array length will be:
        - n_sims (approximately): exactly 2*n_half, then trimmed to n_sims.
    """
    _validate_inputs(S0, K, r, sigma, T, n_sims)

    rng = np.random.default_rng(seed)
    disc = math.exp(-r * T)

    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)

    if not antithetic:
        Z = rng.standard_normal(n_sims)
        ST = S0 * np.exp(drift + vol * Z)
    else:
        n_half = (n_sims + 1) // 2  # ceil(n_sims/2)
        Z = rng.standard_normal(n_half)
        Z_pair = np.concatenate([Z, -Z])  # antithetic pairing
        Z_pair = Z_pair[:n_sims]          # trim if n_sims is odd
        ST = S0 * np.exp(drift + vol * Z_pair)

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    return disc * payoff

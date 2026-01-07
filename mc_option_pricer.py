## person 2 code
# mc_option_pricer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

# types
OptionType = Literal["call", "put"]


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


def discounted_payoffs_and_disc_ST_gbm(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int = 200_000,
    option_type: OptionType = "call",
    seed: Optional[int] = 42,
    antithetic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      discounted_payoffs = e^{-rT} * payoff(S_T)
      discounted_ST      = e^{-rT} * S_T

    Why:
      - discounted_payoffs is what Person 3 needs for convergence plots (running mean)
      - discounted_ST is for control variate (E[discounted_ST] = S0 if no dividends)
    """
    _validate_inputs(S0, K, r, sigma, T, n_sims)

    # RNG w/ seed so the results are repeatable (same seed = same outputs)
    rng = np.random.default_rng(seed)

    # discount factor
    disc = math.exp(-r * T)

    # GBM pieces
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)

    # generate normals (antithetic = pair Z and -Z)
    if not antithetic:
        Z = rng.standard_normal(n_sims)
    else:
        n_half = (n_sims + 1) // 2  # ceil(n_sims/2)
        Z_half = rng.standard_normal(n_half)
        Z = np.concatenate([Z_half, -Z_half])[:n_sims]

    # terminal price under risk-neutral GBM
    ST = S0 * np.exp(drift + vol * Z)

    # payoff at expiry
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    discounted_payoffs = disc * payoff
    discounted_ST = disc * ST

    return discounted_payoffs, discounted_ST


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
    Just give back discounted payoffs.
    (Under the hood we call the "both" function so we don't duplicate logic.)
    """
    discounted_payoffs, _ = discounted_payoffs_and_disc_ST_gbm(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_sims=n_sims,
        option_type=option_type,
        seed=seed,
        antithetic=antithetic,
    )
    return discounted_payoffs


def mc_european_option_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int = 200_000,
    option_type: OptionType = "call",
    seed: Optional[int] = 42,
    antithetic: bool = False,
) -> MCResult:
    """
    Plain Monte Carlo price for a European call/put under GBM.
    Returns price + stderr + 95% confidence interval.
    """
    discounted = discounted_payoffs_gbm(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_sims=n_sims,
        option_type=option_type,
        seed=seed,
        antithetic=antithetic,
    )

    price = float(np.mean(discounted))

    # stats (stderr = std/sqrt(N))
    sample_std = float(np.std(discounted, ddof=1))
    stderr = sample_std / math.sqrt(len(discounted))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr

    return MCResult(price, stderr, ci_low, ci_high, len(discounted), antithetic)


def mc_european_option_price_control_variate(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_sims: int = 200_000,
    option_type: OptionType = "call",
    seed: Optional[int] = 42,
    antithetic: bool = False,
) -> MCResult:
    """
    Monte Carlo w/ control variate (variance reduction).

    Control:
      Y = discounted_ST = e^{-rT} * S_T
      E[Y] = S0 (this assumes no dividends)

    Estimator:
      X_cv = X - b (Y - E[Y])
      b = Cov(X,Y) / Var(Y)
    """
    X, Y = discounted_payoffs_and_disc_ST_gbm(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_sims=n_sims,
        option_type=option_type,
        seed=seed,
        antithetic=antithetic,
    )

    EY = S0  # if we add dividend yield q later, this becomes S0 * exp(-qT)

    # faster than np.cov (same idea tho)
    Yc = Y - Y.mean()
    var_y = float(np.mean(Yc * Yc))

    if var_y == 0.0:
        # edge case: if Y is basically constant (ex: sigma=0), control variate can't help
        X_cv = X
    else:
        Xc = X - X.mean()
        cov_xy = float(np.mean(Xc * Yc))
        b = cov_xy / var_y
        X_cv = X - b * (Y - EY)

    price = float(np.mean(X_cv))

    sample_std = float(np.std(X_cv, ddof=1))
    stderr = sample_std / math.sqrt(len(X_cv))
    ci_low = price - 1.96 * stderr
    ci_high = price + 1.96 * stderr

    return MCResult(price, stderr, ci_low, ci_high, len(X_cv), antithetic)


def put_call_parity_gap(call_price: float, put_price: float, S0: float, K: float, r: float, T: float) -> float:
    """
    Put-call parity (no dividends):
      C - P = S0 - K e^{-rT}

    This returns the "gap" (LHS - RHS). Should be close to 0 if everything is working.
    """
    rhs = S0 - K * math.exp(-r * T)
    return (call_price - put_price) - rhs


def quick_demo() -> None:
    # quick demo params (same as the pdf-style defaults)
    S0 = 100.0
    K = 100.0
    r = 0.025
    sigma = 0.20
    T = 21 / 252
    n_sims = 200_000

    # 1) plain MC
    res_plain = mc_european_option_price(
        S0, K, r, sigma, T,
        n_sims=n_sims,
        option_type="call",
        seed=42,
        antithetic=False,
    )

    # 2) antithetic MC
    res_anti = mc_european_option_price(
        S0, K, r, sigma, T,
        n_sims=n_sims,
        option_type="call",
        seed=42,
        antithetic=True,
    )

    # 3) control variate + antithetic (usually best CI)
    res_cv = mc_european_option_price_control_variate(
        S0, K, r, sigma, T,
        n_sims=n_sims,
        option_type="call",
        seed=42,
        antithetic=True,
    )

    print("Monte Carlo European Call (GBM)")
    print(f"Plain MC ({res_plain.n_sims:,} sims)")
    print(f"  price:  {res_plain.price:.6f}")
    print(f"  stderr: {res_plain.stderr:.6f}")
    print(f"  95% CI: [{res_plain.ci_low:.6f}, {res_plain.ci_high:.6f}]")
    print()
    print(f"Antithetic MC ({res_anti.n_sims:,} sims)")
    print(f"  price:  {res_anti.price:.6f}")
    print(f"  stderr: {res_anti.stderr:.6f}")
    print(f"  95% CI: [{res_anti.ci_low:.6f}, {res_anti.ci_high:.6f}]")
    print()
    print(f"Control Variate + Antithetic ({res_cv.n_sims:,} sims)")
    print(f"  price:  {res_cv.price:.6f}")
    print(f"  stderr: {res_cv.stderr:.6f}")
    print(f"  95% CI: [{res_cv.ci_low:.6f}, {res_cv.ci_high:.6f}]")

    # put-call parity sanity check (no dividends assumption)
    call_cv = mc_european_option_price_control_variate(
        S0, K, r, sigma, T,
        n_sims=n_sims,
        option_type="call",
        seed=7,
        antithetic=True,
    )
    put_cv = mc_european_option_price_control_variate(
        S0, K, r, sigma, T,
        n_sims=n_sims,
        option_type="put",
        seed=7,
        antithetic=True,
    )
    gap = put_call_parity_gap(call_cv.price, put_cv.price, S0, K, r, T)

    print()
    print("Put-call parity gap (should be near 0):", f"{gap:.6e}")

    # for Person 3 convergence:
    # disc_payoffs = discounted_payoffs_gbm(..., antithetic=True)
    # running_mean = np.cumsum(disc_payoffs) / np.arange(1, len(disc_payoffs) + 1)


if __name__ == "__main__":
    quick_demo()

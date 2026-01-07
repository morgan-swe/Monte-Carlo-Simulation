# Monte Carlo Engine (Person 2) — ENGINE.md

## What this is
This file explains the Monte Carlo “engine” I implemented for:
- Pricing European call/put options under GBM
- Computing **tail risk metrics** (5th percentile return / VaR-style)
- Producing arrays for **convergence plots** and diagnostics

If you’re Person 3, this tells you exactly what functions to call for plots, tables, and sanity checks.

---

## Model + Inputs (what the code assumes)

We simulate terminal stock price under **risk-neutral GBM**:

S_T = S0 * exp((r - 0.5*sigma^2)T + sigma*sqrt(T) * Z),  Z ~ N(0,1)

### Inputs
- `S0`: initial stock price  
- `K`: strike price  
- `r`: risk-free rate (annual)  
- `sigma`: volatility (annual)  
- `T`: time to maturity in years (ex: `21/252`)  
- `n_sims`: number of simulations  
- `option_type`: `"call"` or `"put"`  

### Assumptions
- European option (exercise only at maturity)
- Constant `r` and `sigma`
- No dividends (important for control variate expectation)
- Risk-neutral pricing

---
/Users/geff/Downloads/ENGINE_updated.md
## Reproducibility (why results repeat)

By default the engine uses a fixed seed (`seed=42`) with NumPy’s RNG.  
That means if you rerun with the same seed, you should get the **exact same output**.

If you want different randomness, change the seed or pass `seed=None`.

---

## What functions exist + what to use

### 1) Plain pricing (baseline)

Use this when you want the basic Monte Carlo estimate + CI.

```python
mc_european_option_price(
    S0, K, r, sigma, T,
    n_sims=200_000,
    option_type="call",
    seed=42,
    antithetic=False
)
```

Returns an `MCResult`:
- `price`: estimated option price  
- `stderr`: standard error  
- `ci_low`, `ci_high`: 95% confidence interval  
- `n_sims`: number of simulations used  
- `antithetic`: whether antithetic was enabled  

---

### 2) Antithetic variates (variance reduction)

Same function, just set:

```python
antithetic=True
```

This pairs `Z` and `-Z` to reduce variance.

---

### 3) Control variate (best accuracy)

Use this when you want tighter confidence intervals.

```python
mc_european_option_price_control_variate(
    S0, K, r, sigma, T,
    n_sims=200_000,
    option_type="call",
    seed=42,
    antithetic=True
)
```

Control variate uses:

- Y = exp(-rT) * S_T  
- E[Y] = S0 (no dividends)

---

### 4) Convergence plotting (Person 3)

If you want a convergence plot, use:

```python
payoffs = discounted_payoffs_gbm(
    S0, K, r, sigma, T,
    n_sims=200_000,
    option_type="call",
    seed=42,
    antithetic=True
)

running_mean = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)
```

This is the fastest way to get a convergence curve without rerunning the pricer N times.

---

### 5) For control variate + extra diagnostics

```python
payoffs, disc_ST = discounted_payoffs_and_disc_ST_gbm(...)
```

- `payoffs` = discounted payoffs  
- `disc_ST` = discounted terminal stock (used for control variate)  

---

## 6) Risk-style metrics (5th percentile return + return summary)

I also added helpers to compute **tail returns** from the same GBM simulation.

Important note (keep it real):
- These returns are generated under **risk-neutral GBM** (drift uses `r`), not real-world expected returns.
- Still useful as a **VaR-style tail metric** / distribution summary under the model assumptions.

### A) Simulate terminal prices (base function)

If you just want the raw simulated terminal prices:

```python
ST = simulate_terminal_prices_gbm(
    S0, r, sigma, T,
    n_sims=200_000,
    seed=42,
    antithetic=True
)
```

Returns an array of `S_T`.

---

### B) 5th percentile return (what the handout asks for)

Return definition:

R = (S_T - S0) / S0

Call:

```python
p05 = fifth_percentile_return_gbm(
    S0, r, sigma, T,
    n_sims=200_000,
    seed=42,
    antithetic=True
)
```

This returns a single float like `-0.0723` (≈ `-7.23%`).

---

### C) Full return distribution summary (for tables / report)

```python
risk = risk_summary_returns_gbm(
    S0, r, sigma, T,
    n_sims=200_000,
    seed=42,
    antithetic=True
)
```

Returns a `RiskResult` with:
- `p01`, `p05`, `median`, `mean`, `p95`, `p99`
- plus `n_sims` and whether antithetic was used

This is basically “here’s the distribution” in one object so Person 3 doesn’t have to recompute percentiles manually.

---

## Sanity checks I recommend (Person 3)

### Put–call parity (no dividends)

If you price call + put with the same params, parity should roughly hold:

C - P ≈ S0 - K * exp(-rT)

Use:

```python
gap = put_call_parity_gap(call_price, put_price, S0, K, r, T)
```

The gap should be **close to 0** relative to Monte Carlo noise.

---

## Common gotchas

- `T` must be in **years** (ex: `21/252`, not `21`)
- `sigma` should be annualized (ex: `0.2` for 20%)
- If you add dividends later, control variate expectation changes to:
  ```
  S0 * exp(-qT)
  ```

---

## Quick run (demo)

From the repo root, run:

```bash
python ../mc_option_pricer.py
```

It prints:
- Plain MC results
- Antithetic results
- Control variate results
- 5th percentile return (VaR-style)
- Return distribution summary
- Put–call parity gap

---

## How Person 3 should cite results in the writeup (suggestion)

- Report: `price ± 1.96 * stderr` as the 95% CI
- Mention if antithetic / control variate was used
- Mention `n_sims` and seed for reproducibility
- For risk section, report the **5th percentile return** and optionally the full summary table

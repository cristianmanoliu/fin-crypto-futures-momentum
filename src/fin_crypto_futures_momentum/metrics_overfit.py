"""Overfit filters: Deflated Sharpe Ratio and CSCV PBO
(Bailey & Lopez de Prado). stdlib NormalDist — no scipy."""
import itertools
import math
from statistics import NormalDist

import numpy as np

_ND = NormalDist()
_EULER = 0.5772156649015329


def sharpe_moments(returns: np.ndarray) -> tuple[float, float, float]:
    """(per-period Sharpe, skewness, Pearson kurtosis) of a return series.
    Sharpe uses sample std (ddof=1); moment ratios use population std
    (scipy convention)."""
    r = np.asarray(returns, dtype=float)
    mu, sd = float(np.mean(r)), float(np.std(r, ddof=1))
    sr = mu / sd
    z = (r - mu) / np.std(r)          # population sd for moment ratios
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    return sr, skew, kurt


def deflated_sharpe_ratio(
    sr_hat: float, t_obs: int, skew: float, kurt: float,
    trial_sharpes: list[float], *, n_trials: int | None = None,
) -> float:
    """P(true SR > 0) deflated for multiple testing."""
    if n_trials is not None and n_trials < 2:
        raise ValueError("n_trials must be >= 2")
    n = n_trials if n_trials is not None else len(trial_sharpes)
    var = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1)) if n > 1 else 0.0
    if var > 0.0:
        sr0 = math.sqrt(var) * (
            (1 - _EULER) * _ND.inv_cdf(1 - 1 / n)
            + _EULER * _ND.inv_cdf(1 - 1 / (n * math.e))
        )
    else:
        sr0 = 0.0
    denom = math.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
    return _ND.cdf((sr_hat - sr0) * math.sqrt(t_obs - 1) / denom)


def pbo_cscv(returns_matrix: np.ndarray, s_blocks: int = 16) -> float:
    """Probability of Backtest Overfitting via combinatorially symmetric
    cross-validation."""
    if s_blocks % 2 != 0:
        raise ValueError("s_blocks must be even")
    m = np.asarray(returns_matrix, dtype=float)
    t, n = m.shape
    blocks = np.array_split(np.arange(t), s_blocks)

    def block_sharpes(rows: np.ndarray) -> np.ndarray:
        sub = m[rows, :]
        sd = sub.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, sub.mean(axis=0) / sd, -np.inf)

    below_median = 0
    combos = list(itertools.combinations(range(s_blocks), s_blocks // 2))
    for combo in combos:
        in_combo = set(combo)
        train_rows = np.concatenate([blocks[i] for i in combo])
        test_rows = np.concatenate(
            [blocks[i] for i in range(s_blocks) if i not in in_combo])
        best = int(np.argmax(block_sharpes(train_rows)))
        test_sr = block_sharpes(test_rows)
        rank = float(np.sum(test_sr < test_sr[best]))
        omega = (rank + 1) / (n + 1)
        if omega < 0.5:
            below_median += 1
    return below_median / len(combos)

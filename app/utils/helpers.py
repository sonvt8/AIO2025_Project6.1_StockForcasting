"""
Helper functions for data processing
"""

import numpy as np


def last_window(a, w):
    """Get last w elements from array"""
    if len(a) >= w:
        return a[-w:]
    return a


def rolling_std(a, w):
    """Calculate rolling standard deviation"""
    win = last_window(a, w)
    if len(win) <= 1:
        return 0.0
    return float(np.std(win, ddof=1))


def rolling_mean(a, w):
    """Calculate rolling mean"""
    win = last_window(a, w)
    if len(win) == 0:
        return 0.0
    return float(np.mean(win))


def returns_to_prices(last_price: float, future_returns: np.ndarray) -> np.ndarray:
    """Convert returns to prices"""
    prices = []
    p = float(last_price)
    for r in future_returns:
        p = p * np.exp(r)
        prices.append(p)
    return np.array(prices, dtype=float)


def _rsi_from_window(prices_window: np.ndarray, period: int) -> float:
    """Calculate RSI from price window"""
    delta = np.diff(prices_window)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)

"""Deterministic next-open cross-sectional evaluation."""
from dataclasses import dataclass
from math import ceil, sqrt
from statistics import fmean

from .signal import ranks, score


@dataclass(frozen=True)
class Window:
    name: str
    start: int
    end: int  # exclusive, includes room for the final trade exit


def split(panel):
    size = len(panel.dates)
    first, second = size * 3 // 5, size * 4 // 5
    return Window("research", 0, first), Window("validation", first, second), Window("holdout", second, size)


def rank_ic(scores, returns):
    if len(set(scores.values())) < 2 or len(set(returns.values())) < 2:
        return None
    ranked_returns = ranks(returns)
    x_bar, y_bar = fmean(scores.values()), fmean(ranked_returns.values())
    pairs = [(scores[s] - x_bar, ranked_returns[s] - y_bar) for s in scores]
    return sum(x * y for x, y in pairs) / sqrt(sum(x * x for x, _ in pairs) * sum(y * y for _, y in pairs))


def _weights(scores, symbols):
    weights = dict.fromkeys(symbols, 0.0)
    if len(scores) < 6 or len(set(scores.values())) < 2:
        return weights
    ordered = sorted(scores, key=lambda s: (scores[s], s))
    count = len(ordered) // 3
    # Fractional allocation at ties avoids a ticker-based artificial preference.
    for side, cutoff in ((-1, scores[ordered[count - 1]]), (1, scores[ordered[-count]])):
        strict = [s for s in ordered if side * (scores[s] - cutoff) > 0]
        tied = [s for s in ordered if scores[s] == cutoff]
        for symbol in strict:
            weights[symbol] += side * 0.5 / count
        for symbol in tied:
            weights[symbol] += side * 0.5 * (count - len(strict)) / (count * len(tied))
    return weights


def evaluate(panel, tree, window, cost_bps, *, rebalance_every=1):
    """Close t score. Enter open t+1 and exit or rebalance open t+2."""
    symbols = panel.symbols
    drifted = dict.fromkeys(symbols, 0.0)
    gross, net, turnover, exposure, daily_ic, coverage = [], [], [], [], [], []
    for session in range(window.start, window.end - 2):
        scores = score(panel, tree, session)
        coverage.append(len(scores) / len(symbols))
        target = _weights(scores, symbols)
        # Carry marked holdings between scheduled rebalances. Exit on unusable scores.
        weights = target if (session - window.start) % rebalance_every == 0 or not any(target.values()) else drifted
        returns = {s: panel.forward_return(session, s) for s in symbols}
        if len(scores) >= 6:
            ic = rank_ic(scores, {s: returns[s] for s in scores})
            if ic is not None:
                daily_ic.append(ic)
        traded = sum(abs(weights[s] - drifted[s]) for s in symbols)
        pnl = sum(weights[s] * returns[s] for s in symbols)
        # Fixed reference notional. Carry marked positions into the next rebalance.
        drifted = {s: weights[s] * (1 + returns[s]) for s in symbols}
        if session == window.end - 3:
            traded += sum(abs(weight) for weight in drifted.values())
        exposure.append(sum(abs(weight) for weight in weights.values()))
        gross.append(pnl)
        turnover.append(traded)
        net.append(pnl - traded * cost_bps / 10_000)
    days = len(net)
    return {
        "start": panel.dates[window.start].isoformat(),
        "last_decision": panel.dates[window.end - 3].isoformat(),
        "last_exit": panel.dates[window.end - 1].isoformat(),
        "days": days,
        "invested_days": sum(item > 0 for item in exposure),
        "rank_ic_days": len(daily_ic),
        "rank_ic": round(fmean(daily_ic), 6) if daily_ic else None,
        "mean_coverage": round(fmean(coverage), 6),
        "mean_turnover": round(fmean(turnover), 6),
        "gross_bps_per_day": round(10_000 * fmean(gross), 6),
        "net_bps_per_day": round(10_000 * fmean(net), 6),
        "stress_net_bps_per_day": round(10_000 * (fmean(gross) - 2 * fmean(turnover) * cost_bps / 10_000), 6),
        "minimum_ic_days": ceil(days * 0.8),
    }

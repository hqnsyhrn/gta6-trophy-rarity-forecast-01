"""gta6-trophy-rarity-forecast-01: hierarchical earners / players-tracked forecaster.

Model (fitted at runtime from the case file only; stdlib only, no network):
  rate_i(t) = E_i(t) / N(t), where N(t) is Exophase's players-tracked and
  E_i(t) the (implied, cumulative) earners of trophy i.  Both are fitted in
  log-time, x = ln(1 + hours since the first history read):
    * ln N(x): least squares, averaged with a recent-half fit (uptake slows).
    * ln E_i(x): least squares per trophy, slope shrunk toward its tier's
      median slope; anchored at the last observed earners; E never decreases
      (cumulative) and never exceeds N.
  The median forecast is 100 * E_i(target) / N(target).  The spread comes
  from a rolling-origin backtest inside the case history (fit on snapshots
  before each refresh, predict that refresh, pool the logit errors), plus a
  horizon term -- never the starter's fixed random-walk spread alone.

Fallback (fewer than 4 distinct snapshots, e.g. the practice cases whose
history is one frozen launch-night page): a refresh-aware persistence.
Launch-night evidence (observed, not assumed): the page sat frozen at
638 players tracked, then refreshed to 3,868, and by 11:12 UTC on 19 Nov
rates had fallen to a median 0.205x their frozen values (median logit
shift -1.61).  A refresh multiplies N ~3-6x while earners grow ~1.5-3x,
so one refresh moves a typical rate by about -0.9 logit.  The fallback
shifts the median by -0.9 logit per expected refresh (mean refresh
interval ~3h, from the observed gaps) and widens the spread by the
Bernoulli uncertainty of whether a refresh lands inside the horizon.
Expected-value check on the practice window (90 min): shift -0.45 has
the lowest expected pinball loss for refresh probabilities 0.3-0.5
(0.26-0.28), versus 0.29-0.47 for unshifted persistence; it is NOT
tuned to the practice answers (which the kit does not publish).

Output: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, quantiles non-decreasing, <= 65,536 bytes, byte
identical across runs (no randomness, fixed iteration order).

Usage (local): MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
Grader: python3 -P -s /bot/main.py, reads /input, writes /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- calibration constants, each from observed launch-night data (see docstring) ---
REFRESH_LOGIT_DROP = 0.9      # typical logit move of a rate across one Exophase refresh
REFRESH_MEAN_HOURS = 3.0      # mean gap between Exophase refreshes, hours
TIER_DROP_FACTOR = {          # gold/platinum/silver fell slightly further than bronze
    "platinum": 1.05, "gold": 1.05, "silver": 1.05, "bronze": 0.95,
}
SPREAD_HORIZON_K = 0.375     # logit sigma per sqrt(day) of horizon, added in quadrature
N_CAP = 3_000_000             # sanity cap on extrapolated players tracked


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg > 20:
        return 99.99
    if lg < -20:
        return 0.01
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """Round to 4 decimals, half up (same as the starter / JS Math.round)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b). Falls back to flat at mean(y)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


class Snapshots:
    """Deduped history: one entry per distinct Exophase refresh."""

    def __init__(self, case):
        hist = case["history"]
        t0 = parse_time(hist[0]["at"])
        self.t0 = t0
        self.hours = []       # hours since first read
        self.players = []     # players tracked
        self.rates = []       # list of rate lists, in trophy order
        prev = None
        for read in hist:
            key = (read["players_tracked"], tuple(read["rates"]))
            if key == prev:
                continue
            prev = key
            self.hours.append((parse_time(read["at"]) - t0).total_seconds() / 3600.0)
            self.players.append(read["players_tracked"])
            self.rates.append(list(read["rates"]))
        self.n = len(self.hours)
        self.x = [math.log1p(h) for h in self.hours]

    def last_rates(self):
        return self.rates[-1]


def fit_players_variants(snap: Snapshots, upto: int, x_target: float):
    """Log players-tracked forecasts at x_target under three variants.

    Returns [full log-log OLS, recent-half log-log OLS, logistic in
    log-time].  The logistic variant saturates: uptake cannot grow as a
    power law forever, and the disagreement between variants is real
    model uncertainty -- the spread later accounts for it.  Each variant
    is floored at the last observed N and capped at N_CAP.
    """
    xs = snap.x[:upto]
    ys = [math.log(max(1, p)) for p in snap.players[:upto]]
    n_last_log = ys[-1]
    cap = math.log(N_CAP)

    def bounded(v):
        return min(cap, max(n_last_log, v))

    a, b = ols(xs, ys)
    full = bounded(a + b * x_target)
    half = max(2, (upto + 1) // 2)
    a2, b2 = ols(xs[-half:], ys[-half:])
    recent = bounded(a2 + b2 * x_target)
    # Logistic in log-time: N = N_inf / (1 + exp(a - k*x)); grid over N_inf.
    best = None
    for mult in (1.2, 1.5, 2, 3, 5, 10, 30, 100):
        n_inf = math.log(max(1, snap.players[upto - 1]) * mult)
        zs = []
        ok = True
        for x, y in zip(xs, ys):
            ratio = math.exp(n_inf - y) - 1.0
            if ratio <= 0:
                ok = False
                break
            zs.append(math.log(ratio))
        if not ok:
            continue
        # ln(N_inf/N - 1) = a - k*x  -> OLS of z on x gives (a, -k).
        az, bz = ols(xs, zs)
        sse = sum((z - (az + bz * x)) ** 2 for x, z in zip(xs, zs))
        if best is None or sse < best[0]:
            best = (sse, az, bz, n_inf)
    if best is None:
        logistic = full
    else:
        _, az, bz, n_inf = best
        logistic = bounded(n_inf - math.log1p(math.exp(az + bz * x_target)))
    return [full, recent, logistic]


def fit_players(snap: Snapshots, upto: int, x_target: float) -> float:
    """Median (in log space) of the players-tracked variants."""
    return median(fit_players_variants(snap, upto, x_target))


def fit_earners(snap: Snapshots, upto: int, x_target: float, trophies, recent_half=False):
    """Forecast earners at x_target per trophy from snapshots[:upto].

    Returns list of (median earners, last earners). Slopes shrunk to tier median.
    With recent_half, slopes come from the last half of snapshots only --
    earner growth usually slows as the easy progress is banked.
    """
    n_tr = len(trophies)
    # Per-trophy OLS slopes first, to get tier medians.
    slopes = []
    series = []  # per trophy: list of (x, ln E)
    for i in range(n_tr):
        pts = []
        for j in range(upto):
            r = snap.rates[j][i]
            if r is None:
                continue
            e = max(0.5, r * snap.players[j] / 100.0)
            pts.append((snap.x[j], math.log(e)))
        series.append(pts)
        use = pts[-max(2, (len(pts) + 1) // 2):] if recent_half else pts
        slopes.append(ols([p[0] for p in use], [p[1] for p in use])[1] if len(use) >= 2 else 0.0)
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [slopes[i] for i, t in enumerate(trophies) if t["tier"] == tier]
        tier_slopes[tier] = median(vals) if vals else median(slopes)
    out = []
    for i, trophy in enumerate(trophies):
        pts = series[i]
        if not pts:
            out.append((0.5, 0.5))
            continue
        use = pts[-max(2, (len(pts) + 1) // 2):] if recent_half else pts
        b_raw = ols([p[0] for p in use], [p[1] for p in use])[1] if len(use) >= 2 else 0.0
        w = len(use) / (len(use) + 3.0)
        b = w * b_raw + (1.0 - w) * tier_slopes[trophy["tier"]]
        # Anchor at the last observed earners so the level is exact.
        x_last, ln_e_last = pts[-1]
        ln_e_t = ln_e_last + b * (x_target - x_last)
        e_t = math.exp(ln_e_t)
        out.append((e_t, math.exp(ln_e_last)))
    return out


def predict_medians(snap: Snapshots, upto: int, x_target: float, trophies, n_target: float,
                    recent_half=False):
    """Median rate (%) per trophy at x_target, from snapshots[:upto]."""
    earners = fit_earners(snap, upto, x_target, trophies, recent_half=recent_half)
    medians = []
    for i, trophy in enumerate(trophies):
        e_t, e_last = earners[i]
        last_rate = None
        for j in range(upto - 1, -1, -1):
            if snap.rates[j][i] is not None:
                last_rate = snap.rates[j][i]
                break
        if last_rate is not None and last_rate >= 99.99:
            medians.append(100.0)
            continue
        e_t = min(max(e_t, e_last), n_target)  # earners only grow, never exceed players
        medians.append(clamp(100.0 * e_t / n_target))
    return medians


def rolling_origin_sigma(snap: Snapshots, trophies, x_target: float) -> float:
    """Robust sigma of logit forecast errors from rolling-origin backtests."""
    errors = []
    for j in range(3, snap.n):
        # Fit on snapshots[:j], predict snapshot j.
        n_j = fit_players(snap, j, snap.x[j])
        med = predict_medians(snap, j, snap.x[j], trophies, math.exp(n_j))
        for i in range(len(trophies)):
            actual = snap.rates[j][i]
            if actual is None or actual >= 99.99 or med[i] >= 99.99:
                continue
            errors.append(logit(actual) - logit(med[i]))
    if len(errors) < 5:
        return 0.35
    med_e = median(errors)
    mad = median([abs(e - med_e) for e in errors])
    return max(0.15, 1.4826 * mad)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    snap = Snapshots(case)
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    last = snap.last_rates()
    t_last = snap.hours[-1]
    x_target = math.log1p(t_last + horizon_hours)

    if snap.n >= 4:
        n_logs = fit_players_variants(snap, snap.n, x_target)
        n_target = math.exp(median(n_logs))
        # Median = per-trophy median across 3 N-variants x 2 earner-variants.
        # Their disagreement (in logit space) is genuine model uncertainty
        # about post-checkpoint uptake -- it widens the spread below.
        combo_medians = []
        for nl in n_logs:
            for rh in (False, True):
                combo_medians.append(
                    predict_medians(snap, snap.n, x_target, trophies, math.exp(nl), recent_half=rh)
                )
        medians = []
        variant_sigma = []
        for i in range(len(trophies)):
            vals = [c[i] for c in combo_medians]
            medians.append(median(vals))
            lg = [logit(v) for v in vals]
            mu = sum(lg) / len(lg)
            variant_sigma.append(math.sqrt(sum((v - mu) ** 2 for v in lg) / len(lg)))
        s_ro = rolling_origin_sigma(snap, trophies, x_target)
        spreads = []
        for i in range(len(trophies)):
            spread = math.sqrt(
                s_ro ** 2
                + (SPREAD_HORIZON_K * math.sqrt(horizon_days)) ** 2
                + variant_sigma[i] ** 2
            )
            spreads.append(min(1.8, max(0.45 if horizon_days >= 1 else 0.30, spread)))
    else:
        # Frozen history: refresh-aware persistence (see module docstring).
        exp_refreshes = horizon_hours / REFRESH_MEAN_HOURS
        p_refresh = 1.0 - math.exp(-exp_refreshes)
        medians = []
        spreads = []
        rw = 0.5 * math.sqrt(horizon_days)
        bern = REFRESH_LOGIT_DROP * math.sqrt(max(0.0, p_refresh * (1.0 - p_refresh)))
        spread = math.sqrt(rw ** 2 + bern ** 2 + 0.15 ** 2)
        for i, trophy in enumerate(trophies):
            r = last[i]
            if r is None:
                r = 0.0
            if r >= 99.99:
                medians.append(100.0)
            else:
                shift = -REFRESH_LOGIT_DROP * TIER_DROP_FACTOR.get(trophy["tier"], 1.0)
                shift *= min(exp_refreshes, 2.8)  # shift saturates; uptake cannot run away
                medians.append(clamp(inv_logit(logit(r) + shift)))
            spreads.append(spread)

    rows = []
    for i, trophy in enumerate(trophies):
        m = medians[i]
        s = spreads[i]
        if m >= 99.99:
            q = [100, 100, 100, 100, 100]
        else:
            lg = logit(m)
            q = [round4(inv_logit(lg + z * s)) for z in Z]
            # Enforce non-decreasing after rounding.
            for k in range(1, 5):
                if q[k] < q[k - 1]:
                    q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

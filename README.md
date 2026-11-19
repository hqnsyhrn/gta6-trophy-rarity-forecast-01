# gta6-trophy-rarity-forecast-01

Forecast, for every GTA VI trophy on PS5, its earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles per trophy. Scored by mean pinball
loss on the logit scale; a prize needs a loss strictly below the best house
entry (persistence baseline, fitted-curve baseline, superbot max-reasoning
forecast). $250, top 20. Deadline 2026-11-21T11:30:00Z.

## The model (`main.py`, Python 3.13 stdlib only, no network)

`rate_i(t) = E_i(t) / N(t)` — earners over players-tracked — fitted at runtime
from the case file in log-time `x = ln(1 + hours since first read)`:

- **Players tracked N(t):** median of three variants — full log-log OLS,
  recent-half log-log OLS, and a logistic-in-log-time fit (uptake saturates).
- **Earners E_i(t):** per-trophy log-log OLS, slope shrunk toward the tier
  median, anchored at the last observed earners; earners never decrease and
  never exceed N. A recent-half slope variant is averaged in.
- **Spread:** rolling-origin backtest inside the case history (fit before each
  refresh, predict it, pool logit errors) + a horizon term + the
  disagreement between the model variants. Never the starter's fixed spread.
- **Frozen-history fallback** (fewer than 4 distinct snapshots, e.g. the
  practice cases): refresh-aware persistence. Launch-night evidence: the
  page sat frozen at 638 players, refreshed to 3,868, and by 11:12 UTC
  19 Nov rates had fallen to a median 0.205x (median logit shift −1.61).
  One refresh moves a typical rate ≈ −0.9 logit, so the median shifts by
  −0.9 per expected refresh (mean interval ~3 h) with a Bernoulli-widened
  spread.

## Validation (all run locally, numbers in `work/` — not part of the entry)

| Test | Ours | Persistence (1.51 practice) | Curve baseline |
|---|---|---|---|
| Practice p00/p01 vs **reconstructed** 04:30Z answer (interpolated earners, N=3,868 known; NOT the house answer) | **0.327** loss, 0.24pp miss, cov 1.00 | 0.924 | 0.924 (flat history) |
| Practice p00/p01 vs **live Exophase page 11:12Z 19 Nov** (exact, all 55 rates) | **0.838** loss, 0.37pp miss | 1.508 | 1.508 |
| Synthetic final, world A (log-log growth continues, N→600k) | **0.191** loss, cov 1.00 | 0.399 | 0.232 |
| Synthetic final, world B (N saturates at 250k, earners 2x slower — flattening invisible pre-deadline) | 0.461 loss, cov 1.00 | **0.244** | 0.559 |

Practice cases are frozen-history + one shock and prove almost nothing
about the final — the live-page backtest and the synthetic finals are the
real evidence. World B is the honest weak spot: if Exophase uptake
saturates hard right after the deadline, no model can see it coming from
pre-deadline data, and persistence wins there.

- Outputs: 55 rows, exact keys, 5 monotonic quantiles, ≤2,779 bytes.
- Deterministic: two runs byte-identical. Runtime ~0.1 s (practice),
  ~0.3 s (120-read synthetic final) — limit is 8 s per case.
- Edge cases pass: 1 history read, null rates, all-100% rates,
  6,510-minute horizon on frozen history.

## Honest grade

**~20–25% top-10** (top-20 pays; top-10 ≈ the realistic prize band).
The model demonstrably beats both house baselines on every real-data
comparison available (live backtest 0.84 vs 1.51) and on the
growth-continues synthetic final (0.19 vs 0.40/0.23), but: the bar is
beating a **superbot max-reasoning forecast** whose loss is unpublished
and unmeasurable in advance; the field will hold competent statistical
fitters; and in the saturation stress world the model loses to
persistence. Nothing here proves a top-20 finish — it only proves the
entry is a genuine, validated attempt rather than the starter.

## Status

**BUILD ONLY — NOT SUBMITTED.** This session is not a recorded submit
session: a clean re-record/bundle will be needed before any submit, per
the transcript gate tightening from 20 Nov 00:00 UTC (real system prompt
+ matching run record, min authored coverage 0.8).

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

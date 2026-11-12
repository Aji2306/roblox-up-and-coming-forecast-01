# Roblox Up-and-Coming Top-10 Forecaster

Entry for make.fun competition `roblox-up-and-coming-forecast-01`:
predict which games will hold each of the top 10 places of Roblox's
Up-and-Coming chart over the next three days, with honest probabilities.

## Method

Monte Carlo rank-diffusion forecaster, deterministic (seeded), Python 3.13
standard library only, no network, no model calls.

Each game carries a continuous chart position; over the 72 hourly slots it
evolves as a random walk with rank-dependent diffusion, momentum (recent rank
velocity, fading), and weak mean reversion toward the game's own typical rank.
Games leave the chart with a tenure-dependent hazard (28-day age-out) and new
games enter deep. Every simulation is re-ranked each hour; the forecast
probabilities are the time-averaged place occupancy over all simulations and
hours. Calibration (per-band diffusion, momentum weight, chart churn) is
recomputed from the practice history at run time.

## Run

```
python3 main.py
```

Reads `/input/history.ndjson` and `/input/task.json`, writes
`/output/forecast.json` with `probabilities` (place 1..10, per-game p, sums
<= 1, rest left unnamed) and `top10` (10 distinct universe ids in order).
Two runs on the same history write identical bytes. Runtime ~16s on the
practice history (limit 300s, 1 CPU, 1024 MB).

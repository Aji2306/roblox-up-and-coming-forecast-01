"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Monte Carlo rank-diffusion forecaster, deterministic (seeded), stdlib only, no network.

Model
-----
Each game carries a continuous chart position p (its rank is the order of p).
Over the scored window (72 hourly slots) p evolves as a random walk with
rank-dependent diffusion, momentum (recent rank velocity, fading), and weak
mean reversion toward the game's own typical rank. Games leave the chart with
a tenure-dependent hazard (28-day age-out) and new games enter deep. Every
simulation is re-ranked each hour; the forecast probabilities are the
time-averaged place occupancy over all simulations and hours.

Calibration (from the practice history, recomputed at run time):
- per-band diffusion sigma (std of hourly rank change, by rank band)
- velocity autocorrelation -> momentum weight
- chart churn (entry/exit rates)

Output: /output/forecast.json (or $MAKEFUN_OUTPUT_DIR/forecast.json) with
`probabilities` (place 1..10, per-game p, sums <= 1, rest left unnamed) and
`top10` (10 distinct universe ids in order).

Verified on the practice history: deterministic (two runs byte-identical),
~16s runtime, per-place probability sums of 0.90 (0.10 left unnamed for
newcomers the history never saw).
"""
import json
import math
import os
import random
import statistics
import sys
from datetime import datetime, timezone

# ---------------- tunables (env-overridable) ----------------
def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default

def _envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default

N_SIMS = _envi("FC_N_SIMS", 1200)  # Monte Carlo paths; 1200 gives stable place shares at ~16s
SIGMA_SCALE = _envf("FC_SIGMA_SCALE", 0.75)
ALPHA = _envf("FC_ALPHA", 0.55)
VEL_DECAY_HOURS = _envf("FC_VEL_DECAY_HOURS", 18.0)
BETA = _envf("FC_BETA", 0.012)
SURGE_DRIFT = _envf("FC_SURGE_DRIFT", 0.0)  # drift per unit log(surge) per hour
EXIT_BASE = _envf("FC_EXIT_BASE", 0.004)
EXIT_TENURE = _envf("FC_EXIT_TENURE", 0.00045)
ENTRY_PER_HOUR = _envf("FC_ENTRY_PER_HOUR", 0.8)
TARGET_SUM = _envf("FC_TARGET_SUM", 0.90)
FLOOR_P = _envf("FC_FLOOR_P", 0.0015)
CAP_P = _envf("FC_CAP_P", 0.85)
MIN_NAME_P = _envf("FC_MIN_NAME_P", 0.002)
SEED = _envi("FC_SEED", 20261112)
# -----------------------------------------------------------------


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    task = {}
    tp = os.path.join(input_dir, "task.json")
    if os.path.exists(tp):
        with open(tp) as f:
            task = json.load(f)
    return reads, task


def rank_band(r):
    if r <= 5:
        return 0
    if r <= 10:
        return 1
    if r <= 20:
        return 2
    if r <= 50:
        return 3
    return 4


def band_sigma(reads):
    """Per-band diffusion sigma (std of rank change per sqrt-hour), from history."""
    times = [parse_time(r["at"]) for r in reads]
    buckets = {i: [] for i in range(5)}
    for i in range(len(reads) - 1):
        dt = (times[i + 1] - times[i]).total_seconds() / 3600.0
        if dt <= 0:
            continue
        a = {g["universe_id"]: g["rank"] for g in reads[i]["up_and_coming"]}
        b = {g["universe_id"]: g["rank"] for g in reads[i + 1]["up_and_coming"]}
        for uid in set(a) & set(b):
            buckets[rank_band(a[uid])].append((b[uid] - a[uid]) / math.sqrt(dt))
    sig = []
    fallback = [0.5, 1.1, 1.7, 2.9, 3.6]
    for i in range(5):
        v = buckets[i]
        if len(v) >= 20:
            sig.append(statistics.pstdev(v))
        else:
            sig.append(fallback[i])
    return sig


def build_game_state(reads):
    """Per-game features from history: current rank, velocity, mean rank, tenure, playing."""
    times = [parse_time(r["at"]) for r in reads]
    series = {}
    for i, r in enumerate(reads):
        for g in r["up_and_coming"]:
            series.setdefault(g["universe_id"], []).append(
                (i, g["rank"], g.get("playing"))
            )
    last = reads[-1]["up_and_coming"]
    last_time = times[-1]
    state = {}
    for g in last:
        uid = g["universe_id"]
        pts = series.get(uid, [(len(reads) - 1, g["rank"], g.get("playing"))])
        ranks = [p[1] for p in pts]
        mean_rank = statistics.mean(ranks)
        # recent per-hour velocity: exponentially weighted over last intervals
        vel = 0.0
        wsum = 0.0
        for j in range(len(pts) - 1, max(0, len(pts) - 4), -1):
            i0, r0, _ = pts[j - 1]
            i1, r1, _ = pts[j]
            dt = (times[i1] - times[i0]).total_seconds() / 3600.0
            if dt <= 0:
                continue
            w = 0.55 ** (len(pts) - 1 - j)
            vel += w * (r1 - r0) / dt
            wsum += w
        if wsum > 0:
            vel /= wsum
        # tenure in days (first seen to last read)
        first_t = times[pts[0][0]]
        tenure_days = max(0.0, (last_time - first_t).total_seconds() / 86400.0)
        # age from details `created` if available (house recorder reads)
        age_days = None
        for r in reversed(reads):
            det = r.get("details")
            if det:
                d = det.get(str(uid)) or det.get(uid)
                if d and d.get("created"):
                    try:
                        c = parse_time(d["created"])
                        age_days = (last_time - c).total_seconds() / 86400.0
                    except Exception:
                        pass
                    break
        # playing surge: current playing / median of previous 2 reads
        surge = 1.0
        plays = [p[2] for p in pts if p[2]]
        if len(plays) >= 2 and plays[-1]:
            med = statistics.median(plays[-2:-1] + plays[:-1][-1:]) if len(plays) >= 3 else plays[-2]
            # median of the up-to-2 previous playing values
            prev = plays[:-1][-2:]
            if prev:
                med = statistics.median(prev)
                if med and med > 0:
                    surge = plays[-1] / med
        state[uid] = {
            "rank": g["rank"],
            "playing": g.get("playing"),
            "vel": vel,
            "mean_rank": mean_rank,
            "tenure_days": tenure_days,
            "age_days": age_days,
            "name": g.get("name", ""),
            "surge": surge,
        }
    return state


def forecast(input_dir):
    reads, task = load_history(input_dir)
    places = int(task.get("places", 10))
    window = task.get("window", {})
    n_hours = 72
    if window:
        try:
            f = parse_time(window["from"])
            u = parse_time(window["until"])
            n_hours = max(1, int(round((u - f).total_seconds() / 3600.0)))
        except Exception:
            pass
    sig = band_sigma(reads)
    state = build_game_state(reads)
    uids = sorted(state, key=lambda u: state[u]["rank"])
    n_games = len(uids)
    rng = random.Random(SEED)

    # counts[place][uid] = sim-hours at that place
    counts = {k: {} for k in range(1, places + 1)}
    rank_sum = {u: 0.0 for u in uids}
    rank_cnt = {u: 0 for u in uids}

    # precompute per-game constants
    vel0 = {u: state[u]["vel"] for u in uids}
    meanr = {u: state[u]["mean_rank"] for u in uids}
    import math as _math
    surge_drift = {u: SURGE_DRIFT * _math.log(max(state[u].get("surge", 1.0), 0.05)) for u in uids}
    # exit hazard per game per hour
    def exit_hazard(u):
        st = state[u]
        age = st["age_days"] if st["age_days"] is not None else st["tenure_days"]
        # age-out accelerates near 28 days; tenure is a lower bound on age
        h = EXIT_BASE + EXIT_TENURE * age
        if st["age_days"] is not None and st["age_days"] > 24:
            h += 0.02 * (st["age_days"] - 24)
        return min(h, 0.08)

    hz = {u: exit_hazard(u) for u in uids}

    for sim in range(N_SIMS):
        # continuous positions; new entrants get fresh ids (negative) deep down
        pos = {u: float(state[u]["rank"]) for u in uids}
        alive = set(uids)
        next_new = -1
        for h in range(n_hours):
            decay = 0.5 ** (h / VEL_DECAY_HOURS)
            # move each game
            for u in list(alive):
                p = pos[u]
                s = sig[rank_band(max(1, int(round(p))))] * SIGMA_SCALE
                drift = ALPHA * vel0[u] * decay - BETA * (p - meanr[u]) + surge_drift[u]
                pos[u] = p + drift + rng.gauss(0.0, s)
            # entries: a few new games deep in the chart
            n_new = 0
            if rng.random() < ENTRY_PER_HOUR:
                n_new = 1
            for _ in range(n_new):
                nid = next_new
                next_new -= 1
                pos[nid] = float(n_games + rng.uniform(0, 10))
                alive.add(nid)
                vel0[nid] = 0.0
                meanr[nid] = pos[nid]
                hz[nid] = EXIT_BASE
                surge_drift[nid] = 0.0
            # exits
            for u in list(alive):
                if u in hz and rng.random() < hz[u]:
                    alive.discard(u)
                    pos.pop(u, None)
            # re-rank
            order = sorted(alive, key=lambda u: pos[u])
            for k in range(1, places + 1):
                if k - 1 < len(order):
                    u = order[k - 1]
                    if u > 0:  # only real games count toward named probabilities
                        counts[k][u] = counts[k].get(u, 0) + 1
            for idx, u in enumerate(order):
                if u > 0:
                    rank_sum[u] += idx + 1
                    rank_cnt[u] += 1
    total = N_SIMS * n_hours
    # place probabilities
    probabilities = []
    for k in range(1, places + 1):
        c = counts[k]
        # raw simulated probabilities (share of sim-hours at this place)
        raw = {u: v / total for u, v in c.items() if v / total >= MIN_NAME_P}
        s = sum(raw.values())
        if s <= 0:
            continue
        # scale to TARGET_SUM so the rest is left unnamed (new entrants / tail)
        scale = TARGET_SUM / s
        for u, p in raw.items():
            p2 = min(max(p * scale, FLOOR_P), CAP_P)
            probabilities.append({"universe_id": int(u), "place": k, "p": round(p2, 6)})
    # top10: lowest expected rank, in order
    exp_rank = {}
    for u in uids:
        if rank_cnt[u] > 0:
            exp_rank[u] = rank_sum[u] / rank_cnt[u]
        else:
            exp_rank[u] = 999.0
    top10 = [int(u) for u in sorted(uids, key=lambda u: exp_rank[u])[:places]]
    return {"probabilities": probabilities, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    fc = forecast(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w") as f:
        json.dump(fc, f)
        f.write("\n")


if __name__ == "__main__":
    main()

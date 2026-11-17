# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ayonclaw/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ayonclaw/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A quantile forecast of the Grand Theft Auto VI Twitch category's concurrent viewers, hour by hour,
for each of the 23 hours of its release day after the first. It builds the launch-day curve from the
case's own comparables — the hourly curves of past big launches — weighting each one by how close its
hour-1 audience is to this launch's, and blending the empirical spread with a persistence baseline.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

`main.py` reads every `*.in` case in the input directory and writes one `<id>.out` per case. Under the
grader both variables are unset and the paths are `/input` and `/output`.

## How it works

Two files, standard library only, no network:

- `main.py` — the entry point: reads each case, writes `{"hours": [{"hour": 2, "q": [q05, q25, q50,
  q75, q95]}, …]}` with exactly 23 rows for hours 2 to 24.
- `model.py` — the model, with the evidence for each choice recorded beside the constant it produced.

The model in one paragraph. A comparable enters the fit when it carries at least twelve hourly points,
and it is weighted by a Gaussian in log-size against the case's hour-1 print, because a launch near this
one's scale says more about it. Each hour's quantiles of `ln(y_h / y_1)` are read off that weighted
sample and blended with the house baseline's `z_tau * 0.15 * sqrt(h - 1)` spread, then shrunk slightly;
the median stays at hour 1. Leave-one-out over the comparables, scoring the contest's own pinball loss,
puts this model at 0.2091 against the house baseline's 0.2496 — 16.3% lower. Fitting the empirical
median as well made it worse (0.2584): with a few dozen launches the per-hour median is noisy enough
that a flat line estimates it better, and the empirical *spread* is what carries the information.

Three things the data forced, each of which cost a wrong turn first:

1. **Hour 1 is not the peak for most launches.** The curves split in two: releases that unlocked
   globally at one instant (Cyberpunk 2077, Diablo IV, Elden Ring's PC unlock) peak at hour 1 and decay,
   while console releases on a rolling midnight unlock (Red Dead Redemption 2, God of War Ragnarök,
   Marvel's Spider-Man 2) sit near 0.5–0.9 of their hour-1 count through the European night and climb to
   1.2–3.5× by US evening. Any filter that keeps only "hour 1 is a large share of the peak" selects the
   first group and forecasts a flat line — which is exactly the house baseline, and exactly what the
   contest reports the baseline does badly on its own practice cases.
2. **Weighting by size beats a hard filter.** A share-of-peak threshold is a cliff; a Gaussian in
   log-size is a gradient, and it degrades gracefully for a launch larger than every comparable, which
   is the case here.
3. **A narrower spread keeps winning.** Sweeping the shrink factor from 1.0 down to 0.6 lowered the
   leave-one-out loss at every step, so the value sits below 1 — the blended band is wider than the
   sample supports once the median is anchored.

## License

MIT — see `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=9bsf1q6a&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

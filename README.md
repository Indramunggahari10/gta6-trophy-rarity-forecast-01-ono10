[![competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vjmunggahari-ui/gta6-trophy-rarity-forecast-01-ono10/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/vjmunggahari-ui/gta6-trophy-rarity-forecast-01-ono10/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts how many GTA VI players earn each trophy in week one: for every trophy it
predicts the earn rate (percent of players) at the case's target time, as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) on the logit scale, scored by mean pinball loss.

Entry point: `main.ts` at the repo root (Node 22, standard library only, no network).
It reads every `<id>.in` from the input directory and writes `<id>.out` to the output
directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}` — 55 rows,
one per trophy, quantiles non-decreasing in [0, 100], deterministic byte-for-byte across
runs, well under the 8 s per-case cap.

## How the model works

The growth model fits a per-trophy logit growth curve to the case history (least squares
on ln(1 + hours), fitted after the last detected refresh, shrunk to a difficulty prior
when the history is flat), then mixes it with a Poisson refresh mixture: Exophase's
player base grows in jumps as new cohorts arrive, so each trophy's rate is a mixture
over the cohorts that have seen it.

- Refresh rate λ = 0.212766/h and player multiplier ×6.0627, re-fitted from the case
  history when refreshes are visible, otherwise the published-aggregate priors.
- New-cohort fraction ρ is **time-dependent**: later cohorts have played longer, so ρ
  rises with time instead of staying flat.
- Player-base growth **saturates** rather than compounding at the launch-night
  multiplier, so long-horizon medians do not collapse toward zero.
- Mixture quantiles are produced by CDF inversion. `MAKEFUN_MODEL=baseline1|baseline2`
  selects the two house baselines instead of the growth model.

## House baselines (included, for comparison)

- **baseline1** — persistence: every trophy's rate at the target is its rate in the
  last read, with a random-walk spread on the logit scale.
- **baseline2** — fitted curve: the same least-squares line fit as the growth model,
  without the refresh mixture.

## Practice scores (proxy answers — read the caveat below)

Scored locally with `tools/score.mjs` on the kit's practice cases p00/p01:

| model | loss | typical miss | 90% coverage |
|---|---|---|---|
| **growth** | **0.533512** | 0.65 pp | 1.00 |
| baseline1 | 1.404752 | 0.68 pp | 0.02 |
| baseline2 | 1.404752 | 0.68 pp | 0.02 |

Perturbed-proxy mean (logit jitter on the proxy answers): growth 0.5443 / 0.5413,
baselines 1.384009 — the ranking is stable under proxy error.

On a synthetic final-like case (2-day history with refreshes, 6,510-minute horizon)
the growth model scores 0.0594, after the time-dependent ρ and saturating-growth
fixes (before them it was 1.375, with medians collapsing toward ~0.01).

## Proxy caveat — read this before trusting the numbers above

These practice losses are scored against **proxy answers reconstructed from published
aggregates** (a uniform dilution of the last rates, derived from the player and trophy
totals the competition itself publishes), NOT the real practice answers, which are
server-side only. The proxy favours dilution forecasters (including this one) and
understates the true loss: the house baselines' published real-answer loss is ~1.51,
against ~1.40 on the proxy. The server practice grade is the real check; treat the
numbers above as a relative check and a format check, never as the true loss.

## The final case (f00)

f00 does **not** ship with the kit. It is cut at the deadline 2026-11-21T11:30Z, with
target 2026-11-26T00:00Z (horizon 6,510 minutes) — no f00 history exists yet to check
against, so nothing here is tuned on it. The final-horizon behaviour was validated on
the synthetic final-like case above instead.

## Reproduce

```sh
node tools/score.mjs <kit-input-dir>   # scores main.ts on p00/p01
mkdir -p out && MAKEFUN_INPUT_DIR=<kit-input-dir> MAKEFUN_OUTPUT_DIR=out node main.ts
```

## License

MIT — see `LICENSE`.

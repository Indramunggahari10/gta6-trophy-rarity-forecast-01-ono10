// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01, @ono10).
//
// A program that reads one case (makefun/gta6-trophies-case/v1) per run and writes, for
// each of the game's trophies, five quantiles of its earn rate on Exophase at the case's
// target. Node 22, standard library only, no network, deterministic byte-for-byte.
//
// The model (growth mode, the default). Three facts from the competition's own rules
// drive it, and everything else is fitted from the case file itself:
//
//   1. RATES GROW along a curve. Each trophy's logit earn rate moves roughly linearly
//      in ln(1 + hours since the release, 2026-11-19T00:00Z). The line is fitted per
//      trophy over the case's history (least squares, reads after the last detected
//      refresh only, so a refresh does not drag the slope down); where the history is
//      flat or too short to fit, the slope shrinks toward a difficulty prior (bronze
//      story trophies climb fastest, platinum slowest; hidden trophies a little slower).
//   2. EXOPHASE REFRESHES on its own schedule, hours apart, and a refresh re-samples
//      the tracked players: the player base jumps (the one refresh the rules record
//      went 638 -> 3,868 players, x6.0627) and every rate but the one at 100% falls,
//      because the newly tracked players hold far fewer trophies (from the rules'
//      aggregates they held 103 non-100% trophies across 3,230 new players, against
//      the old cohort's 305 across 638: a new cohort's per-trophy rate is about
//      0.0667 of the old cohort's). One refresh therefore dilutes a rate by about
//      (1 + (6.0627 - 1) * 0.0667) / 6.0627 = 0.2206, the 100% trophy excepted:
//      every new player holds it, so it stays at 100%.
//   3. THE REFRESH COUNT in the horizon is uncertain, so the forecast is a MIXTURE:
//      refreshes arrive as a Poisson process (rate 0.212766 an hour, i.e. one every
//      ~4.7 h, the cadence the rules' one observed refresh implies), each refresh
//      diluting toward the new-cohort profile and the rate then growing again along
//      its curve until the next one. The five quantiles are the mixture's own
//      quantiles (its CDF inverted by bisection), so the lower quantiles sit near the
//      refreshed outcome and the upper ones near the unrefreshed outcome, with a
//      calibrated logit spread inside each component that widens with the horizon.
//
// The house baselines live in the same program so a local scorer can compare them:
// MAKEFUN_MODEL=baseline1 is the persistence baseline (the kit starter's numbers:
// q50 the last rate, spread 0.5 * sqrt(horizon days) on the logit scale), and
// MAKEFUN_MODEL=baseline2 is the fitted-curve baseline (the same least-squares line
// read at the target, spread the random walk's and the fit's residual combined;
// fewer than 3 reads falls back to persistence). The grader runs the program with
// no environment set, so the default model is growth.
//
// Output: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}, one row per
// trophy, quantiles non-decreasing in [0, 100], rounded to 4 decimals.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MODEL = process.env.MAKEFUN_MODEL ?? 'growth';
/** Calibrated logit spread inside each mixture component, at a 1-day horizon. */
const SPREAD = Number(process.env.MAKEFUN_SPREAD ?? '0.10');

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The release day, 2026-11-19T00:00:00Z, the origin of the growth curves. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Refresh hazard, per hour: one refresh every ~4.7 h (the rules' observed cadence). */
const REFRESH_LAMBDA_PER_HOUR = 0.212766;
/** Player-base multiplier of one refresh (638 -> 3,868 in the rules' refresh). */
const REFRESH_PLAYER_MULT = 6.0627;
/** A new cohort's per-trophy rate, as a fraction of the old cohort's (rules' aggregates). */
const NEW_COHORT_FRAC = 0.0667;
/** Random-walk spread, logit units per square-root day (the house baseline's 0.5). */
const RANDOM_WALK_SIGMA = 0.5;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: 'platinum' | 'gold' | 'silver' | 'bronze';
  points: number;
  secret: boolean;
}

interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: HistoryRead[];
  notes: string;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours from the release to an ISO instant. */
const hoursSinceRelease = (iso: string): number =>
  (Date.parse(iso) - RELEASE_MS) / 3_600_000;

/** Standard normal CDF (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7). */
function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const tail = Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI) * poly;
  return x >= 0 ? 1 - tail : tail;
}

/** Poisson pmf, computed in log space so large k stay finite. */
function poissonPmf(k: number, mean: number): number {
  let logP = -mean;
  for (let i = 1; i <= k; i++) logP += Math.log(mean / i);
  return Math.exp(logP);
}

/** The last non-null rate of trophy i in the history, else 0.5. */
function lastRate(c: Case, i: number): number {
  for (let h = c.history.length - 1; h >= 0; h--) {
    const r = c.history[h].rates[i];
    if (r !== null && r !== undefined) return r;
  }
  return 0.5;
}

/** Reads whose players_tracked jumped by more than 20% over the previous read. */
function refreshIndices(c: Case): number[] {
  const out: number[] = [];
  for (let h = 1; h < c.history.length; h++) {
    const prev = c.history[h - 1].players_tracked;
    const cur = c.history[h].players_tracked;
    if (prev > 0 && cur / prev > 1.2) out.push(h);
  }
  return out;
}

/**
 * Least-squares line of logit rate on u = ln(1 + hours since release), over the
 * given read indices. Returns slope, intercept and the residual standard deviation.
 * Reads with a null rate, or a rate at the 0/100 clamps' edge, are skipped.
 */
function fitLogitLine(c: Case, i: number, indices: number[]): { slope: number; intercept: number; residSd: number; n: number } {
  const pts: Array<[number, number]> = [];
  for (const h of indices) {
    const r = c.history[h].rates[i];
    if (r === null || r === undefined) continue;
    if (r <= 0.01 || r >= 99.99) continue;
    pts.push([Math.log1p(Math.max(0, hoursSinceRelease(c.history[h].at))), logit(r)]);
  }
  const n = pts.length;
  if (n < 2) return { slope: 0, intercept: pts.length === 1 ? pts[0][1] : logit(lastRate(c, i)), residSd: 0, n };
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (const [x, y] of pts) { sx += x; sy += y; sxx += x * x; sxy += x * y; }
  const denom = n * sxx - sx * sx;
  const slope = denom === 0 ? 0 : (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  let ss = 0;
  for (const [x, y] of pts) ss += (y - (intercept + slope * x)) ** 2;
  const residSd = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return { slope, intercept, residSd, n };
}

/** Difficulty prior for a trophy's growth slope (logit per unit of ln-hours). */
function priorSlope(t: Trophy, lastPct: number): number {
  const tierFactor = t.tier === 'platinum' ? 0.25 : t.tier === 'gold' ? 0.45 : t.tier === 'silver' ? 0.7 : 1.0;
  const secretFactor = t.secret ? 0.85 : 1.0;
  const headroom = Math.min(1, Math.max(0.2, 1 - lastPct / 100));
  return 0.35 * tierFactor * secretFactor * headroom;
}

/**
 * A trophy's growth slope: the line fitted over the reads after the last detected
 * refresh (the current regime), shrunk toward the difficulty prior when the fit is
 * thin, and clamped so no trophy is ever forecast to fall off a cliff or explode.
 */
function growthSlope(c: Case, i: number): number {
  const t = c.trophies[i];
  const lastPct = lastRate(c, i);
  const prior = priorSlope(t, lastPct);
  const refreshes = refreshIndices(c);
  const from = refreshes.length > 0 ? refreshes[refreshes.length - 1] : 0;
  const indices: number[] = [];
  for (let h = from; h < c.history.length; h++) indices.push(h);
  const fit = fitLogitLine(c, i, indices);
  if (fit.n < 3) {
    // Too thin to trust: the prior, nudged by whatever the fit says.
    const w = fit.n / (fit.n + 3);
    return Math.min(2, Math.max(-1, w * fit.slope + (1 - w) * prior));
  }
  const w = fit.n / (fit.n + 3);
  return Math.min(2, Math.max(-1, w * fit.slope + (1 - w) * prior));
}

/** Refresh rate and player multiplier: fitted from detected refreshes, else priors. */
function refreshParams(c: Case): { lambdaPerHour: number; playerMult: number } {
  const idx = refreshIndices(c);
  if (idx.length === 0) return { lambdaPerHour: REFRESH_LAMBDA_PER_HOUR, playerMult: REFRESH_PLAYER_MULT };
  const spanHours = Math.max(1, (Date.parse(c.at) - Date.parse(c.history[0].at)) / 3_600_000);
  let mults: number[] = [];
  for (const h of idx) mults.push(c.history[h].players_tracked / c.history[h - 1].players_tracked);
  mults.sort((a, b) => a - b);
  const medianMult = mults[Math.floor(mults.length / 2)];
  return {
    lambdaPerHour: Math.min(1, Math.max(0.02, idx.length / spanHours)),
    playerMult: Math.min(20, Math.max(1.2, medianMult)),
  };
}

/**
 * One refresh's dilution factor for a trophy whose rate is p (fraction) and whose
 * new-cohort fraction is rho: the refreshed rate is (p + (M - 1) * rho * p) / M.
 * The ubiquitous trophy (rho = 1) is undiluted.
 */
function dilute(p: number, playerMult: number, rho: number): number {
  return (p + (playerMult - 1) * rho * p) / playerMult;
}

/** Simulate one refresh-count scenario: k refreshes, evenly spaced, growth between. */
function simulateScenario(c: Case, i: number, k: number, slope: number): number {
  const horizonHours = c.horizon_minutes / 60;
  const t0 = c.at;
  const u0 = Math.log1p(Math.max(0, hoursSinceRelease(t0)));
  const u1 = Math.log1p(Math.max(0, hoursSinceRelease(c.target)));
  const { playerMult } = refreshParams(c);
  const lastPct = lastRate(c, i);
  const ubiquitous = lastPct >= 99.99;
  const rho = ubiquitous ? 1 : NEW_COHORT_FRAC;
  let l = logit(lastPct);
  if (k === 0) return invLogitPct(l + slope * (u1 - u0));
  // Refresh j lands at u0 + (j + 0.5) / k * (u1 - u0) in log-time.
  for (let j = 0; j < k; j++) {
    const uBefore = u0 + (j / k) * (u1 - u0);
    const uAfter = u0 + ((j + 1) / k) * (u1 - u0);
    const uRefresh = u0 + ((j + 0.5) / k) * (u1 - u0);
    l += slope * (uRefresh - uBefore);
    const p = invLogitPct(l) / 100;
    const pDiluted = ubiquitous ? p : dilute(p, playerMult, rho);
    l = logit(pDiluted * 100);
    l += slope * (uAfter - uRefresh);
  }
  return invLogitPct(l);
}

/** Quantiles of the refresh-count mixture for one trophy, in percent. */
function growthQuantiles(c: Case, i: number): number[] {
  const horizonHours = c.horizon_minutes / 60;
  const horizonDays = c.horizon_minutes / 1440;
  const { lambdaPerHour } = refreshParams(c);
  const mean = lambdaPerHour * horizonHours;
  const slope = growthSlope(c, i);
  const lastPct = lastRate(c, i);
  if (lastPct >= 99.99) return [100, 100, 100, 100, 100];
  // Mixture components: k refreshes, Poisson weights, up to 99.99% of the mass.
  const comps: Array<{ w: number; medianPct: number }> = [];
  let cum = 0;
  for (let k = 0; k <= 60; k++) {
    const w = poissonPmf(k, mean);
    if (w < 1e-9 && k > mean + 10) break;
    comps.push({ w, medianPct: simulateScenario(c, i, k, slope) });
    cum += w;
    if (cum >= 0.9999 && k >= mean) break;
  }
  const totalW = comps.reduce((s, co) => s + co.w, 0);
  for (const co of comps) co.w /= totalW;
  const spread = SPREAD * Math.sqrt(1 + horizonDays);
  // Invert the mixture CDF (in logit space) at each level by bisection.
  const cdf = (l: number): number => {
    let s = 0;
    for (const co of comps) s += co.w * normCdf((l - logit(co.medianPct)) / spread);
    return s;
  };
  const out: number[] = [];
  let prev = 0;
  for (const tau of LEVELS) {
    let lo = -15, hi = 15;
    for (let it = 0; it < 100; it++) {
      const mid = (lo + hi) / 2;
      if (cdf(mid) < tau) lo = mid; else hi = mid;
    }
    let q = round4(Math.min(100, Math.max(0, invLogitPct((lo + hi) / 2))));
    if (q < prev) q = prev;
    out.push(q);
    prev = q;
  }
  return out;
}

/** The persistence baseline (the kit starter's numbers, byte for byte in effect). */
function baseline1Quantiles(c: Case, i: number): number[] {
  const spread = RANDOM_WALK_SIGMA * Math.sqrt(c.horizon_minutes / 1440);
  const l = logit(lastRate(c, i));
  const out: number[] = [];
  let prev = 0;
  for (const z of Z) {
    let q = round4(Math.min(100, Math.max(0, invLogitPct(l + z * spread))));
    if (q < prev) q = prev;
    out.push(q);
    prev = q;
  }
  return out;
}

/** The fitted-curve baseline: the logit line over the last 24 h read at the target. */
function baseline2Quantiles(c: Case, i: number): number[] {
  const atMs = Date.parse(c.at);
  const indices: number[] = [];
  for (let h = 0; h < c.history.length; h++) {
    if (atMs - Date.parse(c.history[h].at) <= 24 * 3_600_000) indices.push(h);
  }
  if (indices.length < 3) return baseline1Quantiles(c, i);
  const fit = fitLogitLine(c, i, indices);
  const uTarget = Math.log1p(Math.max(0, hoursSinceRelease(c.target)));
  const medianLogit = fit.intercept + fit.slope * uTarget;
  const rw = RANDOM_WALK_SIGMA * Math.sqrt(c.horizon_minutes / 1440);
  const spread = Math.sqrt(rw * rw + fit.residSd * fit.residSd);
  const out: number[] = [];
  let prev = 0;
  for (const z of Z) {
    let q = round4(Math.min(100, Math.max(0, invLogitPct(medianLogit + z * spread))));
    if (q < prev) q = prev;
    out.push(q);
    prev = q;
  }
  return out;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  return c.trophies.map((t, i) => {
    const q =
      MODEL === 'baseline1' ? baseline1Quantiles(c, i)
      : MODEL === 'baseline2' ? baseline2Quantiles(c, i)
      : growthQuantiles(c, i);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}

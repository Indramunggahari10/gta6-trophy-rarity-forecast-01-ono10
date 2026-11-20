// Local practice scorer for gta6-trophy-rarity-forecast-01.
//
// Scores main.ts on the practice cases p00/p01 against PROXY answers reconstructed
// from the competition's own published aggregates (competition.yaml / the rules):
//
//   * The practice histories are flat: every read shows 638 players tracked, 943
//     trophies earned, and the same rates. The answers are the refreshed page: the
//     rules record that Exophase refreshed between 04:00Z and 04:30Z on 2026-11-19,
//     to 3,868 players tracked and 4,276 trophies earned, every rate but the one at
//     100% lower, and the 05:00Z read unchanged again (so p01's answer is the same
//     refreshed page as p00's).
//   * Per-trophy counts are recovered exactly from the printed rates: count =
//     round(rate * 638 / 100). The recovered counts sum to 943, the published
//     trophies-earned total, which checks the recovery (each printed rate is a
//     2-decimal rounding of count / 638).
//   * The refresh added 3,868 - 638 = 3,230 new players and 4,276 - 943 = 3,333
//     newly earned trophies. The trophy at 100% stayed at 100%, so all 3,230 new
//     players hold it: that accounts for 3,230 of the 3,333. The remaining 103
//     newly earned trophies, over the other 54 trophies, are distributed in
//     proportion to the old counts (their sum is 943 - 638 = 305), i.e. the new
//     cohort's per-trophy rate is (103 / 3,230) / (305 / 638) = 0.0667 of the old
//     cohort's. The proxy answer for a trophy is therefore
//         y = (oldCount + 103 * oldCount / 305) / 3,868 * 100,
//     a uniform dilution of the old rate by (305 + 103) / 3,868 / (305 / 638)
//     = 0.2206, rounded to the page's 2 decimals; the 100% trophy stays 100.
//   * This is a PROXY, not the real answer: the real 04:30Z rates are unknown, and
//     the house baseline's real-answer loss is ~1.51 against ~1.40 on this proxy.
//     The perturbed-proxy check below (each proxy rate jittered in logit space)
//     measures how much the ranking moves if the proxy is off.
//
// The scorer also runs the house baselines (MAKEFUN_MODEL=baseline1/baseline2) for
// comparison, checks determinism (two growth runs byte-identical) and runtime.
//
// Usage: node tools/score.mjs [--spread 0.20] [--models growth,baseline1,baseline2]

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const WORKDIR = join(dirname(fileURLToPath(import.meta.url)), '..');
const KIT_INPUT = join(process.env.HOME, 'workspace/makefun/gta6-trophy-rarity-forecast-01-kit-v2/gta6-trophy-rarity-forecast-01-kit/input');
const TMP = join(WORKDIR, '.score-tmp');

const args = {};
{
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue;
    const key = argv[i].slice(2);
    if (argv[i + 1] && !argv[i + 1].startsWith('--')) args[key] = argv[++i];
    else args[key] = 'true';
  }
}
const SPREAD = args.spread ?? '0.10'; // must match main.ts's calibrated default
const MODELS = (args.models ?? 'growth,baseline1,baseline2').split(',');

const clampPct = (x) => Math.min(99.99, Math.max(0.01, x));
const logit = (pct) => { const p = clampPct(pct) / 100; return Math.log(p / (1 - p)); };

/** Pinball loss on the logit scale, exactly as the rules score it. */
function pinballLoss(forecast, answer) {
  const levels = [0.05, 0.25, 0.5, 0.75, 0.95];
  let sum = 0, n = 0, miss = 0, covered = 0;
  for (const row of forecast.trophies) {
    const y = answer[row.id];
    const ly = logit(y);
    row.q.forEach((q, j) => {
      const tau = levels[j];
      const d = ly - logit(q);
      sum += 2 * Math.max(tau * d, (tau - 1) * d);
      n++;
    });
    miss += Math.abs(y - row.q[2]);
    if (row.q[0] <= y && y <= row.q[4]) covered++;
  }
  return { loss: sum / n, typicalMissPp: miss / forecast.trophies.length, coverage90: covered / forecast.trophies.length };
}

/** Seeded PRNG (mulberry32) + Box-Muller, so perturbed runs are reproducible. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function normal01(r) {
  const u1 = Math.max(1e-12, r()), u2 = r();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Reconstruct the proxy answers for one case (see the header). */
function proxyAnswers(c) {
  const last = c.history[c.history.length - 1];
  const n0 = last.players_tracked, e0 = last.trophies_earned;
  const n1 = 3868, e1 = 4276; // the refreshed page, per the rules
  const counts = last.rates.map((r) => Math.round((r * n0) / 100));
  const sumCounts = counts.reduce((a, b) => a + b, 0);
  const i100 = last.rates.findIndex((r) => r === 100);
  const oldNon100 = e0 - counts[i100];
  const newNon100 = (e1 - e0) - (n1 - n0);
  const ans = {};
  let dilutionSum = 0, dilutionN = 0;
  c.trophies.forEach((t, i) => {
    if (i === i100) { ans[t.id] = 100; return; }
    const y = Math.round(((counts[i] + (newNon100 * counts[i]) / oldNon100) / n1) * 100 * 100) / 100;
    ans[t.id] = y;
    dilutionSum += y / last.rates[i]; dilutionN++;
  });
  return { ans, sumCounts, e0, dilution: dilutionSum / dilutionN, newNon100, oldNon100 };
}

/** Run main.ts once for a model on one case dir, return {ms, outPath}. */
function runModel(model, inputDir, outputDir) {
  const t0 = Date.now();
  const r = spawnSync('node', [join(WORKDIR, 'main.ts')], {
    cwd: WORKDIR,
    env: { ...process.env, MAKEFUN_INPUT_DIR: inputDir, MAKEFUN_OUTPUT_DIR: outputDir, MAKEFUN_MODEL: model, MAKEFUN_SPREAD: SPREAD },
    encoding: 'utf8',
  });
  if (r.status !== 0) throw new Error(`main.ts (${model}) failed: ${r.stderr}`);
  return { ms: Date.now() - t0, outPath: join(outputDir, readdirSync(inputDir).find((f) => f.endsWith('.in')).slice(0, -3) + '.out') };
}

const cases = readdirSync(KIT_INPUT).filter((f) => f.endsWith('.in')).sort();
console.log(`spread=${SPREAD} models=${MODELS.join(',')}`);
const summary = {};
for (const file of cases) {
  const id = file.slice(0, -3);
  const c = JSON.parse(readFileSync(join(KIT_INPUT, file), 'utf8'));
  const { ans, sumCounts, e0, dilution, newNon100, oldNon100 } = proxyAnswers(c);
  console.log(`\n== ${id} (at ${c.at}, horizon ${c.horizon_minutes} min, ${c.trophies.length} trophies, history ${c.history.length} reads)`);
  console.log(`   proxy: recovered counts sum ${sumCounts} (published earned ${e0}) ${sumCounts === e0 ? 'OK' : 'MISMATCH'}, newNon100 ${newNon100} / oldNon100 ${oldNon100}, dilution factor ${dilution.toFixed(4)}`);
  // One input dir per case, so each run sees only its own case file.
  const inDir = join(TMP, id, 'in'), outDir = join(TMP, id, 'out');
  rmSync(join(TMP, id), { recursive: true, force: true });
  mkdirSync(inDir, { recursive: true });
  writeFileSync(join(inDir, file), readFileSync(join(KIT_INPUT, file), 'utf8'));
  for (const model of MODELS) {
    const { ms, outPath } = runModel(model, inDir, outDir);
    const fc = JSON.parse(readFileSync(outPath, 'utf8'));
    const rows = fc.trophies.length;
    const bad = fc.trophies.filter((r) => r.q.length !== 5 || r.q.some((q, j) => !Number.isFinite(q) || q < 0 || q > 100 || (j > 0 && q < r.q[j - 1]))).length;
    const s = pinballLoss(fc, ans);
    summary[`${id}:${model}`] = s.loss;
    console.log(`   ${model.padEnd(10)} loss ${s.loss.toFixed(6)}  miss ${s.typicalMissPp.toFixed(2)}pp  cov90 ${s.coverage90.toFixed(2)}  rows ${rows} bad ${bad}  ${ms} ms  ${existsSync(outPath) ? readFileSync(outPath, 'utf8').length : 0} bytes`);
    if (model === 'growth') {
      // Determinism: run again into a second dir and compare bytes.
      const outDir2 = join(TMP, id, 'out2');
      runModel(model, inDir, outDir2);
      const b1 = readFileSync(outPath), b2 = readFileSync(join(outDir2, `${id}.out`));
      console.log(`   growth determinism: ${b1.equals(b2) ? 'BYTE-IDENTICAL' : 'MISMATCH'} (2 runs)`);
      // Perturbed proxy: 8 seeds, each proxy rate jittered +-0.15 logits.
      let pertSum = 0;
      for (let seed = 1; seed <= 8; seed++) {
        const r = rng(seed * 7919);
        const pAns = {};
        for (const t of c.trophies) {
          const y = ans[t.id];
          if (y >= 100) { pAns[t.id] = 100; continue; }
          const ly = logit(y) + 0.15 * normal01(r);
          pAns[t.id] = Math.min(99.99, Math.max(0.01, 100 / (1 + Math.exp(-ly))));
        }
        pertSum += pinballLoss(fc, pAns).loss;
      }
      console.log(`   growth perturbed-proxy mean (8 seeds, +-0.15 logit): ${(pertSum / 8).toFixed(6)}`);
    }
  }
}
console.log('\n== totals (mean over practice cases)');
for (const model of MODELS) {
  const vals = cases.map((f) => summary[`${f.slice(0, -3)}:${model}`]);
  console.log(`   ${model.padEnd(10)} ${(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(6)}   (per case: ${vals.map((v) => v.toFixed(6)).join(', ')})`);
}

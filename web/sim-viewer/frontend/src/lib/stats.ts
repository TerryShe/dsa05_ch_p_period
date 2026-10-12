// Descriptive helpers over per-deployment values. Nothing here replaces the
// pre-registered analysis; it only arranges the formal records for display.

/** The offset of the analysis' log10(w + delta) sensitivity metric (design §5.1). */
export const DELTA = 1e-4;

export function logRate(w: number): number {
  return Math.log10(w + DELTA);
}

/** Linear map of [d0, d1] onto [r0, r1]. */
export function linear(d0: number, d1: number, r0: number, r1: number) {
  const k = (r1 - r0) / (d1 - d0);
  return (v: number) => r0 + (v - d0) * k;
}

/** Empirical CDF as steps: share of values <= x[i]. */
export function ecdf(values: number[]): { x: number[]; y: number[] } {
  const sorted = [...values].sort((p, q) => p - q);
  const x: number[] = [];
  const y: number[] = [];
  sorted.forEach((v, i) => {
    if (i + 1 < sorted.length && sorted[i + 1] === v) return;
    x.push(v);
    y.push((i + 1) / sorted.length);
  });
  return { x, y };
}

/** Share of the step function at x (0 left of the first step). */
export function shareAt(steps: { x: number[]; y: number[] }, x: number): number {
  let lo = 0;
  let hi = steps.x.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (steps.x[mid] <= x) lo = mid + 1;
    else hi = mid;
  }
  return lo === 0 ? 0 : steps.y[lo - 1];
}

export interface Tally {
  aLower: number;
  bLower: number;
  equal: number;
  bothZero: number;
}

export function tally(a: number[], b: number[]): Tally {
  const t: Tally = { aLower: 0, bLower: 0, equal: 0, bothZero: 0 };
  a.forEach((v, i) => {
    if (v < b[i]) t.aLower += 1;
    else if (v > b[i]) t.bLower += 1;
    else {
      t.equal += 1;
      if (v === 0) t.bothZero += 1;
    }
  });
  return t;
}

export interface Difference {
  instance: number;
  a: number;
  b: number;
  diff: number;
}

/** The k deployments with the largest |a - b|; ties keep instance order. */
export function largestDifferences(
  instances: number[],
  a: number[],
  b: number[],
  k: number,
): Difference[] {
  return instances
    .map((instance, i) => ({ instance, a: a[i], b: b[i], diff: a[i] - b[i] }))
    .filter((d) => d.diff !== 0)
    .sort((p, q) => Math.abs(q.diff) - Math.abs(p.diff) || p.instance - q.instance)
    .slice(0, k);
}

/** Index of the point nearest to (px, py) within radius, or -1. */
export function nearest(
  xs: number[],
  ys: number[],
  px: number,
  py: number,
  radius: number,
): number {
  let best = -1;
  let bestDistance = radius * radius;
  for (let i = 0; i < xs.length; i += 1) {
    const d = (xs[i] - px) ** 2 + (ys[i] - py) ** 2;
    if (d <= bestDistance) {
      best = i;
      bestDistance = d;
    }
  }
  return best;
}

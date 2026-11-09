import type { Comparison, Leaderboard, Estimate } from '../types';
export const confidenceInterval = (ci: unknown): [number, number] | null => Array.isArray(ci) && ci.length === 2 && ci.every(v => typeof v === 'number' && Number.isFinite(v)) ? [ci[0], ci[1]] : null;
export function adaptLeaderboard(data: Leaderboard) {
  return { ...data, rows: data.rows.map(row => ({ ...row, rank: row.value === null ? null : row.rank ?? null, ci: confidenceInterval(row.ci) })) };
}
export function adaptCompare(data: Comparison) {
  if (data.groups.some(g => g.env_version !== data.benchmark.env_version || g.total_steps !== data.benchmark.total_steps)) throw new Error('The server returned mixed benchmarks. Choose a single benchmark.');
  return {
    benchmark: data.benchmark,
    groups: data.groups.map(g => ({ ...g, final: { ...g.final, ci: confidenceInterval(g.final.ci) },
      points: g.curve.map(p => ({ x: p.step, y: p.mean, lo: p.lo, hi: p.hi, n: p.n })),
      inDistribution: g.generalization.default ?? null,
      outOfDistribution: Object.entries(g.generalization).filter(([name]) => name !== 'default').map(([name, estimate]) => ({ name, ...estimate })),
    })),
  };
}
export function estimateText(estimate?: Estimate | null) { return estimate?.mean == null ? null : estimate.mean; }

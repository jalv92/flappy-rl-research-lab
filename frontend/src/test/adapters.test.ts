import { describe, expect, it } from 'vitest';
import type { Comparison, Leaderboard, Metrics } from '../types';
import { adaptCompare, adaptLeaderboard, confidenceInterval } from '../lab/adapters';
import { mergeMetrics } from '../lab/Training';
import { fixture } from './fixtures';
describe('comparison fixtures',()=>{
  it('preserves every measured point and CI, and separates ID from OOD',()=>{
    const original=fixture<Comparison>('compare'), result=adaptCompare(original);
    expect(result.benchmark).toEqual(original.benchmark);
    result.groups.forEach((g,i)=>{
      expect(g.points).toEqual(original.groups[i].curve.map(p=>({x:p.step,y:p.mean,lo:p.lo,hi:p.hi,n:p.n})));
      expect(g.final.ci).toEqual(confidenceInterval(original.groups[i].final.ci));
      expect(g.inDistribution).toEqual(original.groups[i].generalization.default ?? null);
      expect(g.outOfDistribution.some(s=>s.name==='default')).toBe(false);
      expect(g.outOfDistribution.length).toBe(Object.keys(original.groups[i].generalization).filter(k=>k!=='default').length);
    });
  });
  it('keeps unevaluated generalization empty',()=>{
    const data=fixture<Comparison>('compare'); data.groups[0].generalization={};
    const group=adaptCompare(data).groups[0];
    expect(group.inDistribution).toBeNull();expect(group.outOfDistribution).toEqual([]);
  });
  it('rejects mixed benchmarks',()=>{
    const data=fixture<Comparison>('compare'); data.groups[0].total_steps+=1;
    expect(()=>adaptCompare(data)).toThrow('mixed benchmarks');
  });
});
describe('leaderboard fixtures',()=>{
  it.each(['leaderboard_final_mean','leaderboard_steps_to_50','leaderboard_test_mean'])('preserves server ranking, metric direction and missing values in %s',name=>{
    const original=fixture<Leaderboard>(name), result=adaptLeaderboard(original);
    expect(result.lower_is_better).toBe(original.lower_is_better);
    expect(result.benchmark).toEqual(original.benchmark);
    result.rows.forEach((r,i)=>{
      expect(r.value).toBe(original.rows[i].value);
      expect(r.rank).toBe(r.value===null?null:original.rows[i].rank??null);
      expect(r.n_seeds).toBe(original.rows[i].n_seeds);
      expect(r.ci).toEqual(confidenceInterval(original.rows[i].ci));
    });
  });
  it('shows unranked held-out groups and does not turn an empty CI object into zero',()=>{
    const rows=adaptLeaderboard(fixture<Leaderboard>('leaderboard_test_mean')).rows;
    expect(rows.some(r=>r.value===null&&r.rank===null)).toBe(true);
    expect(rows.every(r=>r.ci===null)).toBe(true);
  });
});
describe('incremental training measurements',()=>{
  it('merges captured measurements by step and replaces duplicates without losing other series',()=>{
    const data=fixture<Metrics>('run_metrics');
    const key=Object.keys(data)[0], points=data[key];
    const base={...data,[key]:points.slice(0,3)};
    const merged=mergeMetrics(base,{[key]:points.slice(2)});
    expect(merged).toEqual(data);
    expect(mergeMetrics({loss:[[1000,4],[2000,2]]},{loss:[[2000,1],[3000,.5]]}).loss).toEqual([[1000,4],[2000,1],[3000,.5]]);
  });
});

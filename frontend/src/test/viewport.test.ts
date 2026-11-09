import { describe, expect, it } from 'vitest';
import { readStageSize, viewWidthChanged, viewWidthForStage } from '../game/viewport';
import type { GameConfig } from '../types';
import { fixture } from './fixtures';
import styles from '../styles.css?raw';

const world = fixture<GameConfig>('game_config').difficulties.normal;
describe('landscape viewport geometry', () => {
  it.each([
    [984, 480, 1049], [810, 560, 740], [2000, 480, 1296], [360, 600, 432],
  ])('fits %s × %s without exceeding the server range', (width, height, expected) => {
    expect(viewWidthForStage({ width, height }, world, [432, 1296])).toBe(expected);
  });
  it('uses advertised geometry and limits, including non-default worlds', () => {
    expect(viewWidthForStage({ width: 900, height: 500 }, { ...world, world_height: 600 }, [500, 1000])).toBe(1000);
  });
  it('measures the content box without counting borders or temporary scrollbars', () => {
    const stage = document.createElement('div');
    stage.style.border = '1px solid';
    stage.getBoundingClientRect = () => ({ width: 802, height: 482 } as DOMRect);
    // client dimensions are smaller when the old world needs scrollbars.
    Object.defineProperties(stage, { clientWidth: { value: 785 }, clientHeight: { value: 465 } });
    document.body.appendChild(stage);
    try { expect(readStageSize(stage)).toEqual({ width: 800, height: 480 }); }
    finally { stage.remove(); }
  });
  it('uses the default view when size is unavailable or the backend predates the width range', () => {
    expect(viewWidthForStage({ width: 0, height: 0 }, world, [432, 1296])).toBe(world.view_width);
    expect(viewWidthForStage({ width: 1000, height: 500 }, world)).toBe(world.view_width);
  });
  it('ignores tiny layout changes but updates a materially different view', () => {
    expect(viewWidthChanged(1000, 1001)).toBe(false);
    expect(viewWidthChanged(1000, 1049)).toBe(false);
    expect(viewWidthChanged(1000, 1050)).toBe(true);
    expect(viewWidthChanged(432, 450)).toBe(false);
    expect(viewWidthChanged(432, 456)).toBe(true);
  });
  it.each([[984, 480], [810, 560]])('covers at least 80%% of a %s × %s stage at uniform scale with full height', (width, height) => {
    const view = viewWidthForStage({ width, height }, world, [432, 1296]);
    const scale = Math.min(width / view, height / world.world_height);
    expect(view * scale / width).toBeGreaterThanOrEqual(.8);
    expect(world.world_height * scale).toBe(height);
    expect(world.world_height * scale).toBeGreaterThanOrEqual(480);
  });
  it('enforces the desktop minimum in the shipped CSS and allows page scrolling', () => {
    // jsdom cannot lay out container units; inspect the actual CSS contract as well
    // as the uniform-scale calculations above. Browser measurements are separate.
    const style = document.createElement('style');
    style.textContent = styles;
    document.head.appendChild(style);
    try {
      const rules = Array.from(style.sheet!.cssRules);
      const rule = rules.find(r => (r as CSSStyleRule).selectorText === '.game-workspace') as CSSStyleRule;
      expect(rule.style.getPropertyValue('--stage-height')).toBe('clamp(482px, calc(100dvh - 338px), 722px)');
      expect(rule.style.height).toBe('');
      const desktop = rules.find(r => (r as CSSMediaRule).conditionText === '(min-width: 1001px)') as CSSMediaRule;
      const worldRule = Array.from(desktop.cssRules).find(r => (r as CSSStyleRule).selectorText === '.game-workspace .world') as CSSStyleRule;
      expect(worldRule.style.getPropertyValue('min-width')).toBe('calc(480px * var(--world-aspect))');
      const stageRule = Array.from(desktop.cssRules).find(r => (r as CSSStyleRule).selectorText === '.game-workspace .flight-stage') as CSSStyleRule;
      expect(stageRule.style.overflow).toBe('auto');
      const constrained = Array.from(desktop.cssRules).find(r => (r as CSSStyleRule).selectorText === '.game-workspace .flight-stage.width-constrained') as CSSStyleRule;
      expect(constrained.style.height).toBe('calc(var(--stage-height) + 20px)');
    } finally { style.remove(); }
  });
});

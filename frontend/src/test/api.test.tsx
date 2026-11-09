import { act, cleanup, render, renderHook, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { request, useAction } from '../api';
import { Report } from '../components/Report';
import { fixture } from './fixtures';
afterEach(()=>{cleanup();vi.unstubAllGlobals();});
describe('REST failures',()=>{
  it.each([404,409,422])('retains the real %s detail',async status=>{
    const body=fixture<{detail:unknown}>(`error_${status}`);
    vi.stubGlobal('fetch',vi.fn(()=>Promise.resolve(new Response(JSON.stringify(body),{status}))));
    await expect(request('/api/models/missing')).rejects.toThrow(`${status}: ${typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)}`);
  });
  it('does not navigate or update a screen after its mutation outlives the screen',async()=>{
    let finish:((value:string)=>void)|undefined;
    const callback=vi.fn();
    const {result,unmount}=renderHook(()=>useAction());
    let action:Promise<void>|undefined;
    act(()=>{action=result.current.execute(()=>new Promise<string>(resolve=>{finish=resolve;}),callback);});
    unmount();
    await act(async()=>{finish!('done');await action;});
    expect(callback).not.toHaveBeenCalled();
  });
});
describe('experiment report rendering',()=>{
  it('renders server report structure and treats HTML as inert text',()=>{
    render(<Report markdown={'# Experiment 1\n\n- Decision: **inconclusive**\n- Code: `abc123`\n\n## Results\n\n| arm | mean |\n|---|---|\n| control | 124 |\n\n<script>alert(1)</script>'}/>);
    expect(screen.getByRole('heading',{name:'Experiment 1'})).toBeTruthy();
    expect(screen.getByRole('table')).toBeTruthy();
    expect(screen.getByText('inconclusive').tagName).toBe('STRONG');
    expect(screen.getByText('abc123').tagName).toBe('CODE');
    expect(document.querySelector('script')).toBeNull();
    expect(screen.getByText('<script>alert(1)</script>')).toBeTruthy();
  });
});

import { Component, useEffect, useRef, useState, type ReactNode } from 'react';
import { HashRouter, Link, NavLink, Route, Routes, useLocation } from 'react-router-dom';
import { Activity, ArrowUpRight, Bird, Bot, ChartNoAxesCombined, ChevronRight, CircleDot, FlaskConical, Gamepad2, History, LayoutDashboard, Menu, Play, Settings, SlidersHorizontal, Trophy, Users, X, type LucideIcon } from 'lucide-react';
import { MotionConfig } from 'motion/react';
import { useResource } from './api';
import type { MatchStats, Mode } from './types';
import { Badge, ErrorMessage, fmt, PageHeader, Panel, ResourceView, Stat } from './components/ui';
import { GamePage, modes } from './game/GamePage';
import { BirdArt, CanyonArt } from './game/art';
import { SettingsPage, SettingsProvider, useSettings } from './settings';
import { Overview } from './lab/Overview';
import { Training } from './lab/Training';
import { ComparisonPage, LeaderboardPage } from './lab/Comparison';
import { ModelsPage } from './lab/Models';
import { ControlPage, HistoryPage } from './lab/Research';
const flightNav: { to:string; label:string; icon:LucideIcon }[] = [
  { to:'/', label:'Flight deck', icon:Gamepad2 }, { to:'/play/human', label:'Human', icon:Bird },
  { to:'/play/ai', label:'AI flight', icon:Bot }, { to:'/play/human_vs_ai', label:'Human vs AI', icon:Users },
  { to:'/play/ai_vs_ai', label:'AI vs AI', icon:CircleDot }, { to:'/replays', label:'Replay archive', icon:History },
];
const labNav: typeof flightNav = [
  { to:'/lab',label:'Overview',icon:LayoutDashboard }, { to:'/lab/training',label:'Live training',icon:Activity },
  { to:'/lab/compare',label:'Algorithm comparison',icon:ChartNoAxesCombined }, { to:'/lab/history',label:'Research history',icon:History },
  { to:'/lab/leaderboard',label:'Leaderboard',icon:Trophy }, { to:'/lab/models',label:'Model viewer',icon:Bot },
  { to:'/lab/control',label:'Control centre',icon:SlidersHorizontal },
];
class ScreenBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };
  static getDerivedStateFromError(error: Error) { return { error: error.message }; }
  render() { return this.state.error ? <Panel title="This view could not be displayed"><ErrorMessage message={this.state.error}/><button className="button" onClick={() => this.setState({ error: null })}>Try again</button><p className="muted">The backend may have returned data this view cannot read.</p></Panel> : this.props.children; }
}
function Lobby() {
  const stats = useResource<MatchStats>('/api/matches/stats');
  return <><section className="lobby-hero"><CanyonArt/><div className="hero-shade"/><div className="hero-copy"><div className="hero-kicker"><span className="tiny-mark"/>Flight meets intelligence</div><h1>aerie<span>Find your<br/>next horizon.</span></h1><p>Follow your instincts. Challenge a policy.<br/>See how far a little lift can take you.</p><Link className="button primary hero-cta" to="/play/human_vs_ai"><Play size={18}/>Challenge the AI<ChevronRight size={18}/></Link></div><div className="hero-footnote"><span>Original flight arcade</span><span>One shared world. Two ways to fly.</span></div></section>
    <div className="section-title"><div><h2>Choose your flight.</h2><p>Take the controls, or let curiosity lead.</p></div><span className="muted">Space · click · tap</span></div>
    <div className="mode-grid">{([
      ['human','Just you & the sky.','Find a rhythm and chase your personal best.',Bird],
      ['ai','A policy in motion.','Watch a model think, one decision at a time.',Bot],
      ['human_vs_ai','A worthy rival.','Your reflexes against a trained agent.',Users],
      ['ai_vs_ai','Settle it in the sky.','Two models. Shared conditions. Clear results.',CircleDot],
    ] as const).map(([mode,title,description,Icon]) => <Link to={`/play/${mode}`} className={`mode-card ${mode === 'human_vs_ai' ? 'featured' : ''}`} key={mode}><div className="mode-card-top"><Icon size={23}/><span>{modes[mode].caption}</span><ArrowUpRight size={18}/></div><h3>{title}</h3><p>{description}</p><span className="mode-open">{mode === 'human' ? 'Take flight' : mode === 'ai' ? 'Choose a model' : 'Enter the arena'}<ChevronRight size={14}/></span></Link>)}</div>
    <div className="lobby-bottom"><Panel title="Your flight log"><ResourceView resource={stats}>{s => <div className="stat-strip small"><Stat label="Human flights" value={fmt(s.human.games,0)}/><Stat label="Normal personal best" value={fmt(s.human.best.normal,0)} detail="Obstacles cleared"/><Stat label="Head-to-head wins" value={fmt(s.human.vs_ai.wins,0)}/></div>}</ResourceView><Link className="text-link" to="/replays">Revisit a recorded flight<ArrowUpRight size={15}/></Link></Panel><Link className="lab-invitation" to="/lab"><div className="lab-invitation-icon"><FlaskConical size={26}/><BirdArt/></div><div><span>The research laboratory</span><h2>Beyond the flight.</h2><p>Train policies. Test ideas.<br/>Follow the evidence.</p></div><ArrowUpRight size={24}/></Link></div></>;
}
function Shell() {
  const location = useLocation(); const [mobileOpen,setMobileOpen] = useState(false);
  const [isMobile,setIsMobile] = useState(() => window.matchMedia('(max-width: 1000px)').matches);
  const menuButton = useRef<HTMLButtonElement>(null);
  const sidebar = useRef<HTMLElement>(null);
  const wasOpen = useRef(false);
  useEffect(() => {
    const media = window.matchMedia('(max-width: 1000px)');
    const update = () => { setIsMobile(media.matches); if (!media.matches) setMobileOpen(false); };
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  useEffect(() => {
    if (mobileOpen && isMobile) sidebar.current?.querySelector<HTMLAnchorElement>('a')?.focus();
    else if (wasOpen.current && isMobile) menuButton.current?.focus();
    wasOpen.current = mobileOpen;
    const escape = (event:KeyboardEvent) => { if (event.key === 'Escape') setMobileOpen(false); };
    if (mobileOpen) window.addEventListener('keydown', escape);
    return () => window.removeEventListener('keydown', escape);
  }, [mobileOpen,isMobile]);
  useEffect(() => {
    if (!mobileOpen || !isMobile) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      const targets = sidebar.current?.querySelectorAll<HTMLElement>('a,button');
      if (!targets?.length) return;
      const first = targets[0], last = targets[targets.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    window.addEventListener('keydown', trapFocus);
    return () => { document.body.style.overflow = overflow; window.removeEventListener('keydown', trapFocus); };
  }, [mobileOpen, isMobile]);
  const main = useRef<HTMLElement>(null); const initial = useRef(true);
  const health = useResource<{ ok:boolean; api_version:string; engine_version:string; env_version:string }>('/api/health',10000);
  const isLab = location.pathname.startsWith('/lab');
  const pageLabel = [...flightNav,...labNav,{ to:'/settings',label:'Settings' }].find(n => n.to === location.pathname)?.label ?? 'Flight deck';
  const { reduce } = useSettings();
  useEffect(() => { setMobileOpen(false); document.title = `${pageLabel} · Aerie`; if (!initial.current) main.current?.focus(); initial.current = false; window.scrollTo(0,0); }, [location.pathname,pageLabel]);
  return <MotionConfig reducedMotion={reduce ? 'always' : 'user'}><a className="skip-link" href="#main" onClick={e=>{e.preventDefault();main.current?.focus();}}>Skip to content</a><div className="app-shell"><aside ref={sidebar} id="main-navigation" inert={isMobile && !mobileOpen} className={`sidebar ${mobileOpen ? 'open' : ''}`}>
    <Link className="brand" to="/"><img src="/aerie.svg" alt=""/><span>aerie<small>Flight & intelligence</small></span></Link><button className="mobile-close icon-button" aria-label="Close navigation" onClick={() => setMobileOpen(false)}><X size={20}/></button>
    <nav aria-label="Main navigation"><div className="nav-section-label">Arcade</div>{flightNav.map(({to,label,icon:Icon}) => <NavLink key={to} to={to} end className={({isActive}) => `nav-link ${isActive?'active':''}`}><Icon size={17}/><span>{label}</span>{to === '/play/human_vs_ai' && <span className="nav-star">✦</span>}</NavLink>)}<div className="nav-section-label lab-label">Laboratory</div>{labNav.map(({to,label,icon:Icon}) => <NavLink key={to} to={to} end className={({isActive})=>`nav-link ${isActive?'active':''}`}><Icon size={17}/><span>{label}</span></NavLink>)}</nav>
    <div className="sidebar-bottom"><NavLink className="nav-link" to="/settings"><Settings size={17}/>Settings</NavLink><div className="sidebar-caption"><span className="tiny-mark"/><span>Flappy RL Research Lab<small>Built for curious minds.</small></span></div></div>
  </aside>{isMobile && mobileOpen && <button className="nav-scrim" tabIndex={-1} aria-label="Close navigation" onClick={() => setMobileOpen(false)}/>}
    <div className="workspace" inert={isMobile && mobileOpen}><header className="topbar"><div className="breadcrumb"><button ref={menuButton} className="icon-button mobile-menu" aria-label="Open navigation" aria-expanded={mobileOpen} aria-controls="main-navigation" onClick={() => setMobileOpen(true)}><Menu size={21}/></button><span>{isLab ? 'Laboratory' : 'Arcade'}</span><ChevronRight size={14}/><strong>{pageLabel}</strong></div><div className="backend-status" title={health.error ?? health.data?.env_version}><span className={`status-dot ${health.error ? 'offline' : health.data?.ok ? 'online' : ''}`}/>{health.error ? 'Backend unavailable' : health.data?.ok ? 'Backend connected' : 'Connecting to backend'}{health.error && <button className="link-button" onClick={health.refresh}>Retry</button>}</div></header>
    <main id="main" ref={main} tabIndex={-1} className={isLab ? 'lab-main' : 'arcade-main'}><ScreenBoundary key={location.pathname}><Routes><Route path="/" element={<Lobby/>}/>{(Object.keys(modes) as Mode[]).map(mode => <Route key={mode} path={`/play/${mode}`} element={<GamePage key={mode} mode={mode}/>}/>)}<Route path="/replays" element={<GamePage key="replay" replay/>}/><Route path="/settings" element={<SettingsPage/>}/><Route path="/lab" element={<Overview/>}/><Route path="/lab/training" element={<Training/>}/><Route path="/lab/compare" element={<ComparisonPage/>}/><Route path="/lab/history" element={<HistoryPage/>}/><Route path="/lab/leaderboard" element={<LeaderboardPage/>}/><Route path="/lab/models" element={<ModelsPage/>}/><Route path="/lab/control" element={<ControlPage/>}/><Route path="*" element={<><PageHeader title="This flight went off course." description="That page does not exist."/><Link className="button" to="/">Back to the flight deck</Link></>}/></Routes></ScreenBoundary><footer className="workspace-footer"><span>Aerie / Flappy RL Research Lab</span><Badge>{isLab ? 'Evidence before conclusions' : 'A little lift goes a long way'}</Badge></footer></main>
  </div></div></MotionConfig>;
}
export default function App() { return <HashRouter><SettingsProvider><Shell/></SettingsProvider></HashRouter>; }

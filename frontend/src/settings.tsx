import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { useReducedMotion } from 'motion/react';
import { Field, PageHeader, Panel, ErrorMessage } from './components/ui';
interface Settings { flapKey: string; reducedMotion: boolean }
const defaults: Settings = { flapKey: 'w', reducedMotion: false };
function readSettings(): Settings {
  try {
    const value: unknown = JSON.parse(localStorage.getItem('aerie.settings') ?? 'null');
    if (value && typeof value === 'object' && 'flapKey' in value && 'reducedMotion' in value && typeof value.flapKey === 'string' && /^[a-z0-9]$/.test(value.flapKey) && typeof value.reducedMotion === 'boolean') return { flapKey: value.flapKey, reducedMotion: value.reducedMotion };
  } catch { /* Defaults also work when storage is unavailable. */ }
  return defaults;
}
const SettingsContext = createContext<{ settings: Settings; reduce: boolean; save: (s: Settings) => void; storageError: string | null }>({ settings: defaults, reduce: false, save: () => {}, storageError: null });
export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState(readSettings);
  const [storageError, setStorageError] = useState<string | null>(null);
  const systemReduce = useReducedMotion();
  const reduce = settings.reducedMotion || !!systemReduce;
  useEffect(() => { document.documentElement.dataset.reducedMotion = String(reduce); }, [reduce]);
  const save = (value: Settings) => {
    setSettings(value);
    try { localStorage.setItem('aerie.settings', JSON.stringify(value)); setStorageError(null); }
    catch { setStorageError('These preferences work for this visit, but browser storage is unavailable.'); }
  };
  return <SettingsContext.Provider value={{ settings, reduce, save, storageError }}>{children}</SettingsContext.Provider>;
}
export const useSettings = () => useContext(SettingsContext);
export function SettingsPage() {
  const { settings, save, storageError } = useSettings();
  return <><PageHeader title="Make yourself at home." description="Your flight controls and display preferences, saved on this device."/><div className="narrow"><Panel title="Flight controls"><Field label="Alternative flap key" help="Space and tapping the flight area always work. Choose a letter or number."><select value={settings.flapKey} onChange={e => save({ ...settings, flapKey: e.target.value })}>{'abcdefghijklmnopqrstuvwxyz0123456789'.split('').map(k => <option key={k} value={k}>{k.toUpperCase()}</option>)}</select></Field><p className="muted">Inputs are ignored while you type in a form. Hold does not repeat a flap.</p></Panel><Panel title="Motion"><label className="check-row"><input type="checkbox" checked={settings.reducedMotion} onChange={e => save({ ...settings, reducedMotion: e.target.checked })}/><span>Reduce motion<small>Disable decorative parallax and wing feedback. Your system preference is also respected.</small></span></label><p className="muted">Birds and gates still move so you can follow the flight.</p></Panel>{storageError ? <ErrorMessage message={storageError}/> : <p className="success-text" role="status">Preferences are saved automatically.</p>}</div></>;
}

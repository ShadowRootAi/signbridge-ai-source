export type IconName = 'arrow' | 'camera' | 'text' | 'upload' | 'shield' | 'sound' | 'refresh' | 'close' | 'check' | 'info' | 'play' | 'menu' | 'hand' | 'spark' | 'file' | 'pause';
const paths: Record<IconName, string> = {
  arrow: 'M4 12h15m-6-6 6 6-6 6', camera: 'M8 5 6 8H3v12h18V8h-3l-2-3ZM16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
  text: 'M4 5h16v12H9l-5 4ZM8 9h8M8 13h5', upload: 'M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6',
  shield: 'm12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Zm-4 9 3 3 5-6', sound: 'm11 4-6 5H2v6h3l6 5ZM15 8c3 2 3 6 0 8m3-11c5 4 5 10 0 14',
  refresh: 'M20 7v5h-5M4 17v-5h5M6 6a8 8 0 0 1 13 2M5 16a8 8 0 0 0 13 2', close: 'm6 6 12 12M6 18 18 6',
  check: 'm5 12 4 4L19 6', info: 'M12 11v6m0-10v.1M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  play: 'm9 5 11 7-11 7Z', menu: 'M4 6h16M4 12h16M4 18h16',
  hand: 'M8 12V5a2 2 0 0 1 4 0v6-8a2 2 0 0 1 4 0v9-6a2 2 0 0 1 4 0v9c0 5-3 7-7 7-3 0-5-1-7-4l-3-5a2 2 0 0 1 3-2l2 2',
  spark: 'm12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3Z', file: 'M6 2h8l4 4v16H6Zm8 0v5h4M9 12h6M9 16h6', pause: 'M8 4v16M16 4v16',
};
export function Icon({ name, size = 22, className = '' }: { name: IconName; size?: number; className?: string }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className}><path d={paths[name]} /></svg>;
}

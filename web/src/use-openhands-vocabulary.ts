import { useEffect, useState } from 'react';
import { useOpenHands } from './recognition-release';

export function useOpenHandsVocabularyLabel() {
  const [size, setSize] = useState<number>();
  useEffect(() => {
    if (!useOpenHands) return;
    let mounted = true;
    void import('./openhands-pipeline').then(engine => engine.getOpenHandsPolicy()).then(policy => {
      if (mounted && Array.isArray(policy.supportedLabels)) setSize(policy.supportedLabels.length);
    }).catch(() => { /* Recognition reports load errors when analysis is requested. */ });
    return () => { mounted = false; };
  }, []);
  return size ? `${size} μεμονωμένα νοήματα σε δοκιμή` : 'Μεμονωμένα νοήματα';
}

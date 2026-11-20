import { useEffect, useState } from 'react';
export interface AppEvidence {
  modelVersion?: string;
  labels: string[];
  historicalV2: { knownN: number; acceptedKnownCorrect: number; unknownN: number; falseAcceptedUnknown: number; perKnownSign?: { label: string; n: number; acceptedCorrect: number }[] };
  currentV4?: { status?: string; trackingPolicyVersion?: string; independentHumanEvaluation?: boolean };
}
// Frozen B measurements. They are explicitly historical, never current V4 accuracy.
export const historicalEvidence: AppEvidence = {
  modelVersion: 'signbridge-open-set-v2-B_tcn_diverse_other', labels: ['ΓΕΙΑ', 'ΕΥΧΑΡΙΣΤΩ', 'ΕΝΤΑΞΕΙ'],
  historicalV2: { knownN: 79, acceptedKnownCorrect: 59, unknownN: 50, falseAcceptedUnknown: 24 },
  currentV4: { independentHumanEvaluation: false },
};
export function useEvidence() {
  const [evidence, setEvidence] = useState<AppEvidence>(historicalEvidence);
  const [live, setLive] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch('/reports/app-evidence.json', { signal: controller.signal, cache: 'no-store' }).then(async response => {
      if (!response.ok) return;
      const value = await response.json() as AppEvidence;
      const h = value.historicalV2;
      if (Array.isArray(value.labels) && value.labels.every(label => typeof label === 'string') && h
        && [h.knownN, h.acceptedKnownCorrect, h.unknownN, h.falseAcceptedUnknown].every(n => Number.isSafeInteger(n) && n >= 0)
        && h.knownN > 0 && h.unknownN > 0 && h.acceptedKnownCorrect <= h.knownN && h.falseAcceptedUnknown <= h.unknownN) {
        setEvidence(value); setLive(true);
      }
    }).catch(() => {});
    return () => controller.abort();
  }, []);
  return { evidence, live };
}
export function displaySign(value: string) {
  const names: Record<string, string> = {
    ΓΕΙΑ: 'Γεια', ΕΥΧΑΡΙΣΤΩ: 'Ευχαριστώ', ΕΝΤΑΞΕΙ: 'Εντάξει',
    ΝΑΙ: 'Ναι', ΟΧΙ: 'Όχι', ΚΑΛΟ: 'Καλό', ΦΥΣΙΚΑ: 'Φυσικά',
  };
  return names[value] ?? value;
}

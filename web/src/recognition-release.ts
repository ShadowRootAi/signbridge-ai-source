const loopback = ['localhost', '127.0.0.1'].includes(window.location.hostname);
export const publicOpenHands = import.meta.env.VITE_SIGNBRIDGE_PUBLIC_OPENHANDS === '1';
export const localOpenHands = import.meta.env.DEV
  && loopback
  && new URLSearchParams(window.location.search).get('model') === 'openhands';
export const useOpenHands = publicOpenHands || localOpenHands;
export const sequencePilot = import.meta.env.VITE_SIGNBRIDGE_CONTINUOUS === '1' || localOpenHands;
const localAssistantBuild = import.meta.env.VITE_SIGNBRIDGE_LOCAL_ASSISTANT === '1' && loopback;
export const localAssistantEnabled = localAssistantBuild
  && new URLSearchParams(window.location.search).get('assistant') === '1';
export const personalTeaching = localAssistantBuild && useOpenHands;
export const judgeDemo = import.meta.env.VITE_SIGNBRIDGE_JUDGE_DEMO === '1' && useOpenHands
  && loopback
  && new URLSearchParams(window.location.search).get('judge') === '1';

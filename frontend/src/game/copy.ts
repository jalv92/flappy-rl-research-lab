// Grammar only; match outcomes and distances always come from the server.
export const pilotVerb = (label: string, verb: 'win' | 'lead') => `${label} ${label === 'You' ? verb : `${verb}s`}`;

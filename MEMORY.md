# HarMoCAP — Project Memory

> Last updated: 2026-07-18. Session: gpt-5.6-terra → k3.

## Status

Beacon ecosystem re-architecture: **COMPLETE** (26/26 Kanban cards done). F0-F4 + F6 executed across 4 repos.

## Active repos

| Repo | Branch | Head | Role |
|------|--------|------|------|
| harmonic-weaver | main | 6d84203 | Routing engine + Stage WS + patchbay + rehearsal harness |
| beacon-spatial | main | de2768c | 13-band binaural + nature layer + modulation presets |
| harmonic-shaper | main | 7f1dfc2 | Standalone additive synth 32-voice |
| harmonic-beacon-tines | main | — | Archived |

## Live test findings (S13)

- Crowd→beacon master routing WORKS with live camera.
- Shaper voice routes (5 harmonics) NOT firing — cause not fully isolated.
- CUDA crash on RTX 2060 with cu126 torch (fallback to cu124 or CPU needed).
- Engine source lease: expires permanently after 2500ms, no auto-recovery.

## Quick-start for /new

Load BITACORA.md (entries S12–S13), then the rehearsal harness in harmonic-weaver/rehearsal/.

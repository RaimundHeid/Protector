## Unsuccessful Changes

### Limit cut-node quiet move reduction to MGS_REST stage only (2026-04-23)

**Change:** In `search.c`, added `stage == MGS_REST` to the cut-node quiet move reduction condition,
so killer moves (MGS_KILLER_MOVES stage) were no longer reduced by 2 plies in cut nodes.

```c
// Before:
if (cutNode && quietMove) {
    variableDepth -= 2048;
}
// After:
if (cutNode && quietMove && stage == MGS_REST) {
    variableDepth -= 2048;
}
```

**Result:** 501 games at 10+0.1 TC, W=143 L=121 D=237, score=52.2%, LOS=63.0%, LLR=+0.523.
No SPRT decision (limits: PASS>+0.875, FAIL<-0.629). LOS < 66.7% → REVERTED.

**Note:** The positive LLR/LOS trend suggests the change may have a small beneficial effect,
but it was not statistically significant over 500 games.

## Unsuccessful Changes

### Dynamic depth reduction after alpha improvement (Stockfish Step 20) (2026-05-03)

**Change:** Added `effectiveRestDepth` variable before the move loop; after an alpha improvement without beta cutoff, reduced it by 2 for remaining moves (mirrors Stockfish Step 20: `if depth > 2 && depth < 15: depth -= 2`).

```c
// Before loop:
int effectiveRestDepth = restDepth;

// pvDepth/reducedDepth used effectiveRestDepth - 1 instead of restDepth - 1

// After "if (best >= beta) { break; }":
if (effectiveRestDepth > 2 && effectiveRestDepth < 15 &&
    abs(best) <= -VALUE_ALMOST_MATED) {
    effectiveRestDepth -= 2;
}
```

**Result:** 154 games at 10+0.1 TC, W=30 D=70 L=54, score=42.2%, LOS=0.4%, LLR=−0.755.
LLR crossed FAIL bound (< −0.629) at 154 games → **REVERTED**.

**Note:** Clearly negative result. Reducing search depth for remaining moves after alpha improvement
hurts Protector significantly — possibly because Protector's shallower searches are already less
accurate than Stockfish's, so further depth reduction degrades move selection more than it saves time.

---

### Tighten quiet move futility count to Stockfish formula (2026-05-03)

**Change:** Replaced Protector's quiet move futility count threshold with Stockfish's exact formula.

```c
// Before:
if (numMovesPlayed >= (improving ? 79 : 45) * (3 + restDepth * restDepth) / 64) {
    continue;
}
// After:
if (numMovesPlayed >= (improving ? 3 + restDepth * restDepth : (3 + restDepth * restDepth) / 2)) {
    continue;
}
```

Stockfish's formula is 22–44% more aggressive (prunes earlier), especially when not improving.

**Result:** 245 games at 10+0.1 TC, W=63 D=113 L=69, score=48.8%, LOS=30.1%, LLR=−0.275.
LOS at N=200 was ≈42% < 60% → early abort. **REVERTED.**

**Note:** Stockfish's formula cuts quiet moves far too aggressively for Protector's playing style.
Protector's higher thresholds appear to be deliberately tuned. Do not apply Stockfish's raw
futility_move_count formula without scaling up.

---

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

---

### Extend static pruning: remove hashmove==NO_MOVE restriction, deepen depth 4→6 (2026-05-01)

**Change:** Decoupled static pruning from the `hashmove == NO_MOVE` condition and extended the
depth limit from 4 to 6. This mirrors Stockfish's approach of applying static pruning unconditionally.

```c
// Before (inside `if (pvNode == FALSE && inCheck == FALSE && hashmove == NO_MOVE)`):
if (restDepth <= 4) {
    const int margin = 22 + 44 * restDepth - (isImproving(variation) ? 24 : 0);
    if (getStaticValue(variation) - margin >= beta) { return beta; }
}

// After (separate block, no hashmove restriction):
if (pvNode == FALSE && inCheck == FALSE && restDepth <= 6 && abs(beta) <= -VALUE_ALMOST_MATED) {
    const int margin = 22 + 44 * restDepth - (isImproving(variation) ? 24 : 0);
    if (getStaticValue(variation) - margin >= beta) { return beta; }
}
```

**Result:** 1000 games at 10+0.1 TC, W=274 D=464 L=262, score=50.60%, LOS=64.8%, LLR=−0.129.
No SPRT decision (LLR stayed between bounds). LOS < 66.7% → REVERTED.

**Note:** Score was not meaningfully above 50%. The change had highly volatile LLR (peaked at +0.534
at 146 games, dropped to −0.437 at 911 before recovering to −0.129 at finish). Likely neutral or
very small effect; extending depth 4→6 may over-prune.


---

### Increase LMR by 800 (approx 0.8 plies) when not improving (2026-05-01)

**Change:** In `search.c`, increased the Late Move Reductions (LMR) by 800 units if the current position is not improving (`improving == FALSE`).

```c
// After:
int reductions = log1024[restDepth] * log1024[numMovesPlayed] / 2176 + (cutNode ? 2048 : 0);
if (improving == FALSE) {
    reductions += 800;
}
```

**Result:** 558 games at 10+0.1 TC, W=149 L=152 D=257, score=49.7%, LLR=-0.187.
Match stopped early. Trend suggests neutral or slightly negative impact. Reverted.

---

### Small ProbCut: TT lower-bound early return (2026-05-02)

**Change:** After the real TT lookup (before IIR/NMP), added an early return when the TT has a
lower-bound entry with value ≥ beta + 160 at depth ≥ restDepth − 4. Margin 160 is Stockfish's
416 scaled to Protector's evaluation units (~38%). This mirrors Stockfish's "Step 12" heuristic.

```c
// After TT lookup, before updateGains:
if (pvNode == FALSE && excludeMove == NO_MOVE && bestTableHit != NULL &&
    abs(beta) <= -VALUE_ALMOST_MATED &&
    getHashentryFlag(bestTableHit) == HASHVALUE_LOWER_LIMIT &&
    getHashentryImportance(bestTableHit) >= restDepth - 1) {
    const int smallProbCutHashValue = calcEffectiveValue(getHashentryValue(bestTableHit), ply);
    const int smallProbCutBeta = beta + 160;
    if (smallProbCutHashValue >= smallProbCutBeta && abs(smallProbCutHashValue) <= -VALUE_ALMOST_MATED) {
        best = smallProbCutBeta;
        goto storeResult;
    }
}
```

**Result:** 899 games at 10+0.1 TC, W=231 D=434 L=234, score=49.8%, LLR=−0.459, LOS=49.7%.
LLR oscillated near zero throughout (peak +0.131 at 448g, then drifted negative). LOS < 66.7% → REVERTED.

**Note:** Neutral result. The threshold of 160 may be slightly too low (triggering too often on
marginally reliable entries) or too high (rarely triggering). The Stockfish equivalent fires in a
richer context with correction history and ttPv flags that Protector lacks.

---

### doShallowerSearch in LMR re-search (2026-05-02)

**Change:** In the LMR full-depth re-search, use `pvDepth - 1` instead of `pvDepth` when the value
only marginally exceeds the best score seen so far (mirrors Stockfish Step 17 `doShallowerSearch`).

```c
// Before:
if (value > alpha) {
    value =
        -searchBest(variation, -alpha - 1, -alpha, ply + 1, pvDepth, &bestReply, FALSE, FALSE, NO_MOVE);
}
// After:
if (value > alpha) {
    const int fullDepth = (value < best + 10) ? max(reducedDepth + 1, pvDepth - 1) : pvDepth;
    value = -searchBest(variation, -alpha - 1, -alpha, ply + 1, fullDepth, &bestReply, FALSE, FALSE,
                        NO_MOVE);
}
```

**Result:** 328 games at 10+0.1 TC, W=83 D=147 L=98, score=47.7%, LOS=13.2%, LLR=−0.568.
LOS below 60% throughout (≈14% at 200 games) → REVERTED.

**Note:** Clearly negative result. Reducing the re-search depth even by one ply when the LMR
value only marginally beats best appears to lose accuracy rather than save time.

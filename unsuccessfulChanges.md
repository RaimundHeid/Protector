## Unsuccessful Changes

### Stockfish-style Negative Extensions in Singular Search (2026-05-24)

**Change:** Scaled negative extensions inside the singular search verification block of `searchBest`. Instead of always reducing by 1 ply (`extensions -= 1024`) when `cutNode` is true, we applied Stockfish's scaling: a 3-ply reduction when the hash entry is assumed to fail high (`hashEntryValue >= beta`) and a 2-ply reduction when `cutNode` is true.

```c
// Before:
if (excludeValue < limitValue) {
    extensions += 1024;
} else if (excludeValue >= beta && abs(excludeValue) <= -VALUE_ALMOST_MATED) {
    best = excludeValue;
    goto storeResult;
} else if (cutNode) {
    extensions -= 1024;
}

// After:
if (excludeValue < limitValue) {
    extensions += 1024;
} else if (excludeValue >= beta && abs(excludeValue) <= -VALUE_ALMOST_MATED) {
    best = excludeValue;
    goto storeResult;
} else if (hashEntryValue >= beta) {
    extensions -= 3072;
    // 3 plies
} else if (cutNode) {
    extensions -= 2048;
    // 2 plies
}
```

**Result:** Negative/Neutral. Games: 1000, W-L-D: 236-230-534, LLR: -0.0369, LOS: 60.90%. Match finished 1000 games without crossing SPRT bounds. LOS was below 2/3. **REVERTED.**

**Note:** Although the change showed a slightly positive score (+6 games net), it did not achieve a statistically significant improvement over 1000 games, and final LOS was below the 2/3 threshold. Scaling up negative extensions in Protector might be too aggressive for its search framework, or the reduction units may require tuning specific to Protector's evaluation scale.

---

### Implementing Internal Iterative Reductions (IIR) (2026-05-23)

**Change:** Implemented Internal Iterative Reductions (IIR) in `searchBest`: if a node has no transposition table move (`hashmove == NO_MOVE`) and the remaining depth is high (`restDepth >= 6`), we reduce the search depth by 1 ply (`restDepth--`). This was intended to save nodes at subtrees with sub-optimal move ordering.

```c
// Forward declaration & definition:
static int searchBest(Variation *variation, int alpha, int beta, const int ply, int restDepth, Move *bestMove, ...);

// Inside searchBest, right after TT probe:
if (hashmove == NO_MOVE && restDepth >= 6) {
    restDepth--;
}
```

**Result:** Negative. Games: 300, W-L-D: 75-78-147, LLR: -0.1456, LOS: 40.40%. Aborted early at N=300 (LOS < 60.0% at N>=300 milestone). **REVERTED.**

**Note:** Implementing IIR in Protector degrades playing strength. Although IIR reduces the search time in nodes without a hash move, the 1-ply reduction at depth >= 6 loses too much tactical accuracy in those nodes. In Protector's search framework, maintaining full-depth searches in TT-miss nodes is necessary for search robustness.

---

### Widening Aspiration Window bounds from [4, 6] to [6, 12] (2026-05-23)

**Change:** Widened the bounds of the aspiration window from `min(6, max(4, ...))` to `min(12, max(6, ...))`. The rationale was that an aspiration window of 4 to 6 centipawns is extremely narrow, leading to frequent fail highs/lows and expensive full re-searches. Widening it to `[6, 12]` was intended to reduce aspiration re-searches and save nodes.

```c
// Before:
aspirationWindow = min(6, max(4, (abs(iv1 - iv2) + abs(iv2 - iv3)) / 2));
// After:
aspirationWindow = min(12, max(6, (abs(iv1 - iv2) + abs(iv2 - iv3)) / 2));
```

**Result:** Negative. Games: 206, W-L-D: 48-65-93, LLR: -0.4909, LOS: 6.40%. Aborted early at N=206 (LOS < 60.0% at N>=200 milestone). **REVERTED.**

**Note:** Widening the aspiration window degrades Protector's playing strength. Although widening the window avoids some re-searches, it reduces the overall search pruning efficiency (as wider windows search more nodes on average), which outweighs the time saved from avoided re-searches. Keep the tight [4, 6] window.

---

### Restricting PV-node singular extensions to restDepth >= 5 (2026-05-23)

**Change:** Increased the depth threshold for PV-node singular extension checks from `4` to `5` (`restDepth >= (pvNode ? 5 : 8)`). The rationale was that singular extension checks at `restDepth = 4` run a verification search at `restDepth / 2 = 2` plies, which is highly unstable and noisy. Raising the threshold to `5` was intended to save nodes.

```c
// Before:
if (movesAreEqual(currentMove, hashmove) && excludeMove == NO_MOVE && restDepth >= (pvNode ? 4 : 8) &&
// After:
if (movesAreEqual(currentMove, hashmove) && excludeMove == NO_MOVE && restDepth >= (pvNode ? 5 : 8) &&
```

**Result:** Negative. Games: 224, W-L-D: 53-59-112, LLR: -0.2194, LOS: 38.70%. Aborted early at N=224 (LOS < 60.0% at N>=200 milestone). **REVERTED.**

**Note:** Restricting singular extensions to `restDepth >= 5` in PV nodes degrades playing strength. Even though verification searches at depth 4 are shallow, singular extensions are crucial for tactical safety in Protector's search, and pruning them leads to tactical oversights. Do not retry.

---

### Eval-margin-based additional NMP depth reduction (2026-05-17)

**Change:** Added an extra 0–3 plies of NMP depth reduction when the static evaluation significantly exceeds beta. The rationale was that when the position is already far above beta, even a shallower null move verification suffices — the opponent would need a correspondingly large swing with their free move.

```c
// Before:
const int newDepth = restDepth - 5 - restDepth / 4;
// After:
const int evalExcess = min(3, (getStaticValue(variation) - beta) / 200);
const int newDepth = restDepth - 5 - restDepth / 4 - evalExcess;
```

**Result:** Negative. Score of proHigh vs Stockfish 16 classic eval: 125 - 101 - 122 [0.534] 348 games, Elo: +24.0 ± 29.5, LOS: 94.5%, DrawRatio: 35.1%. **REVERTED.**

**Note:** The extra reduction fires in positions that already cleared the `staticValue >= beta` bar, so it compounds an existing aggressive reduction (`R = 5 + depth/4`) without a sufficiently reliable signal. A position 400 cp above beta can still have deep tactical refutations that require the full verification depth to find. The improvement is not statistically significant and the directional effect is unclear. Do not retry.

---

### LMR extended to MGS_BAD_CAPTURES stage (2026-05-17)

**Change:** Extended the `reduce` condition in the main move loop to include `MGS_BAD_CAPTURES` in addition to `MGS_REST`, so that bad captures (negative SEE) played late in the move list are searched at reduced depth like late quiet moves.

```c
// Before:
const bool reduce = numMovesPlayed > 1 && reductions >= 1024 && extensions == 0 && inCheck == FALSE &&
                    restDepth >= 3 && stage == MGS_REST;
// After:
const bool reduce = numMovesPlayed > 1 && reductions >= 1024 && extensions == 0 && inCheck == FALSE &&
                    restDepth >= 3 && (stage == MGS_REST || stage == MGS_BAD_CAPTURES);
```

**Result:** Negative. Elo difference: 24.7 ± 19.6, LOS: 99.3%, DrawRatio: 40.1%. **REVERTED.**

**Note:** Bad captures arrive after all quiet moves, so `numMovesPlayed` is large and `reductions` is large when they are reached, causing them to be reduced by 2–4 plies. Although bad captures rarely are the best move, they include material sacrifices that lead to dynamic combinations (pins, discoveries, mating attacks) — precisely the tactical patterns where SEE is unreliable. Reducing them causes these lines to be searched shallowly and the re-search safety net is not triggered often enough to compensate. The LOS of 99.3% confirms this is a clear regression, not noise. Do not retry.

---

### Improving-conditional NMP depth reduction (2026-05-14)

**Change:** Used `isImproving(variation)` to modulate the null move depth reduction: aggressive `R = 5 + depth/4` when improving, conservative `R = 4 + depth/4` when not improving. The rationale was that a declining static eval trend suggests a temporarily inflated eval, warranting deeper verification.

```c
// Before:
const int newDepth = restDepth - 5 - restDepth / 4;
// After:
const int newDepth = restDepth - (isImproving(variation) ? 5 : 4) - restDepth / 4;
```

**Result:** Negative. **REVERTED.**

**Note:** NMP already requires `staticValue >= beta`, so the position is well above beta regardless of trend. The one-ply difference in verification depth did not meaningfully improve NMP reliability. Do not retry.

---

### Depth-proportional history malus for failed quiet moves (2026-05-14)

**Change:** Applied a depth-proportional `freq` malus to all quiet moves that failed to produce a cutoff. All quiet moves received `freq += bonus + malus` (where `malus = bonus * restDepth / 2`) in the loop; the cutoff move's `freq` was corrected back down by `malus` after the loop in a single statement.

```c
// Before:
for (int i = 0; i < quietMoveCount; i++) {
    variation->moveHistory[ply][quietMoveIndex[i]].freq += bonus;
}
// After:
const int malus = bonus * restDepth / 2;
for (int i = 0; i < quietMoveCount; i++) {
    variation->moveHistory[ply][quietMoveIndex[i]].freq += bonus + malus;
}
// ... then inside the bestMove block:
variation->moveHistory[ply][historyIndex(*bestMove, position)].freq -= malus;
```

**Result:** Negative. **REVERTED.**

**Note:** The per-ply history already penalizes failed quiet moves implicitly via the `succ/freq` ratio. Adding an explicit depth-scaled malus appears to over-penalize moves and degrade move ordering, possibly because the existing ratio-based formula is already well-calibrated for the historyPerPly system.

---

### SEE-scaled quiet move pruning at all depths (re-applied to historyPerPly branch) (2026-05-05)

**Change:** Replaced `if (restDepth < 4 && seeMove(position, currentMove) < 0)` with a depth-independent quadratic SEE threshold, mirroring Stockfish's `!see_ge(move, -25 * lmrDepth²)`. The `seeLmrDepth` naturally incorporates the per-ply history bonus already in `reductions`.

```c
// Before:
if (restDepth < 4 && seeMove(position, currentMove) < 0) {
    continue;
}
// After:
const int seeLmrDepth = max(1, restDepth - 1 - reductions / 1024);
if (seeMove(position, currentMove) < -7 * seeLmrDepth * seeLmrDepth) {
    continue;
}
```

**Result:** 254 games at 10+0.1 TC, W=68 D=112 L=74, score=47.5% at n=200, LOS=17.2%, LLR=−0.665 at n=200.
LOS early abort triggered at n=200 (LOS=17.2% < 60%). LLR also crossed FAIL bound (−0.665 < −0.629). **REVERTED.**

**Note:** This change was previously a PASS on the `reducedNegaScout` branch (af1125f, LOS=86.9%), but fails on the `historyPerPly` branch. The per-ply history already adjusts LMR reductions — using `reductions/1024` to further scale the SEE threshold over-prunes moves with bad history that should still be searched due to the richer history context. The two mechanisms interact negatively.

---

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

---

### Depth-proportional history bonus (restDepth²) for freq/succ updates (2026-05-15)

**Change:** Scaled the history bonus by `min(restDepth * restDepth, 400)` so that deeper cutoffs contribute proportionally more to the `succ/freq` ratio. The rationale was that shallow restDepth searches dominate history updates due to exponential node counts, making the `plyScore = 16000 * succ / freq − 8000` noisy. A depth-10 cutoff would contribute 100× more than a depth-1 cutoff, ideally improving LMR calibration at zero per-node cost (the multiply is only in the update path).

```c
// Before:
const int bonus = (pvNode ? 2 : 1);
variation->moveHistory[ply][quietMoveIndex[i]].freq += bonus;
// ...
variation->moveHistory[ply][historyIndex(*bestMove, position)].succ += bonus;

// After:
const int bonus = min(restDepth * restDepth, 400) * (pvNode ? 2 : 1);
variation->moveHistory[ply][quietMoveIndex[i]].freq += bonus;
// ...
variation->moveHistory[ply][historyIndex(*bestMove, position)].succ += bonus;
```

**Result:** Negative. **REVERTED.**

**Note:** The depth-squared multiplier causes the accumulated `freq` and `succ` values to be dominated by the rare high-depth updates, effectively discarding the signal from the many shallow nodes. The flat bonus, despite being dominated numerically by shallow data, appears to be the right calibration for the historyPerPly system — the existing `succ/freq` ratio already works well because both numerator and denominator are shaped similarly by search volume.

---

### Extend SEE-based quiet move pruning from restDepth < 4 to restDepth < 6 (2026-05-15)

**Change:** Raised the depth threshold in the `isSpecialMove`/SEE pruning branch of the optimistic futility cuts, so that quiet moves with negative SEE are also pruned at depths 4 and 5. The existing `isSpecialMove` guard (checks and passed-pawn advances) was left intact.

```c
// Before:
if ((cheapPrune || restDepth < 4) && isSpecialMove(position, currentMove) == FALSE) {
// After:
if ((cheapPrune || restDepth < 6) && isSpecialMove(position, currentMove) == FALSE) {
```

**Result:** Negative. **REVERTED.**

**Note:** Although losing-SEE quiet moves are almost never best, the SEE pruning at depths 4–5 appears to interact badly with Protector's existing move ordering and LMR framework. The per-ply history already steers the search away from bad moves via reductions; adding SEE pruning on top over-prunes and loses accuracy.

---

### Improving-conditional singular extension verification depth (2026-05-15)

**Change:** Used the `improving` flag (already in scope) to shorten the singular extension verification search by one ply when the position is not improving. The rationale: a declining static eval trend means fewer viable alternatives, so the hash move is more likely to be genuinely singular, and a shallower verification suffices.

```c
// Before:
const int excludeValue = searchBest(variation, limitValue - 1, limitValue, ply, restDepth / 2,
                                    &bestReply, FALSE, cutNode, hashmove);
// After:
const int verifDepth = improving ? restDepth / 2 : max(1, restDepth / 2 - 1);
const int excludeValue = searchBest(variation, limitValue - 1, limitValue, ply, verifDepth,
                                    &bestReply, FALSE, cutNode, hashmove);
```

**Result:** Negative. **REVERTED.**

**Note:** The `improving` flag reflects the static eval trend, not the actual branching factor of the verification search. In practice, non-improving positions still have many candidate alternatives that require deep verification to rule out — the shallower search produces more false-positive extensions, which costs more than it saves.

---

### Null window for null move verification search (2026-05-15)

**Change:** Replaced the full-window `(alpha, beta)` verification search with a null window `(beta-1, beta)`. The verification's only purpose is to determine whether the position scores ≥ beta; a null window makes the same binary decision without exploring moves scoring between alpha and beta−1.

```c
// Before:
searchBest(variation, alpha, beta, ply, newDepth, &bestReply, FALSE, FALSE, NULLMOVE) >= beta
// After:
searchBest(variation, beta - 1, beta, ply, newDepth, &bestReply, FALSE, FALSE, NULLMOVE) >= beta
```

**Result:** Negative. **REVERTED.**

**Note:** The verification search runs with `excludeMove = NULLMOVE`, which disables the transposition table probe at the root of that call. Without TT cutoffs at the top level, the narrower window removes the alpha-floor that helped prune branches in the full-window version, resulting in more nodes rather than fewer.

---

### Defer staticValue/improving computation to pvNode==FALSE && inCheck==FALSE (2026-05-15)

**Change:** Wrapped the unconditional `getStaticValue()`/`isImproving()` calls (previously computed for every node) in a `pvNode == FALSE && inCheck == FALSE` guard, since both values are only consumed inside ProbCut and the optimistic futility cuts — both of which are already guarded by that condition. For PV nodes and in-check nodes without a TT entry, this avoids the NNUE finalization call entirely.

```c
// Before:
const int staticValue = getStaticValue(variation);
const bool improving = isImproving(variation) || staticValue >= beta;

// After:
int staticValue = 0;
bool improving = FALSE;
if (pvNode == FALSE && inCheck == FALSE) {
    staticValue = getStaticValue(variation);
    improving = isImproving(variation) || staticValue >= beta;
}
```

**Result:** 720 games at 10+0.1 TC, W=243 D=298 L=179, score=54.4%, Elo=+31.0 ± 19.4, LOS=99.9%. Reference value Elo +45. **REVERTED.**

**Note:** The change is theoretically sound — PV and in-check nodes with a TT miss do pay for an NNUE evaluation they don't use. However, the reference baseline is Elo +45, and this change only achieved +31. The improvement exists but is below the bar needed to be counted as a real gain. In practice, TT hit rates are high, so the NNUE call at line 662 is rarely expensive (the cached path is cheap); the savings are smaller than expected.

---

### History-score-informed move-count pruning threshold (2026-05-17)

**Change:** Moved the `plyScore` computation (from `succ/freq` history ratio) to before the optimistic futility cut and used it to adapt the move-count threshold. Moves with negative history were pruned sooner; moves with positive history survived longer. The `plyScore / 500` adjustment shifted the base threshold (45 or 79) by up to ±16.

```c
// Before:
if (pvNode == FALSE && inCheck == FALSE && quietMove && best > VALUE_ALMOST_MATED) {
    const bool cheapPrune = (numMovesPlayed >= (improving ? 79 : 45) * (3 + restDepth * restDepth) / 64) || ...;
    ...
}
if (quietMove) {
    const MoveHistoryEntry *histEntry = &variation->moveHistory[ply][historyIndexMove];
    const int plyScore = (int)(16000LL * (histEntry->succ + 1) / (histEntry->freq + 2) - 8000LL);
    reductions = max(0, reductions - plyScore / 8);
}

// After:
int plyScore = 0;
if (quietMove) {
    const MoveHistoryEntry *histEntry = &variation->moveHistory[ply][historyIndexMove];
    plyScore = (int)(16000LL * (histEntry->succ + 1) / (histEntry->freq + 2) - 8000LL);
    reductions = max(0, reductions - plyScore / 8);
}
if (pvNode == FALSE && inCheck == FALSE && quietMove && best > VALUE_ALMOST_MATED) {
    const int adjustedBase = (improving ? 79 : 45) + plyScore / 500;
    const bool cheapPrune = (numMovesPlayed >= adjustedBase * (3 + restDepth * restDepth) / 64) || ...;
    ...
}
```

**Result:** Negative. **REVERTED.**

**Note:** The per-ply history already shapes LMR reductions for quiet moves. Feeding the same signal into the move-count pruning threshold creates a double-penalty for bad-history moves (earlier pruning AND more reduction if not pruned) and a double-benefit for good-history moves. This over-correlates two mechanisms that were tuned independently, degrading overall accuracy.

---

### ProbCut hash upper-bound skip (2026-05-16)

**Change:** Computed `probCutBeta` before the outer ProbCut guard and folded a TT upper-bound check into that condition. When `bestTableHit` has `HASHVALUE_UPPER_LIMIT` with importance >= `restDepth - 1` and proven value < `probCutBeta`, the entire ProbCut section (capture move generation, SEE filtering, qsearch, and deep search) is skipped using four cheap comparisons on the already-loaded hash entry.

```c
// Before:
if (pvNode == FALSE && inCheck == FALSE && restDepth >= 5 && excludeMove == NO_MOVE &&
    abs(beta) <= -VALUE_ALMOST_MATED) {
    const int probCutBeta = min(-VALUE_ALMOST_MATED, beta + 200);
    // ... full ProbCut body

// After:
const int probCutBeta = min(-VALUE_ALMOST_MATED, beta + 200);
if (pvNode == FALSE && inCheck == FALSE && restDepth >= 5 && excludeMove == NO_MOVE &&
    abs(beta) <= -VALUE_ALMOST_MATED &&
    (bestTableHit == 0 ||
     getHashentryFlag(bestTableHit) != HASHVALUE_UPPER_LIMIT ||
     getHashentryImportance(bestTableHit) < restDepth - 1 ||
     calcEffectiveValue(getHashentryValue(bestTableHit), ply) >= probCutBeta)) {
    // ... full ProbCut body
```

**Result:** Negative. **REVERTED.**

**Note:** The TT upper-bound entries at depth >= `restDepth - 1` that also fall below `probCutBeta` appear to be rare enough in practice that the skip fires infrequently. ProbCut itself already requires `restDepth >= 5`, and by that point most positions either lack a TT entry at sufficient depth or the upper bound is above `probCutBeta`. The overhead of the four comparisons on every eligible node slightly outweighs the occasional ProbCut skip.

---

### Restricting singular extensions to PV nodes only (2026-05-24)

**Change:** Restricted the singular extension check to PV nodes only, avoiding verification searches in non-PV subtrees (which are typically cut nodes or all nodes).

```c
// Before:
if (movesAreEqual(currentMove, hashmove) && excludeMove == NO_MOVE && restDepth >= (pvNode ? 4 : 8) &&
// After:
if (pvNode && movesAreEqual(currentMove, hashmove) && excludeMove == NO_MOVE && restDepth >= 4 &&
```

**Result:** Negative. Games: 200, W-L-D: 56-54-90, LLR: +0.0147, LOS: 57.60%. Aborted early at N=200 milestone because LOS (57.60%) dropped below the 60.0% threshold. **REVERTED.**

**Note:** Restricting singular extensions to PV nodes only degrades playing strength or is at best neutral. Although skipping singular extensions in non-PV nodes saves nodes and computation time, singular extensions in non-PV nodes are still important for tactical safety in deep subtrees (where non-PV nodes can be reached at depth >= 8), and disabling them leads to tactical oversights that negate the search speedup.

---

### Raising non-PV singular extension depth threshold from 8 to 10 (2026-05-24)

**Change:** Raised the minimum depth check for singular extensions in non-PV nodes from `8` to `10` (`restDepth >= (pvNode ? 4 : 10)`).

```c
// Before:
if (movesAreEqual(currentMove, hashmove) && excludeMove == NO_MOVE && restDepth >= (pvNode ? 4 : 8) &&
// After:
if (movesAreEqual(currentMove, hashmove) && excludeMove == NO_MOVE && restDepth >= (pvNode ? 4 : 10) &&
```

**Result:** Negative. Games: 200, W-L-D: 44-56-100, LLR: -0.3896, LOS: 11.50%. Aborted early at N=200 milestone because LOS (11.50%) dropped below the 60.0% threshold. **REVERTED.**

**Note:** Raising the threshold for non-PV singular extensions from 8 to 10 degrades playing strength. Although it avoids costly verification searches at depths 8 and 9, these singular extensions are still essential for safety in cut subtrees at those depths. Restricting them leads to tactical blindspots in deep non-PV lines.

---

### Quiet move history decaying across iterations (2026-05-24)

**Change:** Decayed accumulated quiet move histories by dividing `freq` and `succ` counters by 2 at the start of each search iteration step (depth > 1) to prevent history sluggishness.

```c
if (variation->iteration > 1) {
    for (int p = 0; p < MAX_DEPTH_ARRAY_SIZE; p++) {
        for (int h = 0; h < HISTORY_SIZE; h++) {
            variation->moveHistory[p][h].freq /= 2;
            variation->moveHistory[p][h].succ /= 2;
        }
    }
}
```

**Result:** Negative. Games: 200, W-L-D: 44-52-104, LLR: -0.2839, LOS: 20.70%. Aborted early at N=200 milestone because LOS (20.70%) dropped below the 60.0% threshold. **REVERTED.**

**Note:** Decaying move histories by 2 degrades Protector's playing strength. Even though decaying histories prevents sluggishness in theory, in practice, the frequency-success ratio relies on the accumulation of absolute sample sizes across all search steps to remain statistically stable and smooth. Artificially halving the counts makes the move sorting score overly noisy and unstable at deep iterations, which degrades move ordering.




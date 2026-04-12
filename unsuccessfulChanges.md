# Unsuccessful Changes to Protector search.c

Changes that were tested in engine matches and showed no improvement (LOS below threshold).
Do not repeat these in future sessions.

---

## Followup-move LMR bonus
**Date:** 2026-04-11 (session 1)
**LOS:** 38.6% at 199 games (threshold: 50% at 200 games)

Extended the counter-move LMR reduction bonus to followup moves (`killerMove5/6`).
Counter-moves (`killerMove3/4`) already receive a 1-ply LMR reduction bonus (searched deeper).
Giving followup moves the same bonus caused over-searching of moves that don't merit full depth.

```c
// Changed effectiveReduction to:
const bool followupMove = movesAreEqual(currentMove, fm1) || movesAreEqual(currentMove, fm2);
const int effectiveReduction = max(0, reduction - ((counterMove || followupMove) ? DEPTH_RESOLUTION : 0));
```

---

## Double singular extension
**Date:** 2026-04-11 (session 2)
**LOS:** 28.4% at 112 games (threshold: 40% at 100 games)

When a move was extremely singular (all alternatives fail by >20 cp below the singular window),
extended by `2 * DEPTH_RESOLUTION` instead of `DEPTH_RESOLUTION`. Caused search tree explosion
in positions with already-extended singular moves, hurting performance under time pressure.

```c
// Changed the singular extension block to:
if (excludeValue < limitValue - 20) {
    extension = 2 * DEPTH_RESOLUTION;
} else if (excludeValue < limitValue) {
    extension = DEPTH_RESOLUTION;
}
```

---

## No LMR on hash move
**Date:** 2026-04-11 (session 2)
**LOS:** 7.9% at 102 games (threshold: 40% at 100 games)

Added `variation->plyInfo[ply].isHashMove == FALSE` to the history-pruning (LMR trigger)
condition, preventing the hash move from ever being reduced. Hypothesis: saving the wasted
reduced search when the hash move (usually the best move) beats alpha.

Result: badly hurt playing strength. The LMR on the hash move serves a useful purpose at
cut nodes — quickly verifying the hash move fails before moving on, rather than wasting a
full-depth search on a position that will fail anyway.

```c
// Added to history pruning condition:
&& variation->plyInfo[ply].isHashMove == FALSE
```


---

## Tighten singular extension hash-entry staleness (3→2 plies)
**Date:** 2026-04-11 (session 3)
**LOS:** 53.9% at 431 games (no significant improvement)

Changed `importance >= restDepth - 3 * DEPTH_RESOLUTION` to `importance >= restDepth - 2 * DEPTH_RESOLUTION`
in the singular extension condition. Hypothesis: hash entries 3 plies shallower than current depth are
unreliable for singularity judgement; requiring fresher entries would save verification search overhead.

Result: neutral — LOS stayed near 50% throughout the match. The 3-ply tolerance is apparently
well-calibrated; tightening it neither saves meaningful time nor improves accuracy.

```c
// Changed in singular extension condition:
if (restDepth >= singleMoveExtensionDepth && importance >= restDepth - 2 * DEPTH_RESOLUTION &&
    flag != HASHVALUE_UPPER_LIMIT) {
```

---

## Remove PV-node recapture extension
**Date:** 2026-04-11 (session 3)
**LOS:** 9.8% at 206 games (threshold: 50% at 200 games)

Removed the balanced-recapture condition from the PV node extension, leaving only check extensions.
Hypothesis: with NNUE evaluation already handling material balance, the recapture extension
(non-pawn capture when piece counts are equal) was redundant and wasteful.

Result: badly hurt playing strength. The recapture extension is load-bearing — it provides
important additional depth along forcing exchange sequences that NNUE alone does not cover.

```c
// Changed from:
if (pvNode && (check || (capturedPiece != NO_PIECE && pieceType(capturedPiece) != PAWN &&
                         numberOfNonPawnPieces(position, WHITE) == numberOfNonPawnPieces(position, BLACK)))) {
// To:
if (pvNode && check) {
```

---

## Raise singular extension minimum depth for non-PV nodes (8→9 plies)
**Date:** 2026-04-11 (session 3)
**LOS:** 29.0% at 100 games (threshold: 40% at 100 games)

Changed `getSingleMoveExtensionDepth` to return `9 * DEPTH_RESOLUTION` for non-PV nodes
(was `8 * DEPTH_RESOLUTION`). Hypothesis: prevent expensive half-depth verification at
depth 8, saving node budget.

Result: clearly hurt. The singular extension at depth 8 (non-PV) is load-bearing.
The 3-ply staleness tightening (also tried) was neutral; depth threshold is more sensitive.

```c
// Changed:
return (pvNode ? 4 : 9) * DEPTH_RESOLUTION;
```

---

## Increase non-PV LMR aggressiveness (divisor 2.21→2.00)
**Date:** 2026-04-11 (session 4)
**LOS:** 27.2% at 100 games (threshold: 40% at 100 games)

Reduced the divisor in the non-PV quiet move LMR formula from 2.21 to 2.00, making reductions
more aggressive for middle-ranked moves at moderate depths. At baseFactor ∈ [4.34, 4.80]
(e.g. move 7 at depth 10), reduction increases by a full ply (5→7 with DEPTH_RESOLUTION=2).
Hypothesis: the re-search safety net would catch any genuinely good moves that get over-reduced.

Result: clearly hurt. Over-reducing quiet moves in non-PV nodes degrades move ordering decisions
and causes the engine to miss important continuations even with re-search available.

```c
// Changed in initializeArrays:
const double nonPvReduction = 0.33 + baseFactor / 2.00;  // was 2.21
```

---

## Extend SEE-based quiet move pruning (predictedDepth < 4*DR → < 5*DR)
**Date:** 2026-04-11 (session 4)
**LOS:** 14.7% at 221 games (threshold: 50% at 200 games)

Extended the SEE-based quiet move pruning condition from `predictedDepth < 4 * DEPTH_RESOLUTION`
to `< 5 * DEPTH_RESOLUTION`, pruning negative-SEE quiet moves one ply deeper. Hypothesis: with
NNUE's accurate static eval, quiet moves losing material at depth 5 rarely deserve a full search.

Result: clearly hurt. Promising start at 100 games (LOS=64.5%), then reversed sharply. Pruning
negative-SEE quiet moves at depth 5 loses important moves that look bad statically but become
relevant at depth 5+ (e.g. quiet moves enabling tactical sequences the SEE misses).

```c
// Changed in optimistic futility cuts:
if (predictedDepth < 5 * DEPTH_RESOLUTION && seeMove(position, currentMove) < 0) {
```

---

## Update followup moves unconditionally (remove isHashMove restriction)
**Date:** 2026-04-12 (session 4)
**LOS:** 63.4% at 600 games (threshold: 66.7%)

Removed the `variation->plyInfo[ply - 1].isHashMove` condition from followup move updates,
so the followup move table is populated on every best quiet move at ply >= 2, not just when
the previous move was the hash move. Hypothesis: response patterns are valid regardless of
whether the previous move was the hash move; more updates → richer, more accurate table.

Result: insufficient — LOS peaked at 96.8% at 200 games but regressed to 63.4% at 600 games.
The isHashMove restriction appears to serve as a quality filter: followup patterns are most
reliable when the previous move was strong (hash move), not arbitrary moves.

```c
// Changed from:
if (ply >= 2 && variation->plyInfo[ply - 1].isHashMove) {
// To:
if (ply >= 2) {
    updateFollowupMoves(variation, ply, killerMove);
```

---

## History-based LMR and no LMR on checks
**Date:** 2026-04-12
**LOS:** 27.6% at 200 games (threshold: 50% at 200 games)

Inhibited LMR for moves that give check and added a history-based reduction bonus
(reduction -= historyValue / 4096). Hypothesis: preventing reduction of tactical
moves (checks) and strong history moves would improve playing strength.

Result: clearly hurt. The history bonus likely caused over-searching of quiet moves,
and the check exclusion might have been redundant or counter-productive in the
context of Protector's existing extensions.

---

## Remove extra LMR step for high-reduction moves (> 2*DR → only +DR/2)
**Date:** 2026-04-12
**LOS:** 46.7% at 300 games (threshold: 50% at 300 games)

Replaced the two-tier extra reduction step with a single tier: removed the `+DEPTH_RESOLUTION`
bonus for base reductions > 2*DR, keeping only `+DEPTH_RESOLUTION/2` for base > DR.
Hypothesis: moderating the extra reduction for heavily-reduced late moves would improve
accuracy without significant node-count increase.

Result: neutral/slightly negative. Started promising (LOS ~80% at 100 games) but steadily
declined to near 50/50 by 300 games. The two-tier step for high reductions appears correctly
calibrated — the extra ply of reduction for late moves at high depth is beneficial for speed.

```c
// Changed from:
if (quietMoveReduction[i][j] > 2 * DEPTH_RESOLUTION) {
    quietMoveReduction[i][j] += DEPTH_RESOLUTION;
} else if (quietMoveReduction[i][j] > DEPTH_RESOLUTION) {
    quietMoveReduction[i][j] += DEPTH_RESOLUTION / 2;
}
// To:
if (quietMoveReduction[i][j] > DEPTH_RESOLUTION) {
    quietMoveReduction[i][j] += DEPTH_RESOLUTION / 2;
}
```

---

## Reduce non-PV LMR aggressiveness (divisor 2.21→2.40)
**Date:** 2026-04-12
**LOS:** 38.3% at 400 games (threshold: 60% at 400 games)

Increased the divisor in the non-PV quiet move LMR formula from 2.21 to 2.40, making reductions
less aggressive (searching more moves deeper). Hypothesis: mirroring the successful PV divisor
increase (2.90→3.20), reducing non-PV aggressiveness would improve accuracy.

Result: started with early promise (~72% LOS at 230 games) but steadily declined to 38.3% at
400 games. Unlike the PV change, the non-PV formula is apparently well-calibrated at 2.21; the
extra reduction from the two-tier step already provides sufficient aggressiveness control.

```c
// Changed in initializeArrays:
const double nonPvReduction = 0.33 + baseFactor / 2.40;  // was 2.21
```

---

## Raise quick refutation search depth threshold (5*DR → 6*DR)
**Date:** 2026-04-12
**LLR:** -1.042 at 662 games (Elo window 2.0–10.0, failing)

Changed the quick refutation search trigger from `restDepth >= 5 * DEPTH_RESOLUTION` to
`restDepth >= 6 * DEPTH_RESOLUTION`, disabling the shallow (depth-1) quick refutation at
depth 5. Hypothesis: at depth 5 the quick refutation searches captures at depth 1, which
is too shallow to be reliable and just adds overhead.

Result: slightly negative. The quick refutation at depth 5 apparently still contributes
useful pruning despite its shallowness; removing it cost more than it saved.

```c
// Changed:
if (pvNode == FALSE && cutsAreAllowed && restDepth >= 6 * DEPTH_RESOLUTION && ...)
```


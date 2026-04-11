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


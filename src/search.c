/*
    Protector -- a UCI chess engine

    Copyright (C) 2009-2010 Raimund Heid (Raimund_Heid@yahoo.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "search.h"

#include "coordination.h"
#include "evaluation.h"
#include "hash.h"
#include "io.h"
#include "movegeneration.h"
#include "nnue.h"
#include "position.h"
#include "tablebase.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern bool resetSharedHashtable;
const int HASH_DEPTH_OFFSET = 3;
const UINT64 GUI_NODE_COUNT_MIN = 250000;

static int log1024[1024];

/* Prototypes */
static int searchBest(Variation *variation, int alpha, int beta, const int ply, const int restDepth, Move *bestMove,
                      const bool pvNode, bool cutNode, Move excludeMove);

static bool moveIsQuietInPosition(const Move move, const Position *position)
{
    return (bool)(position->piece[getToSquare(move)] == NO_PIECE && getNewPiece(move) == NO_PIECE &&
                  (getToSquare(move) != position->enPassantSquare ||
                   pieceType(position->piece[getFromSquare(move)]) != PAWN));
}

static bool moveIsQuiet(const Move move, const Position *position, const MovegenerationStage stage)
{
    switch (stage) {
    case MGS_REST:
    case MGS_KILLER_MOVES:
        return TRUE;
    case MGS_GOOD_CAPTURES_AND_PROMOTIONS:
    case MGS_BAD_CAPTURES:
        return FALSE;
    default:
        return moveIsQuietInPosition(move, position);
    }
}

static void checkTerminationConditions(Variation *variation)
{
    if (variation->searchStatus == SEARCH_STATUS_RUNNING) {
        if (variation->terminate && (variation->iteration > 1 || variation->threadNumber > 0)) {
            variation->searchStatus = SEARCH_STATUS_TERMINATE;
        }
    }
}

bool checkNodeExclusion(int restDepth)
{
    return restDepth >= 6 && getNumberOfThreads() > 1;
}

int getEvalValue(Variation *variation)
{
    const int opt = (variation->singlePosition.activeColor == variation->startPosition.activeColor)
                        ? variation->optimism
                        : -variation->optimism;
    finalizeAccumulator(variation, WHITE);
    finalizeAccumulator(variation, BLACK);
    return getValue(&variation->singlePosition, &variation->plyInfo[variation->ply].accumulator, opt);
}

static int getStaticValue(Variation *variation)
{
    PlyInfo *pi = &variation->plyInfo[variation->ply];

    if (pi->staticValueAvailable == FALSE) {
        pi->staticValue = getEvalValue(variation);
        pi->staticValueAvailable = TRUE;
    } else {
        assert(pi->staticValue == getEvalValue(variation));
    }

    return pi->staticValue;
}

static void updateCounterMoves(Variation *variation, const int ply, Move counterMove)
{
    if (variation->plyInfo[ply - 1].currentMove != NULLMOVE) {
        const int moveIndex = variation->plyInfo[ply - 1].indexCurrentMove;

        if (counterMove != variation->counterMove1[moveIndex]) {
            variation->counterMove2[moveIndex] = variation->counterMove1[moveIndex];
            variation->counterMove1[moveIndex] = counterMove;
        }
    }
}

static void updateFollowupMoves(Variation *variation, const int ply, Move followupMove)
{
    if (variation->plyInfo[ply - 2].currentMove != NULLMOVE) {
        const int moveIndex = variation->plyInfo[ply - 2].indexCurrentMove;

        if (followupMove != variation->followupMove1[moveIndex]) {
            variation->followupMove2[moveIndex] = variation->followupMove1[moveIndex];
            variation->followupMove1[moveIndex] = followupMove;
        }
    }
}

static bool positionIsWellKnown(Variation *variation, Position *position, const UINT64 hashKey,
                                Hashentry **bestTableHit, const int ply, const int alpha, const int beta,
                                const int restDepth, const bool pvNode, Move *hashmove, const Move excludeMove,
                                int *value)
{
    Hashentry *tableHit = getHashentry(getSharedHashtable(), hashKey);

    if (tableHit != NULL) { /* 45% */
        const int importance = getHashentryImportance(tableHit);
        const int flag = getHashentryFlag(tableHit);
        const int hashEntryValue = calcEffectiveValue(getHashentryValue(tableHit), ply);
        PlyInfo *pi = &variation->plyInfo[ply];

        *bestTableHit = tableHit;
        *value = hashEntryValue;

        assert(getHashentryStaticValue(tableHit) == getStaticValue(variation));

        if (pi->staticValueAvailable == FALSE && pvNode == FALSE) {
            pi->staticValue = getHashentryStaticValue(tableHit);
            pi->staticValueAvailable = TRUE;
        }

        if (pvNode == FALSE && excludeMove != NULLMOVE && restDepth <= importance && flag != HASHVALUE_EVAL) { /* 99% */
            assert(flag == HASHVALUE_UPPER_LIMIT || flag == HASHVALUE_EXACT || flag == HASHVALUE_LOWER_LIMIT);
            assert(hashEntryValue >= VALUE_MATED && hashEntryValue < -VALUE_MATED);

            switch (flag) {
            case HASHVALUE_UPPER_LIMIT:
                if (hashEntryValue <= alpha) {
                    *value = (hashEntryValue <= VALUE_MATED ? alpha : hashEntryValue);

                    return TRUE;
                }

                if (restDepth >= HASH_DEPTH_OFFSET && hashEntryValue < pi->staticValue) {
                    variation->plyInfo[ply].staticValue = hashEntryValue;
                }
                break;

            case HASHVALUE_EXACT:
                *value = hashEntryValue;

                return TRUE;

            case HASHVALUE_LOWER_LIMIT:
                if (hashEntryValue >= beta) {
                    *value = (hashEntryValue >= -VALUE_MATED ? beta : hashEntryValue);

                    return TRUE;
                }

                if (restDepth >= HASH_DEPTH_OFFSET && hashEntryValue > pi->staticValue) {
                    variation->plyInfo[ply].staticValue = hashEntryValue;
                }
                break;

            default:;
            }
        }
    }

    if (*bestTableHit != 0) {
        *hashmove = (Move)getHashentryMove(*bestTableHit);

        if (*hashmove != NO_MOVE) { /* 81% */
            assert(moveIsPseudoLegal(position, *hashmove));

            if (moveIsPseudoLegal(position, *hashmove)) {
                assert(moveIsLegal(position, *hashmove));
            } else {
                *hashmove = NO_MOVE;
            }
        }
    }

    return FALSE;
}

static int searchBestQuiescence(Variation *variation, int alpha, int beta, const int ply, const int restDepth,
                                Move *bestMove, const bool pvNode)
{
    const int oldAlpha = alpha;
    UINT8 hashentryFlag;
    UINT8 hashDepth = (restDepth >= 0 ? 2 : 1);
    Position *position = &variation->singlePosition;
    int best = VALUE_MATED, currentValue = VALUE_MATED, historyLimit, i;
    const int VALUE_MATE_HERE = -VALUE_MATED - ply + 1;
    const int VALUE_MATED_HERE = VALUE_MATED + ply;
    Movelist movelist;
    Move currentMove, bestReply, hashmove = NO_MOVE;
    const bool inCheck = variation->plyInfo[ply - 1].currentMoveIsCheck;
    Hashentry *bestTableHit = 0;
    int hashValue;

    assert(alpha >= VALUE_MATED && alpha <= -VALUE_MATED);
    assert(beta >= VALUE_MATED && beta <= -VALUE_MATED);
    assert(alpha < beta);
    assert(ply > 0 && ply < MAX_DEPTH);
    assert(restDepth < 1);
    assert(passiveKingIsSafe(position));
    assert(inCheck == (activeKingIsSafe(position) == FALSE));

    *bestMove = NO_MOVE;
    variation->plyInfo[ply - 1].pv.length = 0;
    variation->plyInfo[ply - 1].pv.move[0] = NO_MOVE;

    variation->nodes++;
    checkTerminationConditions(variation);

    if (variation->searchStatus != SEARCH_STATUS_RUNNING && variation->iteration > 1) {
        return 0;
    }

    /* Check for a draw according to the 50-move-rule */
    /* ---------------------------------------------- */
    if (position->halfMoveClock > 100) {
        return variation->drawScore[position->activeColor];
    }

    /* Check for a draw by repetition. */
    /* ------------------------------- */
    if (position->halfMoveClock >= 4) {
        historyLimit = POSITION_HISTORY_OFFSET + variation->ply - position->halfMoveClock;

        assert(historyLimit >= 0);

        for (i = POSITION_HISTORY_OFFSET + variation->ply - 4; i >= historyLimit; i -= 2) {
            if (position->hashKey == variation->positionHistory[i]) {
                return variation->drawScore[position->activeColor];
            }
        }
    }

    /* Probe the transposition table */
    /* ----------------------------- */

    if (positionIsWellKnown(variation, position, position->hashKey, &bestTableHit, ply, alpha, beta, hashDepth, pvNode,
                            &hashmove, NO_MOVE, &hashValue)) {
        *bestMove = hashmove;

        return hashValue;
    }

    if (hashmove != NO_MOVE && inCheck == FALSE && moveIsQuietInPosition(hashmove, position)) {
        hashmove = NO_MOVE;
    }

    if (inCheck == FALSE) {
        const bool staticValueAvailable = variation->plyInfo[ply].staticValueAvailable;
        best = getStaticValue(variation);

        if (bestTableHit != 0) {
            const int flag = getHashentryFlag(bestTableHit);
            const int requiredFlag = (hashValue > best ? HASHVALUE_LOWER_LIMIT : HASHVALUE_UPPER_LIMIT);

            if (flag == requiredFlag) {
                best = hashValue;
            }
        }

        if (best > alpha) {
            alpha = best;

            if (best >= beta) {
                if (staticValueAvailable == FALSE) {
                    UINT8 hashentryFlag = HASHVALUE_EVAL;

                    setHashentry(getSharedHashtable(), position->hashKey, calcHashtableValue(best, ply), 0,
                                 packedMove(NO_MOVE), hashentryFlag, (INT16)best);
                }

                return best;
            }
        }

        currentValue = best;
    }

    if (ply >= MAX_DEPTH) {
        assert(flipTest(position));

        return getStaticValue(variation);
    }

    if (alpha < VALUE_MATED_HERE && inCheck == FALSE) {
        alpha = VALUE_MATED_HERE;

        if (alpha >= beta) {
            return VALUE_MATED_HERE;
        }
    }

    if (beta > VALUE_MATE_HERE) {
        beta = VALUE_MATE_HERE;

        if (beta <= alpha) {
            return VALUE_MATE_HERE;
        }
    }

    initQuiescenceMovelist(&movelist, &variation->singlePosition, &variation->plyInfo[ply], variation->moveHistory, ply,
                           hashmove, restDepth, inCheck);
    initializePlyInfo(variation);

    const int newDepth = (inCheck ? restDepth : restDepth - 1);
    const int futilityBase = currentValue + 199;
    const bool skipFutility = (bool)(pvNode || inCheck || numberOfNonPawnPieces(position, position->activeColor) <= 1);

    while ((currentMove = getNextMove(&movelist)) != NO_MOVE) {
        int value, optValue = futilityBase + basicValue[position->piece[getToSquare(currentMove)]];
        const Square toSquare = getToSquare(currentMove);

        if (skipFutility == FALSE && optValue < alpha && optValue > VALUE_ALMOST_MATED &&
            movesAreEqual(currentMove, hashmove) == FALSE && getNewPiece(currentMove) != (Piece)QUEEN &&
            (pieceType(position->piece[getFromSquare(currentMove)]) != PAWN ||
             colorRank(position->activeColor, toSquare) != RANK_7)) {
            const bool enPassant = (bool)(toSquare == position->enPassantSquare &&
                                          pieceType(position->piece[getFromSquare(currentMove)]) == PAWN);

            if (getNewPiece(currentMove) != NO_PIECE) {
                optValue += basicValue[getNewPiece(currentMove)] - basicValue[PAWN];
            }

            if (enPassant) {
                optValue += basicValue[PAWN];
            }

            if (optValue < alpha && moveIsCheck(currentMove, position) == FALSE) {
                best = max(best, optValue);

                continue;
            }
        }

        assert(moveIsPseudoLegal(position, currentMove));

        if (makeMoveFast(variation, currentMove) != 0 || passiveKingIsSafe(&variation->singlePosition) == FALSE) {
            unmakeLastMove(variation);

            continue;
        }

        variation->plyInfo[ply].currentMoveIsCheck = activeKingIsSafe(&variation->singlePosition) == FALSE;

        assert(position->piece[getToSquare(currentMove)] != NO_PIECE ||
               (getToSquare(currentMove) == position->enPassantSquare &&
                position->piece[getFromSquare(currentMove)] == (PAWN | position->activeColor)) ||
               getNewPiece(currentMove) != NO_PIECE || inCheck || variation->plyInfo[ply].currentMoveIsCheck);

        assert(inCheck ||
               basicValue[position->piece[getFromSquare(currentMove)]] <=
                   basicValue[position->piece[getToSquare(currentMove)]] ||
               seeMove(position, currentMove) >= 0);

        value = -searchBestQuiescence(variation, -beta, -alpha, ply + 1, newDepth, &bestReply, pvNode);

        unmakeLastMove(variation);

        if (variation->searchStatus != SEARCH_STATUS_RUNNING && variation->iteration > 1) {
            return 0;
        }

        if (value > best) {
            best = value;

            if (best > alpha) {
                alpha = best;
                *bestMove = currentMove;

                if (pvNode) {
                    appendMoveToPv(&(variation->plyInfo[ply].pv), &(variation->plyInfo[ply - 1].pv), currentMove);
                }

                if (best >= beta) {
                    break;
                }
            }
        }
    }

    if (best == VALUE_MATED) {
        /* mate */

        assert(inCheck);

        best = VALUE_MATED + ply;
    }

    /* Store the value in the transposition table. */
    /* ------------------------------------------- */
    if (best >= beta) {
        hashentryFlag = HASHVALUE_LOWER_LIMIT;
    } else {
        hashentryFlag = (best > oldAlpha && pvNode ? HASHVALUE_EXACT : HASHVALUE_UPPER_LIMIT);
    }

    setHashentry(getSharedHashtable(), position->hashKey, calcHashtableValue(best, ply), hashDepth,
                 packedMove(*bestMove), hashentryFlag, (INT16)getStaticValue(variation));

    return best;
}

static bool isImproving(Variation *variation)
{
    return variation->ply >= 2 && getStaticValue(variation) > variation->plyInfo[variation->ply - 2].staticValue;
}

static bool isPassedPawnMove(const Square pawnSquare, const Square targetSquare, const Position *position)
{
    const Piece piece = position->piece[pawnSquare];

    if (pieceType(piece) == PAWN) {
        return pawnIsPassed(position, targetSquare, pieceColor(piece));
    } else {
        return FALSE;
    }
}

static bool isSpecialMove(Position *position, Move move)
{
    return simpleMoveIsCheck(position, move) || isPassedPawnMove(getFromSquare(move), getToSquare(move), position);
}

static int searchBest(Variation *variation, int alpha, int beta, const int ply, const int restDepth, Move *bestMove,
                      const bool pvNode, bool cutNode, Move excludeMove)
{
    Position *position = &variation->singlePosition;
    int best = VALUE_MATED;
    const int VALUE_MATE_HERE = -VALUE_MATED - ply + 1;
    const int VALUE_MATED_HERE = VALUE_MATED + ply;
    const int numPieces = numberOfNonPawnPieces(position, position->activeColor);
    Movelist movelist;
    UINT8 hashentryFlag;
    int i, historyLimit, numMovesPlayed = 0;
    Move hashmove = NO_MOVE;
    Hashentry *bestTableHit = 0;
    Move currentMove, bestReply;
    const bool inCheck = variation->plyInfo[ply - 1].currentMoveIsCheck;
    const UINT64 hashKey = position->hashKey;
    int quietMoveIndex[MAX_MOVES_PER_POSITION], quietMoveCount = 0;
    int deferCount = 0;

    *bestMove = NO_MOVE;
    variation->plyInfo[ply].isHashMove = FALSE;
    variation->plyInfo[ply - 1].pv.length = 0;
    variation->plyInfo[ply - 1].pv.move[0] = NO_MOVE;

    if (ply + 1 > variation->selDepth) {
        variation->selDepth = ply + 1;
    }

    assert(alpha >= VALUE_MATED && alpha <= -VALUE_MATED);
    assert(beta >= VALUE_MATED && beta <= -VALUE_MATED);
    assert(alpha < beta);
    assert(ply > 0 && ply < MAX_DEPTH);
    assert(passiveKingIsSafe(position));
    assert(inCheck == (activeKingIsSafe(position) == FALSE));

    /* Check for a draw according to the 50-move-rule */
    /* ---------------------------------------------- */
    if (position->halfMoveClock > 100) {
        return variation->drawScore[position->activeColor];
    }

    /* Check for a draw by repetition. */
    /* ------------------------------- */
    if (position->halfMoveClock >= 4) {
        historyLimit = POSITION_HISTORY_OFFSET + variation->ply - position->halfMoveClock;

        assert(historyLimit >= 0);

        for (i = POSITION_HISTORY_OFFSET + variation->ply - 4; i >= historyLimit; i -= 2) {
            if (position->hashKey == variation->positionHistory[i]) {
                return variation->drawScore[position->activeColor];
            }
        }
    }

    if (restDepth < 1) {
        return searchBestQuiescence(variation, alpha, beta, ply, 0, bestMove, pvNode);
    }

    variation->nodes++;
    checkTerminationConditions(variation);

    if (variation->searchStatus != SEARCH_STATUS_RUNNING && variation->iteration > 1) {
        return 0;
    }

    /* Probe the tablebases in case of reduced material */
    /* ------------------------------------------------ */
    if (tbAvailable && excludeMove == NO_MOVE) {
        int numPieces = position->numberOfPieces[WHITE] + position->numberOfPieces[BLACK];
        int wdlValue = TABLEBASE_ERROR;

        if (restDepth >= 1 && numPieces <= 6) {
            wdlValue = probeTablebaseWDL(position);
        }

        if (wdlValue != TABLEBASE_ERROR) {
            int score;

            variation->tbHits++;

            if (wdlValue == 2) // Win
                score = -VALUE_MATED - MAX_DEPTH - 1 - ply;
            else if (wdlValue == 1) // Cursed Win
                score = 1;
            else if (wdlValue == 0) // Draw
                score = variation->drawScore[position->activeColor];
            else if (wdlValue == -1) // Blessed Loss
                score = -1;
            else // Loss
                score = VALUE_MATED + MAX_DEPTH + 1 + ply;

            if (wdlValue == 0 || (wdlValue > 0 && score >= beta) || (wdlValue < 0 && score <= alpha)) {
                return score;
            }

            if (score > alpha && wdlValue > 0) {
                alpha = score;
            } else if (score < beta && wdlValue < 0) {
                beta = score;
            }
        }
    }

    /* Probe the transposition table */
    /* ----------------------------- */
    if (excludeMove == NO_MOVE) {
        int hashValue;

        if (positionIsWellKnown(variation, position, hashKey, &bestTableHit, ply, alpha, beta,
                                restDepth + HASH_DEPTH_OFFSET, pvNode, &hashmove, NO_MOVE, &hashValue)) {
            *bestMove = hashmove;

            if (hashValue >= beta && *bestMove != NO_MOVE && moveIsQuietInPosition(*bestMove, position)) {
                Move killerMove = *bestMove;
                const Piece movingPiece = position->piece[getFromSquare(killerMove)];

                setMoveValue(&killerMove, movingPiece);
                registerKillerMove(&variation->plyInfo[ply], killerMove);
            }

            return hashValue;
        }
    }

    if (ply >= MAX_DEPTH) {
        return getStaticValue(variation);
    }

    if (alpha < VALUE_MATED_HERE && inCheck == FALSE) {
        alpha = VALUE_MATED_HERE;

        if (alpha >= beta) {
            return VALUE_MATED_HERE;
        }
    }

    if (beta > VALUE_MATE_HERE) {
        beta = VALUE_MATE_HERE;

        if (beta <= alpha) {
            return VALUE_MATE_HERE;
        }
    }

    initializePlyInfo(variation);

    if (pvNode == FALSE && inCheck == FALSE && hashmove == NO_MOVE) {
        /* Razoring */
        /* -------- */
        if (getStaticValue(variation) < alpha - 132 - 32 * restDepth * restDepth) {
            return searchBestQuiescence(variation, alpha, beta, ply, 0, bestMove, pvNode);
        }

        /* Static pruning */
        /* -------------- */
        if (restDepth <= 4) {
            const int margin = 22 + 44 * restDepth - (isImproving(variation) ? 24 : 0);

            if (getStaticValue(variation) - margin >= beta) {
                return beta;
            }
        }
    }

    /* Null move pruning */
    /* ----------------- */
    if (pvNode == FALSE && inCheck == FALSE && restDepth >= 2 && excludeMove == NO_MOVE && numPieces >= 2 &&
        abs(beta) <= -VALUE_ALMOST_MATED && getStaticValue(variation) >= beta) {
        const int newDepth = restDepth - 5 - restDepth / 4;

        makeMoveFast(variation, NULLMOVE);
        variation->plyInfo[ply].currentMoveIsCheck = FALSE;
        int nullValue =
            -searchBest(variation, -beta, -beta + 1, ply + 1, newDepth, &bestReply, FALSE, !cutNode, NO_MOVE);
        unmakeLastMove(variation);

        if (nullValue >= beta) {
            if (nullValue >= VALUE_MATE_HERE) {
                nullValue = beta;
            }

            if (restDepth < 6 ||
                searchBest(variation, alpha, beta, ply, newDepth, &bestReply, FALSE, FALSE, NULLMOVE) >= beta) {
                best = nullValue;
                goto storeResult;
            }
        }
    }

    /* Internal iterative deepening */
    /* ---------------------------- */
    if (hashmove == NO_MOVE && restDepth >= 3) {
        const Move excludeHere = (excludeMove != NO_MOVE ? excludeMove : NULLMOVE);
        searchBest(variation, alpha, beta, ply, restDepth - 2, &bestReply, pvNode, TRUE, excludeHere);

        if (moveIsPseudoLegal(position, bestReply)) {
            hashmove = bestReply;
        }

        if (hashmove != NO_MOVE && excludeMove == NO_MOVE && restDepth >= (pvNode ? 4 : 8)) {
            Hashentry *tableHit = getHashentry(getSharedHashtable(), variation->singlePosition.hashKey);

            if (tableHit != 0) {
                bestTableHit = tableHit;
            }
        }
    }

    /* Copy counter moves from ply-1 */
    /* ----------------------------- */
    if (ply >= 1) {
        const int moveIndex = variation->plyInfo[ply - 1].indexCurrentMove;

        variation->plyInfo[ply].killerMove3 = variation->counterMove1[moveIndex];
        variation->plyInfo[ply].killerMove4 = variation->counterMove2[moveIndex];
    } else {
        variation->plyInfo[ply].killerMove3 = NO_MOVE;
        variation->plyInfo[ply].killerMove4 = NO_MOVE;
    }

    /* Copy follow-up moves from ply-2 */
    /* ------------------------------- */
    if (ply >= 2) {
        const int moveIndex = variation->plyInfo[ply - 2].indexCurrentMove;

        variation->plyInfo[ply].killerMove5 = variation->followupMove1[moveIndex];
        variation->plyInfo[ply].killerMove6 = variation->followupMove2[moveIndex];
    } else {
        variation->plyInfo[ply].killerMove5 = NO_MOVE;
        variation->plyInfo[ply].killerMove6 = NO_MOVE;
    }

    const int staticValue = getStaticValue(variation);
    const bool improving = isImproving(variation) || staticValue >= beta;

    /* ProbCut: if a capture passes both qsearch and a shallow search above a high threshold, prune */
    /* -------------------------------------------------------------------------------------------- */
    if (pvNode == FALSE && inCheck == FALSE && restDepth >= 5 && excludeMove == NO_MOVE &&
        abs(beta) <= -VALUE_ALMOST_MATED) {
        const int probCutBeta = min(-VALUE_ALMOST_MATED, beta + 200);
        Movelist probMovelist;
        Move probMove, probReply;
        const Move probHashmove =
            (hashmove != NO_MOVE && !moveIsQuietInPosition(hashmove, position)) ? hashmove : NO_MOVE;

        initCaptureMovelist(&probMovelist, position, &variation->plyInfo[ply], variation->moveHistory, ply,
                            probHashmove, FALSE);

        while ((probMove = getNextMove(&probMovelist)) != NO_MOVE) {
            if (seeMove(position, probMove) < probCutBeta - staticValue)
                continue;

            variation->plyInfo[ply].indexCurrentMove = historyIndex(probMove, position);
            variation->plyInfo[ply].isHashMove = movesAreEqual(probMove, hashmove);

            if (makeMoveFast(variation, probMove) != 0 || passiveKingIsSafe(&variation->singlePosition) == FALSE) {
                unmakeLastMove(variation);
                continue;
            }

            variation->plyInfo[ply].currentMoveIsCheck = activeKingIsSafe(&variation->singlePosition) == FALSE;

            int probValue =
                -searchBestQuiescence(variation, -probCutBeta, -probCutBeta + 1, ply + 1, 0, &probReply, FALSE);

            if (probValue >= probCutBeta) {
                const int probCutDepth = restDepth - 4;
                if (probCutDepth > 0) {
                    probValue = -searchBest(variation, -probCutBeta, -probCutBeta + 1, ply + 1, probCutDepth,
                                            &probReply, FALSE, !cutNode, NO_MOVE);
                }
            }

            unmakeLastMove(variation);

            if (variation->searchStatus != SEARCH_STATUS_RUNNING && variation->iteration > 1) {
                return 0;
            }

            if (probValue >= probCutBeta) {
                *bestMove = probMove;
                best = probValue;
                goto storeResult;
            }
        }
    }

    initStandardMovelist(&movelist, &variation->singlePosition, &variation->plyInfo[ply], variation->moveHistory, ply,
                         hashmove, inCheck);

    /* Loop through all moves in this node */
    /* ----------------------------------- */
    while ((currentMove = getNextMove(&movelist)) != NO_MOVE) {
        const MovegenerationStage stage = moveGenerationStage[movelist.currentStage];
        int value;
        const int historyIndexMove = historyIndex(currentMove, position);
        const bool quietMove = moveIsQuiet(currentMove, position, stage);
        bool nodeWasBlocked = FALSE;
        int reductions = log1024[restDepth] * log1024[numMovesPlayed] / 2176 + (cutNode ? 2048 : 0), extensions = 0;

        if (quietMove) {
            const MoveHistoryEntry *histEntry = &variation->moveHistory[ply][historyIndexMove];
            const int plyScore = (int)(16000LL * (histEntry->succ + 1) / (histEntry->freq + 2) - 8000);
            reductions = max(0, reductions - plyScore / 8);
        }

        if (variation->searchStatus != SEARCH_STATUS_RUNNING && variation->iteration > 1) {
            return 0;
        }

        if (excludeMove != NO_MOVE && movesAreEqual(currentMove, excludeMove)) {
            assert(excludeMove != NULLMOVE);

            continue; /* exclude excludeMove */
        }

        variation->plyInfo[ply].indexCurrentMove = historyIndexMove;
        variation->plyInfo[ply].isHashMove = movesAreEqual(currentMove, hashmove);

        assert(moveIsPseudoLegal(position, currentMove));
        assert(hashmove == NO_MOVE || numMovesPlayed > 0 || movesAreEqual(currentMove, hashmove));

        /* Optimistic futility cuts */
        /* ------------------------ */
        if (pvNode == FALSE && inCheck == FALSE && quietMove && best > VALUE_ALMOST_MATED &&
            isSpecialMove(position, currentMove) == FALSE) {
            if (numMovesPlayed >= (improving ? 79 : 45) * (3 + restDepth * restDepth) / 64) {
                continue;
            }

            if (restDepth < 4 && seeMove(position, currentMove) < 0) {
                continue;
            }
        }

        /* Single move extension check */
        /* --------------------------- */
        if (movesAreEqual(currentMove, hashmove) && excludeMove == NO_MOVE && restDepth >= (pvNode ? 4 : 8) &&
            bestTableHit != 0 && getHashentryImportance(bestTableHit) - HASH_DEPTH_OFFSET >= restDepth - 3 &&
            getHashentryFlag(bestTableHit) == HASHVALUE_LOWER_LIMIT) {
            const int hashEntryValue = calcEffectiveValue(getHashentryValue(bestTableHit), ply);
            const int limitValue = hashEntryValue - (50 * restDepth) / 64;

            if (limitValue > VALUE_ALMOST_MATED && limitValue < -VALUE_ALMOST_MATED) {
                const PlyInfo pi = variation->plyInfo[ply];
                const int excludeValue = searchBest(variation, limitValue - 1, limitValue, ply, restDepth / 2,
                                                    &bestReply, FALSE, cutNode, hashmove);
                variation->plyInfo[ply] = pi;

                if (excludeValue < limitValue) {
                    extensions += 1024;
                } else if (excludeValue >= beta && abs(excludeValue) <= -VALUE_ALMOST_MATED) {
                    best = excludeValue;
                    goto storeResult;
                } else if (cutNode) {
                    extensions -= 1024;
                }
            }
        }

        /* Execute the current move and check if it is legal. */
        /* -------------------------------------------------- */
        if (makeMoveFast(variation, currentMove) != 0 || passiveKingIsSafe(&variation->singlePosition) == FALSE) {
            unmakeLastMove(variation);

            continue;
        }

        if (numMovesPlayed > 0 && inCheck == FALSE && stage == MGS_REST && deferCount < 10 &&
            checkNodeExclusion(restDepth)) {
            if (nodeIsInUse(position->hashKey, restDepth)) {
                deferMove(&movelist, currentMove);
                deferCount++;
                unmakeLastMove(variation);
                continue;
            } else {
                nodeWasBlocked = setNodeUsage(position->hashKey, restDepth);
            }
        }

        const bool check = variation->plyInfo[ply].currentMoveIsCheck =
            activeKingIsSafe(&variation->singlePosition) == FALSE;

        if (pvNode) {
            reductions = 3 * reductions / 4;

            if (check && extensions == 0) {
                extensions += 1024;
            }
        }

        const bool reduce = numMovesPlayed > 1 && reductions >= 1024 && extensions == 0 && inCheck == FALSE &&
                            restDepth >= 3 && stage == MGS_REST;

        const int pvDepth = restDepth - 1 + extensions / 1024;
        const int reducedDepth = restDepth - 1 - reductions / 1024;
        const bool pvSearch = pvNode && numMovesPlayed == 0;
        const bool cutNodeValue = (pvSearch == FALSE && cutNode == FALSE) || reduce;

        value = -searchBest(variation, pvSearch ? -beta : -alpha - 1, -alpha, ply + 1,
                            pvSearch || reduce == FALSE ? pvDepth : reducedDepth, &bestReply, pvSearch, cutNodeValue,
                            NO_MOVE);

        if (pvSearch == FALSE && value > alpha) {
            if (reduce && reducedDepth < pvDepth) {
                if (reductions >= 4096) {
                    value = -searchBest(variation, -alpha - 1, -alpha, ply + 1, max(1, pvDepth - 2), &bestReply, FALSE,
                                        FALSE, NO_MOVE);
                }

                if (value > alpha) {
                    value =
                        -searchBest(variation, -alpha - 1, -alpha, ply + 1, pvDepth, &bestReply, FALSE, FALSE, NO_MOVE);
                }
            }

            if (pvNode && value > alpha && value < beta) {
                value = -searchBest(variation, -beta, -alpha, ply + 1, pvDepth, &bestReply, TRUE, FALSE, NO_MOVE);
            }
        }

        assert(value >= VALUE_MATED && value <= -VALUE_MATED);

        if (nodeWasBlocked) {
            resetNodeUsage(position->hashKey);
        }

        unmakeLastMove(variation);
        numMovesPlayed++;

        if (quietMove && inCheck == FALSE) {
            quietMoveIndex[quietMoveCount++] = historyIndex(currentMove, position);
        }

        if (value > best) {
            best = value;

            if (best > alpha) {
                alpha = best;
                *bestMove = currentMove;

                if (pvNode) {
                    appendMoveToPv(&(variation->plyInfo[ply].pv), &(variation->plyInfo[ply - 1].pv), currentMove);
                }

                if (best >= beta) {
                    break; /* cut-off */
                }
            }
        }
    }

    /* No legal move was found. Check if it's mate or stalemate. */
    /* --------------------------------------------------------- */
    if (best == VALUE_MATED) {
        if (excludeMove != NO_MOVE && excludeMove != NULLMOVE) {
            return beta - 1;
        }

        if (inCheck) {
            best = VALUE_MATED + ply; /* mate */
        } else {
            best = variation->drawScore[position->activeColor]; /* stalemate */
        }
    }

    /* Update per-ply move history */
    if (inCheck == FALSE && quietMoveCount > 0 && (excludeMove == NO_MOVE || excludeMove == NULLMOVE)) {
        const int bonus = (pvNode ? 2 : 1);

        for (int i = 0; i < quietMoveCount; i++) {
            variation->moveHistory[ply][quietMoveIndex[i]].freq += bonus;
        }

        /* Update move ordering heuristics. */
        /* --------------------------------- */
        if (*bestMove != NO_MOVE && moveIsQuietInPosition(*bestMove, position)) {
            variation->moveHistory[ply][historyIndex(*bestMove, position)].succ += bonus;

            Move killerMove = *bestMove;
            const Piece movingPiece = position->piece[getFromSquare(killerMove)];

            setMoveValue(&killerMove, movingPiece);
            registerKillerMove(&variation->plyInfo[ply], killerMove);
            updateCounterMoves(variation, ply, killerMove);

            if (ply >= 2) {
                updateFollowupMoves(variation, ply, killerMove);
            }
        }
    }

storeResult:

    /* Store the value in the transposition table. */
    /* ------------------------------------------- */
    if ((excludeMove == NO_MOVE || excludeMove == NULLMOVE) && variation->searchStatus == SEARCH_STATUS_RUNNING) {
        if (best >= beta) {
            hashentryFlag = HASHVALUE_LOWER_LIMIT;
        } else {
            hashentryFlag = (pvNode && *bestMove != NO_MOVE ? HASHVALUE_EXACT : HASHVALUE_UPPER_LIMIT);
        }

        setHashentry(getSharedHashtable(), hashKey, calcHashtableValue(best, ply),
                     (UINT8)(restDepth + HASH_DEPTH_OFFSET), packedMove(*bestMove), hashentryFlag,
                     (INT16)getStaticValue(variation));
    }

    return best;
}

void copyPvFromHashtable(Variation *variation, const int pvIndex, PrincipalVariation *pv, const Move bestBaseMove)
{
    Move bestMove = NO_MOVE;

    if (pvIndex == 0) {
        bestMove = bestBaseMove;
    } else {
        Hashentry *tableHit = getHashentry(getSharedHashtable(), variation->singlePosition.hashKey);

        if (tableHit != NULL) {
            Move currentMove = (Move)getHashentryMove(tableHit);

            if (moveIsLegal(&variation->singlePosition, currentMove)) {
                bestMove = (Move)getHashentryMove(tableHit);
            }
        }
    }

    if (bestMove != NO_MOVE && pvIndex < MAX_DEPTH) {
        pv->move[pvIndex] = (UINT16)bestMove;
        pv->move[pvIndex + 1] = (UINT16)NO_MOVE;
        pv->length = pvIndex + 1;
        makeMove(variation, bestMove);
        copyPvFromHashtable(variation, pvIndex + 1, pv, bestBaseMove);
        unmakeLastMove(variation);
    } else {
        pv->move[pvIndex] = (UINT16)NO_MOVE;
        pv->length = pvIndex;
    }
}

static void copyPvToHashtable(Variation *variation, PrincipalVariation *pv, const int pvIndex)
{
    Move move = (Move)pv->move[pvIndex];

    if (pvIndex < pv->length && moveIsLegal(&variation->singlePosition, move)) {
        UINT8 importance = (UINT8)HASH_DEPTH_OFFSET;
        bool entryExists = FALSE;
        Move bestMove = NO_MOVE;
        Hashentry *tableHit = getHashentry(getSharedHashtable(), variation->singlePosition.hashKey);

        if (tableHit != 0) {
            bestMove = (Move)getHashentryMove(tableHit);
            importance = max(importance, getHashentryImportance(tableHit));

            if (bestMove != NO_MOVE && movesAreEqual(bestMove, move)) {
                entryExists = TRUE;
            }
        }

        if (entryExists == FALSE) {
            UINT8 hashentryFlag = HASHVALUE_LOWER_LIMIT;

            /* Store the move in the transposition table. */
            /* ------------------------------------------- */

            setHashentry(getSharedHashtable(), variation->singlePosition.hashKey, VALUE_MATED, importance,
                         packedMove(move), hashentryFlag, getEvalValue(variation));
        }

        makeMove(variation, move);
        copyPvToHashtable(variation, pv, pvIndex + 1);
        unmakeLastMove(variation);
    }
}

static void registerBestMove(Variation *variation, Move *move, const int value)
{
    if (variation->searchStatus == SEARCH_STATUS_RUNNING) {
        setMoveValue(move, value);
        variation->bestBaseMove = *move;

        if (variation->iteration > 4 && variation->numberOfCurrentBaseMove > 1) {
            variation->bestMoveChangeCount += 256;
        }
    }
}

static int getBaseMoveValue(Variation *variation, const Move move, const int alpha, const int beta)
{
    int depth = variation->iteration;
    int value;
    Move bestReply;

    assert(alpha >= VALUE_MATED);
    assert(alpha <= -VALUE_MATED);
    assert(beta >= VALUE_MATED);
    assert(beta <= -VALUE_MATED);
    assert(alpha < beta);

    makeMoveFast(variation, move);

    if (variation->nodes > GUI_NODE_COUNT_MIN && variation->threadNumber == 0 && variation->handleSearchEvent != 0) {
        getGuiSearchMutex();
        variation->handleSearchEvent(SEARCHEVENT_NEW_BASEMOVE, variation);
        releaseGuiSearchMutex();
    }

    if (activeKingIsSafe(&variation->singlePosition) == FALSE) {
        variation->plyInfo[0].currentMoveIsCheck = TRUE;
        depth++;
    } else {
        variation->plyInfo[0].currentMoveIsCheck = FALSE;
    }

    value = -searchBest(variation, -beta, -alpha, 1, depth - 1, &bestReply, TRUE, FALSE, NO_MOVE);

    unmakeLastMove(variation);

    return value;
}

int getPvScoreType(int value, int alpha, int beta)
{
    if (value <= alpha) {
        return HASHVALUE_UPPER_LIMIT;
    } else if (value >= beta) {
        return HASHVALUE_LOWER_LIMIT;
    } else {
        return HASHVALUE_EXACT;
    }
}

static void sendPvInfo(Variation *variation, const int eventType)
{
    if ((variation->nodes > GUI_NODE_COUNT_MIN || eventType == SEARCHEVENT_PLY_FINISHED) &&
        variation->threadNumber == 0 && variation->handleSearchEvent != 0) {
        int i;

        getGuiSearchMutex();

        for (i = 0; i < numPvs && variation->pv[i].length > 0; i++) {
            variation->pvId = i;
            variation->handleSearchEvent(eventType, variation);
        }

        releaseGuiSearchMutex();
    }
}

static void exploreBaseMoves(Variation *variation, Movelist *basemoves, const int aspirationWindow)
{
    const int previousBest = variation->previousBest;
    const int ply = 0;
    Position *position = &variation->singlePosition;
    const bool fullWindow = (bool)(variation->iteration <= 3);

    variation->optimism = 36 * previousBest / (abs(previousBest) + 22);
    int window = aspirationWindow, best;
    bool exactValueFound = FALSE;
    const int staticValue = getEvalValue(variation);
    int alpha = (fullWindow ? VALUE_MATED : max(VALUE_MATED, previousBest - window));
    int beta = (fullWindow ? -VALUE_MATED : min(-VALUE_MATED, previousBest + window));

    variation->failingLow = FALSE;
    variation->selDepth = variation->iteration;
    initializePvsOfVariation(variation);

    do {
        int pvCount = 0, worstValue = VALUE_MATED;
        const int numPvLimit = min(basemoves->numberOfMoves, numPvs);

        initializeMoveValues(basemoves);
        resetPvsOfVariation(variation);
        best = VALUE_MATED;

        if (tbAvailable && position->numberOfPieces[WHITE] + position->numberOfPieces[BLACK] <= 7) {
            int i;
            bool allProbed = TRUE;

            for (i = 0; i < basemoves->numberOfMoves; i++) {
                Move currentMove = basemoves->moves[i];
                Move tbMove = NO_MOVE;
                int score;

                makeMoveFast(variation, currentMove);
                score = probeTablebaseDTZ(&variation->singlePosition, &tbMove);
                unmakeLastMove(variation);

                if (score != TABLEBASE_ERROR) {
                    // We need to invert the score because probeTablebaseDTZ returns it from the side to move's
                    // perspective
                    setMoveValue(&basemoves->moves[i], -score);
                } else {
                    allProbed = FALSE;
                }
            }

            if (allProbed) {
                variation->tbHits++;
                sortMoves(basemoves);

                // Set the best move as found in TBs
                variation->bestBaseMove = basemoves->moves[0];

                // Initialize PV with the best move
                variation->completePv.move[0] = (UINT16)variation->bestBaseMove;
                variation->completePv.move[1] = (UINT16)NO_MOVE;
                variation->completePv.length = 1;
                variation->completePv.score = getMoveValue(variation->bestBaseMove);
            }
        }

        for (variation->numberOfCurrentBaseMove = 1; variation->numberOfCurrentBaseMove <= basemoves->numberOfMoves;
             variation->numberOfCurrentBaseMove++) {
            int value;
            const int icm = variation->numberOfCurrentBaseMove - 1;
            const bool searchBelowBest = fullWindow || numPvs > 1;
            const int searchAlpha = (searchBelowBest ? alpha : max(alpha, best));

            resetPlyInfo(variation);
            variation->currentBaseMove = basemoves->moves[icm];
            variation->plyInfo[ply].indexCurrentMove = historyIndex(variation->currentBaseMove, position);

            value = getBaseMoveValue(variation, basemoves->moves[icm], searchAlpha, beta);

            if (variation->searchStatus != SEARCH_STATUS_RUNNING && variation->iteration > 1) {
                break;
            }

            if (icm == 0 || value > searchAlpha) {
                PrincipalVariation pv;

                setMoveValue(&basemoves->moves[icm], value);
                pv.score = value;
                pv.scoreType = getPvScoreType(value, searchAlpha, beta);
                appendMoveToPv(&(variation->plyInfo[0].pv), &pv, basemoves->moves[icm]);
                addPvByScore(variation, &pv);

                if (++pvCount >= numPvLimit) {
                    sendPvInfo(variation, SEARCHEVENT_NEW_PV);
                }

                if (icm == 0 || value > best) {
                    registerBestMove(variation, &basemoves->moves[icm], value);

                    if (value > best && value < beta) {
                        variation->completePv = pv;
                    }
                }
            }

            if (value > best) {
                best = value;

                if (value >= beta && numPvs == 1) {
                    break;
                }
            }
        }

        /* Store the value in the transposition table. */
        /* ------------------------------------------- */
        if (variation->searchStatus == SEARCH_STATUS_RUNNING) {
            UINT8 hashentryFlag;
            const int depth = variation->iteration;
            const Move bestMove = variation->bestBaseMove;

            if (best > alpha) {
                hashentryFlag = (best >= beta ? HASHVALUE_LOWER_LIMIT : HASHVALUE_EXACT);
            } else {
                hashentryFlag = HASHVALUE_UPPER_LIMIT;
            }

            setHashentry(getSharedHashtable(), position->hashKey, calcHashtableValue(best, ply),
                         (UINT8)(depth + HASH_DEPTH_OFFSET), packedMove(bestMove), hashentryFlag, (INT16)staticValue);
        }

        worstValue = (numPvs == 1 ? best : max(VALUE_MATED + 3, variation->pv[numPvs - 1].score));

        if (best >= beta) {
            beta = min(-VALUE_MATED, best + window);
        } else if (worstValue <= alpha && worstValue > VALUE_MATED + 2) {
            alpha = max(VALUE_MATED, alpha - window);
            variation->failingLow = TRUE;
        } else {
            exactValueFound = TRUE; /* exact value found */
        }

        window = window + window / 2;

        sortMoves(basemoves);

        assert(fullWindow == TRUE || movesAreEqual(basemoves->moves[0], variation->bestBaseMove));

        if (variation->threadNumber == 0) {
            copyPvToHashtable(variation, &variation->completePv, 0);
        }
    } while (variation->searchStatus == SEARCH_STATUS_RUNNING && exactValueFound == FALSE);

    variation->pv[0].score = getMoveValue(variation->bestBaseMove);

    if (variation->threadNumber == 0 && variation->iteration > 1 && variation->completePv.length <= 1) {
        PrincipalVariation tmpPv;

        copyPvFromHashtable(variation, 0, &tmpPv, variation->bestBaseMove);

        if (tmpPv.length > 1) {
            variation->completePv = tmpPv;
        }
    }

    sendPvInfo(variation, SEARCHEVENT_PLY_FINISHED);
}

Move search(Variation *variation, Movelist *acceptableSolutions)
{
    Movelist movelist;
    long timeTarget;
    int stableIterationCount = 0;
    int iv1 = 0, iv2 = 0, iv3 = 0;

    if (resetSharedHashtable) {
        resetHashtable(getSharedHashtable());
        resetSharedHashtable = FALSE;
    }

    resetHistoryValues(variation);
    memset(variation->moveHistory, 0, sizeof(variation->moveHistory));

    variation->ply = 0;
    resetFinnyTable(&variation->finnyTable);
    refreshAccumulator(&variation->singlePosition, &variation->plyInfo[0].accumulator, &variation->finnyTable);
    variation->ownColor = variation->singlePosition.activeColor;
    variation->optimism = 0;
    variation->nodes = variation->nodesAtTimeCheck = 0;
    variation->startTimeProcess = getProcessTimestamp();
    variation->timestamp = variation->startTime + 1;
    variation->hashSendTimestamp = variation->startTime;
    variation->tbHits = 0;
    variation->numPvUpdates = 0;
    variation->terminateSearchOnPonderhit = FALSE;
    variation->previousBest = getStaticValue(variation);
    variation->bestBaseMove = NO_MOVE;
    variation->failingLow = FALSE;
    initializePlyInfo(variation);
    getLegalMoves(variation, &movelist);

    variation->numberOfBaseMoves = movelist.numberOfMoves;
    setMoveValue(&variation->bestBaseMove, VALUE_MATED);

    for (variation->iteration = 1; variation->iteration <= MAX_DEPTH; variation->iteration++) {
        long calculationTime;
        int iterationValue, aspirationWindow;

        variation->ply = 0;

        aspirationWindow = min(6, max(4, (abs(iv1 - iv2) + abs(iv2 - iv3)) / 2));
        exploreBaseMoves(variation, &movelist, aspirationWindow);
        calculationTime = (unsigned long)(getTimestamp() - variation->startTime);

        iv3 = iv2;
        iv2 = iv1;
        iv1 = iterationValue = getMoveValue(variation->bestBaseMove);

        variation->previousBest = iterationValue;

        assert(calculationTime >= 0);

        if (acceptableSolutions != 0 && listContainsMove(acceptableSolutions, variation->bestBaseMove)) {
            stableIterationCount++;
        } else {
            stableIterationCount = 0;
        }

        /* Check for a fail low. */
        /* --------------------- */

        if (variation->numberOfBaseMoves == 1) {
            timeTarget = (19 * variation->timeTarget) / 256;
        } else {
            const int timeWeight = 160 + (223 * variation->bestMoveChangeCount) / 256;

            timeTarget = (timeWeight * variation->timeTarget) / 256;
        }

        variation->bestMoveChangeCount = (17 * variation->bestMoveChangeCount) / 32;

        getGuiSearchMutex();

        if (variation->threadNumber == 0 && variation->searchStatus == SEARCH_STATUS_RUNNING &&
            variation->iteration > 8 && variation->timeLimit != 0 && calculationTime >= timeTarget) {
            if (variation->ponderMode) {
                variation->terminateSearchOnPonderhit = TRUE;
            } else {
                variation->searchStatus = SEARCH_STATUS_TERMINATE;
            }
        }

        if (variation->searchStatus == SEARCH_STATUS_RUNNING &&
            (getMoveValue(variation->bestBaseMove) <= VALUE_MATED + variation->iteration ||
             getMoveValue(variation->bestBaseMove) >= -VALUE_MATED - variation->iteration)) {
            variation->searchStatus = SEARCH_STATUS_TERMINATE;
        }

        if (variation->searchStatus == SEARCH_STATUS_RUNNING && variation->iteration == MAX_DEPTH) {
            variation->searchStatus = SEARCH_STATUS_TERMINATE;
        }

        if (acceptableSolutions != 0 && stableIterationCount >= 1 &&
            (getMoveValue(variation->bestBaseMove) > 20000 ||
             (stableIterationCount >= 2 &&
              (getMoveValue(variation->bestBaseMove) >= 13 || (getTimestamp() - variation->startTime) >= 3000)))) {
            variation->searchStatus = SEARCH_STATUS_TERMINATE;
        }

        if (variation->searchStatus != SEARCH_STATUS_RUNNING) {
            variation->terminateSearchOnPonderhit = TRUE;
            variation->searchStatus = SEARCH_STATUS_TERMINATE;

            variation->finishTime = getTimestamp();
            variation->finishTimeProcess = getProcessTimestamp();

            if (variation->threadNumber == 0) {
                incrementDate(getSharedHashtable());

                if (variation->handleSearchEvent != 0) {
                    variation->handleSearchEvent(SEARCHEVENT_SEARCH_FINISHED, variation);
                }
            }
        }

        if (variation->searchStatus != SEARCH_STATUS_RUNNING) {
            releaseGuiSearchMutex();
            break;
        }

        releaseGuiSearchMutex();
    }

    if (statCount1 != 0 || statCount2 != 0) {
        logReport("statCount1=%lld statCount2=%lld (%lld%%) \n", statCount1, statCount2,
                  (statCount2 * 100) / max(1, statCount1));
    }

    return variation->bestBaseMove;
}

int initializeModuleSearch(void)
{
    for (int i = 0; i < 1024; i++) {
        log1024[i] = (int)(1024.0 * log((double)(i + 1.0)));
    }

    return 0;
}

int testModuleSearch(void)
{
    return 0;
}

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

#include "evaluation.h"

#include "fen.h"
#include "io.h"
#include "position.h"
#include "search.h"
#include "tablebase.h"
#include "tools.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Bitboard passedPawnCorridor[2][_64_];
Bitboard candidateDefenders[2][_64_];

int getValue(const Position *position, Accumulator *acc, int optimism)
{
    assert(acc != NULL);

    // Use precomputed materialBalance (white minus black, Stockfish values) for
    // small-net selection, and materialCount (total material blend) for the formula.
    int se = (position->activeColor == WHITE) ? position->materialBalance : -position->materialBalance;
    const bool smallNet = abs(se) > 962;

    int psqt, positional;

    if (smallNet) {
        evaluateNnueWithAccumulatorFull((Position *)position, acc, &psqt, &positional);
    } else {
        evaluateBigNnueWithAccumulatorFull((Position *)position, acc, &psqt, &positional);
    }

    int nnue = (125 * psqt + 131 * positional) / 128;

    // Re-evaluate the position when higher eval accuracy is worth the time spent
    if (smallNet && (abs(nnue) < 277)) {
        evaluateBigNnueWithAccumulatorFull((Position *)position, acc, &psqt, &positional);
        nnue = (125 * psqt + 131 * positional) / 128;
    }

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = abs(psqt - positional);
    optimism += optimism * nnueComplexity / 476;
    nnue -= nnue * nnueComplexity / 18236;

    int material = position->materialCount;

    int v = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;

    // Final scaling to centipawns
    int a = win_rate_scaling((Position *)position);
    v = v * 100 / a;

    // Damp down the evaluation linearly when shuffling
    v -= v * position->halfMoveClock / 199;

    // Guarantee evaluation does not hit the tablebase range
    // Stockfish uses VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1
    // i.e. [-31506, 31506] with its mate value 32000.
    // Protector's mate value is 30000, so we use a similar range relative to mate.
    int min_v = -30000 + 2 * 246 + 1 + 1; // approx -29506
    int max_v = 30000 - 2 * 246 - 1 - 1;  // approx 29506
    if (v < min_v)
        v = min_v;
    if (v > max_v)
        v = max_v;

    return v;
}

bool pawnIsPassed(const Position *position, const Square pawnSquare, const Color pawnColor)
{
    const Color defenderColor = opponent(pawnColor);
    const Bitboard corridor = passedPawnCorridor[pawnColor][pawnSquare];
    const Bitboard defenders =
        position->piecesOfType[PAWN | defenderColor] & (candidateDefenders[pawnColor][pawnSquare] | corridor);

    if (defenders == EMPTY_BITBOARD) {
        const Bitboard blockers = position->piecesOfType[PAWN | pawnColor] & corridor;

        return (bool)(blockers == EMPTY_BITBOARD);
    }

    return FALSE;
}

bool hasBishopPair(const Position *position, const Color color)
{
    const Bitboard bishops = position->piecesOfType[BISHOP | color];

    return (bool)((bishops & lightSquares) != EMPTY_BITBOARD && (bishops & darkSquares) != EMPTY_BITBOARD);
}

bool hasWinningPotential(Position *position, Color color)
{
    if (position->piecesOfType[QUEEN | color] != EMPTY_BITBOARD ||
        position->piecesOfType[ROOK | color] != EMPTY_BITBOARD) {
        return TRUE;
    }

    const Bitboard bishops = position->piecesOfType[BISHOP | color];
    const int numBishops = getNumberOfSetSquares(bishops);

    if (numBishops >= 2 || (numBishops > 0 && position->piecesOfType[KNIGHT | color] != EMPTY_BITBOARD)) {
        return TRUE;
    }

    return FALSE;
}

int initializeModuleEvaluation(void)
{
    Square square;

    ITERATE (square) {
        Color color;

        for (color = WHITE; color <= BLACK; color++) {
            passedPawnCorridor[color][square] = candidateDefenders[color][square] = EMPTY_BITBOARD;
        }
    }

    ITERATE (square) {
        const File squarefile = file(square);
        const Rank squarerank = rank(square);
        Square kingsquare, catchersquare;

        ITERATE (kingsquare) {
            const File kingsquarefile = file(kingsquare);
            const Rank kingsquarerank = rank(kingsquare);

            if (kingsquarefile == squarefile) {
                if (kingsquarerank > squarerank) {
                    setSquare(passedPawnCorridor[WHITE][square], kingsquare);
                }

                if (kingsquarerank < squarerank) {
                    setSquare(passedPawnCorridor[BLACK][square], kingsquare);
                }
            }
        }

        ITERATE (catchersquare) {
            if (((file(catchersquare) > squarefile) ? (file(catchersquare) - squarefile)
                                                    : (squarefile - file(catchersquare))) == 1) {
                if (rank(catchersquare) > squarerank) {
                    setSquare(candidateDefenders[WHITE][square], catchersquare);
                }

                if (rank(catchersquare) < squarerank) {
                    setSquare(candidateDefenders[BLACK][square], catchersquare);
                }
            }
        }
    }

    return 0;
}

bool flipTest(Position *position)
{
    int v1, v2;
    Position flippedPosition;
    Accumulator acc1, acc2;
    FinnyTable *finny = malloc(sizeof(FinnyTable));
    if (!finny) {
        logReport("Failed to allocate FinnyTable in flipTest\n");
        return FALSE;
    }

    resetFinnyTable(finny);
    refreshAccumulator(position, &acc1, finny);
    v1 = getValue(position, &acc1, 0);

    memcpy(&flippedPosition, position, sizeof(Position));
    flipPosition(&flippedPosition);
    resetFinnyTable(finny);
    refreshAccumulator(&flippedPosition, &acc2, finny);
    v2 = getValue(&flippedPosition, &acc2, 0);

    free(finny);
    return (bool)(v1 == v2);
}

int testModuleEvaluation(void)
{
    typedef struct {
        const char *fen;
        Move moves[32];
        int numMoves;
        const char *description;
        int expectedEval;
    } TestCase;

    TestCase cases[] = {
        {FEN_GAMESTART,
         {getOrdinaryMove(E2, E4), getOrdinaryMove(C7, C5), getOrdinaryMove(E4, E5), getOrdinaryMove(D7, D5),
          getOrdinaryMove(E5, D6)},
         5,
         "Startpos - EP capture",
         36},
        {"r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
         {getOrdinaryMove(E1, G1), getOrdinaryMove(F8, E7), getOrdinaryMove(D2, D4), getOrdinaryMove(E5, D4)},
         4,
         "Spanish - Castling and capture",
         58},
        {"r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
         {getOrdinaryMove(D7, D6), getOrdinaryMove(D2, D4), getOrdinaryMove(C5, D4), getOrdinaryMove(F3, D4)},
         4,
         "Sicilian - Normal moves and capture",
         -27},
        {"r1b2rk1/pp1nbppp/2p1pn2/q2p2B1/2PP4/2N1PN2/PPQ2PPP/2R1KB1R w K - 3 9",
         {getOrdinaryMove(H2, H3), getOrdinaryMove(H7, H6), getOrdinaryMove(G5, H4), getOrdinaryMove(F8, E8)},
         4,
         "QGD - Normal moves",
         53},
        {"r4rk1/pp3ppp/2pbbn2/3p4/3P4/2N1PN2/PPQ1BPPP/R4RK1 b - - 5 12",
         {getOrdinaryMove(A7, A6), getOrdinaryMove(A2, A3), getOrdinaryMove(H7, H6), getOrdinaryMove(H2, H3)},
         4,
         "Middle game - Normal moves",
         -563},
        {"r3k2r/pppb1ppp/2n1pn2/8/2PP4/2N2N2/PP2BPPP/R2QK2R w KQkq - 0 1",
         {getOrdinaryMove(E1, G1), getOrdinaryMove(E8, G8), getOrdinaryMove(A2, A3), getOrdinaryMove(A7, A6)},
         4,
         "Advantage - Castlings",
         759},
        {"2r2rk1/1p1q1ppp/p1p1p3/3p4/2PP4/PP1QP3/5PPP/2R2RK1 b - - 0 1",
         {getOrdinaryMove(C8, C7), getOrdinaryMove(C1, C2), getOrdinaryMove(F8, C8), getOrdinaryMove(F1, C1)},
         4,
         "Middle heavy - Normal moves",
         -2},
        {"8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1",
         {getOrdinaryMove(E3, F4), getOrdinaryMove(E6, F6), getOrdinaryMove(F4, G4), getOrdinaryMove(F6, G6)},
         4,
         "Endgame Pawn - King moves",
         4},
        {"8/4P3/8/8/8/8/8/k6K w - - 0 1",
         {getOrdinaryMove(H1, G1), getOrdinaryMove(A1, B1), getOrdinaryMove(G1, F1), getOrdinaryMove(B1, C1),
          getPackedMove(E7, E8, WHITE_QUEEN)},
         5,
         "Promotion - Pawn promotion",
         -449},
        {"8/8/8/8/8/2k5/2r5/1K1Q4 w - - 0 1",
         {getOrdinaryMove(D1, C2), getOrdinaryMove(C3, C2), getOrdinaryMove(B1, A2), getOrdinaryMove(C2, B2)},
         4,
         "Endgame Queen/Rook - Captures and king moves",
         -24}};

    Variation *variation = calloc(1, sizeof(Variation));
    int i, j;

    for (i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
        initializeVariation(variation, cases[i].fen);

        for (j = 0; j < cases[i].numMoves; j++) {
            makeMoveFast(variation, cases[i].moves[j]);
            initializePlyInfo(variation);
        }

        int eval = getEvalValue(variation);

        if (eval != cases[i].expectedEval) {
            logReport("Eval value mismatch Test Case %d (%s): got %d, expected %d\n", i, cases[i].description, eval,
                      cases[i].expectedEval);
            free(variation);
            return -1;
        }
    }

    free(variation);
    return 0;
}

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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "position.h"
#include "fen.h"
#include "io.h"
#include "evaluation.h"
#include "tools.h"
#include "tablebase.h"

Bitboard passedPawnCorridor[2][_64_];
Bitboard candidateDefenders[2][_64_];

static int simple_eval(const Position * pos) {
    Color c = pos->activeColor;
    Color opp = opponent(c);
    int material = 208 * (getNumberOfSetSquares(pos->piecesOfType[PAWN | c]) - getNumberOfSetSquares(pos->piecesOfType[PAWN | opp]));
    
    material += 781 * (getNumberOfSetSquares(pos->piecesOfType[KNIGHT | c]) - getNumberOfSetSquares(pos->piecesOfType[KNIGHT | opp]));
    material += 825 * (getNumberOfSetSquares(pos->piecesOfType[BISHOP | c]) - getNumberOfSetSquares(pos->piecesOfType[BISHOP | opp]));
    material += 1276 * (getNumberOfSetSquares(pos->piecesOfType[ROOK | c]) - getNumberOfSetSquares(pos->piecesOfType[ROOK | opp]));
    material += 2538 * (getNumberOfSetSquares(pos->piecesOfType[QUEEN | c]) - getNumberOfSetSquares(pos->piecesOfType[QUEEN | opp]));
    
    return material;
}

static bool use_smallnet(const Position * pos) {
    return abs(simple_eval(pos)) > 962;
}

int getValue(const Position * position, Accumulator * acc, int optimism)
{
    assert(acc != NULL);

    bool smallNet = use_smallnet(position);
    int psqt, positional;

    if (smallNet) {
        evaluateNnueWithAccumulatorFull((Position *)position, acc, &psqt, &positional);
    } else {
        evaluateBigNnueWithAccumulatorFull((Position *)position, acc, &psqt, &positional);
    }

    psqt /= 16;
    positional /= 16;

    int nnue = (125 * psqt + 131 * positional) / 128;

    // Re-evaluate the position when higher eval accuracy is worth the time spent
    if (smallNet && (abs(nnue) < 277))
    {
        evaluateBigNnueWithAccumulatorFull((Position *)position, acc, &psqt, &positional);
        psqt /= 16;
        positional /= 16;
        nnue = (125 * psqt + 131 * positional) / 128;
        smallNet = FALSE;
    }

    // Blend optimism and eval with nnue complexity
    int nnueComplexity = abs(psqt - positional);
    optimism += optimism * nnueComplexity / 476;
    nnue -= nnue * nnueComplexity / 18236;

    int pawn_count = getNumberOfSetSquares(position->piecesOfType[PAWN | WHITE]) + getNumberOfSetSquares(position->piecesOfType[PAWN | BLACK]);
    
    // Use Stockfish piece values for material calculation in the formula
    // This is the sum of both sides, identical to pos.non_pawn_material() in evaluate.cpp
    int non_pawn_material = 
        781 * (getNumberOfSetSquares(position->piecesOfType[KNIGHT | WHITE]) + getNumberOfSetSquares(position->piecesOfType[KNIGHT | BLACK])) +
        825 * (getNumberOfSetSquares(position->piecesOfType[BISHOP | WHITE]) + getNumberOfSetSquares(position->piecesOfType[BISHOP | BLACK])) +
        1276 * (getNumberOfSetSquares(position->piecesOfType[ROOK | WHITE]) + getNumberOfSetSquares(position->piecesOfType[ROOK | BLACK])) +
        2538 * (getNumberOfSetSquares(position->piecesOfType[QUEEN | WHITE]) + getNumberOfSetSquares(position->piecesOfType[QUEEN | BLACK]));

    int material = 534 * pawn_count + non_pawn_material;
    
    int v = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;

    // Damp down the evaluation linearly when shuffling
    v -= v * position->halfMoveClock / 199;

    // Guarantee evaluation does not hit the tablebase range
    // Stockfish uses VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1
    // i.e. [-31506, 31506] with its mate value 32000.
    // Protector's mate value is 30000, so we use a similar range relative to mate.
    int min_v = -30000 + 2 * 246 + 1 + 1; // approx -29506
    int max_v = 30000 - 2 * 246 - 1 - 1; // approx 29506
    if (v < min_v) v = min_v;
    if (v > max_v) v = max_v;

    return v;
}

bool pawnIsPassed(const Position * position, const Square pawnSquare,
                  const Color pawnColor)
{
   const Color defenderColor = opponent(pawnColor);
   const Bitboard corridor = passedPawnCorridor[pawnColor][pawnSquare];
   const Bitboard defenders = position->piecesOfType[PAWN | defenderColor] &
      (candidateDefenders[pawnColor][pawnSquare] | corridor);

   if (defenders == EMPTY_BITBOARD)
   {
      const Bitboard blockers = position->piecesOfType[PAWN | pawnColor] &
         corridor;

      return (bool) (blockers == EMPTY_BITBOARD);
   }

   return FALSE;
}

bool hasBishopPair(const Position * position, const Color color)
{
   const Bitboard bishops = position->piecesOfType[BISHOP | color];

   return (bool) ((bishops & lightSquares) != EMPTY_BITBOARD &&
                  (bishops & darkSquares) != EMPTY_BITBOARD);
}

bool hasWinningPotential(Position * position, Color color)
{
   if (position->piecesOfType[QUEEN | color] != EMPTY_BITBOARD ||
       position->piecesOfType[ROOK | color] != EMPTY_BITBOARD)
   {
      return TRUE;
   }

   const Bitboard bishops = position->piecesOfType[BISHOP | color];
   const int numBishops = getNumberOfSetSquares(bishops);

   if (numBishops >= 2 ||
       (numBishops > 0 &&
        position->piecesOfType[KNIGHT | color] != EMPTY_BITBOARD))
   {
      return TRUE;
   }

   return FALSE;
}

int initializeModuleEvaluation(void)
{
   Square square;

   ITERATE(square)
   {
      Color color;

      for (color = WHITE; color <= BLACK; color++)
      {
         passedPawnCorridor[color][square] =
            candidateDefenders[color][square] = EMPTY_BITBOARD;
      }
   }

   ITERATE(square)
   {
      const File squarefile = file(square);
      const Rank squarerank = rank(square);
      Square kingsquare, catchersquare;

      ITERATE(kingsquare)
      {
         const File kingsquarefile = file(kingsquare);
         const Rank kingsquarerank = rank(kingsquare);

         if (kingsquarefile == squarefile)
         {
            if (kingsquarerank > squarerank)
            {
               setSquare(passedPawnCorridor[WHITE][square], kingsquare);
            }

            if (kingsquarerank < squarerank)
            {
               setSquare(passedPawnCorridor[BLACK][square], kingsquare);
            }
         }
      }

      ITERATE(catchersquare)
      {
         if (((file(catchersquare) > squarefile) ? (file(catchersquare) - squarefile) : (squarefile - file(catchersquare))) == 1)
         {
            if (rank(catchersquare) > squarerank)
            {
               setSquare(candidateDefenders[WHITE][square], catchersquare);
            }

            if (rank(catchersquare) < squarerank)
            {
               setSquare(candidateDefenders[BLACK][square], catchersquare);
            }
         }
      }
   }

   return 0;
}

bool flipTest(Position * position)
{
   int v1, v2;
   Position flippedPosition;
   Accumulator acc1, acc2;

   refreshAccumulator(position, &acc1);
   v1 = getValue(position, &acc1, 0);
   
   memcpy(&flippedPosition, position, sizeof(Position));
   flipPosition(&flippedPosition);
   refreshAccumulator(&flippedPosition, &acc2);
   v2 = getValue(&flippedPosition, &acc2, 0);

   return (bool) (v1 == v2);
}

int testModuleEvaluation(void)
{
   return 0;
}

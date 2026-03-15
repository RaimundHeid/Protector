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

int getValue(const Position * position, Accumulator * acc)
{
   return evaluateNnueWithAccumulator((Position *)position, acc);
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
   return (bool) (getPieceCount(position, (Piece) (WHITE_BISHOP_LIGHT | color)) > 0 &&
                  getPieceCount(position, (Piece) (WHITE_BISHOP_DARK | color)) > 0);
}

bool hasWinningPotential(Position * position, Color color)
{
   const int numQueens = getPieceCount(position, (Piece) (QUEEN | color));
   const int numRooks = getPieceCount(position, (Piece) (ROOK | color));
   const int numBishops = getPieceCount(position, (Piece) (BISHOP | color));
   const int numKnights = getPieceCount(position, (Piece) (KNIGHT | color));

   if (numQueens > 0 || numRooks > 0 || numBishops >= 2 ||
       (numBishops > 0 && numKnights > 0))
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

   v1 = getValue(position, NULL);
   memcpy(&flippedPosition, position, sizeof(Position));
   flipPosition(&flippedPosition);
   v2 = getValue(&flippedPosition, NULL);

   return (bool) (v1 == v2);
}

int testModuleEvaluation(void)
{
   return 0;
}

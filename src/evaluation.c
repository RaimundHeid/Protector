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

MaterialInfo materialInfo[MATERIALINFO_TABLE_SIZE];

Bitboard passedPawnCorridor[2][_64_];
Bitboard candidateDefenders[2][_64_];

KingSafetyHashInfo
   kingSafetyHashtable[MAX_THREADS][KINGSAFETY_HASHTABLE_SIZE];

INT32 materialBalance(const Position * position)
{
   const INT32 bishopPairBonus =
      V(VALUE_BISHOP_PAIR_OPENING, VALUE_BISHOP_PAIR_ENDGAME);
   static const INT32 knightBonus = V(0, 5);
   static const INT32 rookMalus = V(5, 0);
   static const INT32 rookPairMalus = V(17, 25);
   static const INT32 rookQueenMalus = V(8, 12);
   static const INT32 pieceUpBonus =
      V(DEFAULTVALUE_PIECE_UP_OPENING, DEFAULTVALUE_PIECE_UP_ENDGAME);

   INT32 balance = 0;
   const int pawnCountWhite = position->numberOfPawns[WHITE] - 5;
   const int pawnCountBlack = position->numberOfPawns[BLACK] - 5;
   const int numWhiteKnights = getPieceCount(position, WHITE_KNIGHT);
   const int numBlackKnights = getPieceCount(position, BLACK_KNIGHT);
   const int knightSaldo = pawnCountWhite * numWhiteKnights -
      pawnCountBlack * numBlackKnights;
   const int numWhiteRooks = getPieceCount(position, WHITE_ROOK);
   const int numBlackRooks = getPieceCount(position, BLACK_ROOK);
   const int rookSaldo = (numWhiteRooks - numBlackRooks) * 3;

   if (getPieceCount(position, (Piece) WHITE_BISHOP_LIGHT) > 0 &&
       getPieceCount(position, (Piece) WHITE_BISHOP_DARK) > 0)
   {
      balance += bishopPairBonus;
   }

   if (getPieceCount(position, (Piece) BLACK_BISHOP_LIGHT) > 0 &&
       getPieceCount(position, (Piece) BLACK_BISHOP_DARK) > 0)
   {
      balance -= bishopPairBonus;
   }

   balance += knightSaldo * knightBonus - rookSaldo * rookMalus;

   if (numWhiteRooks == 2)
   {
      balance -= rookPairMalus + rookQueenMalus;
   }
   else if (numWhiteRooks == 1 && getPieceCount(position, WHITE_QUEEN) > 0)
   {
      balance -= rookQueenMalus;
   }

   if (numBlackRooks == 2)
   {
      balance += rookPairMalus + rookQueenMalus;
   }
   else if (numBlackRooks == 1 && getPieceCount(position, BLACK_QUEEN) > 0)
   {
      balance += rookQueenMalus;
   }

   if (numberOfNonPawnPieces(position, WHITE) >
       numberOfNonPawnPieces(position, BLACK))
   {
      balance += pieceUpBonus;
   }
   else if (numberOfNonPawnPieces(position, BLACK) >
            numberOfNonPawnPieces(position, WHITE))
   {
      balance -= pieceUpBonus;
   }

   return balance;
}

int getValue(const Position * position, Accumulator * acc)
{
   return evaluateNnue((Position *)position, acc);
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

static void initializeMaterialInfoTable(void)
{
   UINT32 signatureWhite, signatureBlack;

   for (signatureWhite = 0; signatureWhite < 648; signatureWhite++)
   {
      for (signatureBlack = 0; signatureBlack < 648; signatureBlack++)
      {
         int numWhiteQueens, numWhiteRooks, numWhiteLightSquareBishops;
         int numWhiteDarkSquareBishops, numWhiteKnights, numWhitePawns;
         int numBlackQueens, numBlackRooks, numBlackLightSquareBishops;
         int numBlackDarkSquareBishops, numBlackKnights, numBlackPawns;
         const UINT32 signature =
            bilateralSignature(signatureWhite, signatureBlack);
         MaterialInfo *mi = &materialInfo[signature];

         getPieceCounters(signature, &numWhiteQueens, &numWhiteRooks,
                          &numWhiteLightSquareBishops,
                          &numWhiteDarkSquareBishops, &numWhiteKnights,
                          &numWhitePawns, &numBlackQueens, &numBlackRooks,
                          &numBlackLightSquareBishops,
                          &numBlackDarkSquareBishops, &numBlackKnights,
                          &numBlackPawns);

         mi->materialBalance =
            V(VALUE_PAWN_OPENING, VALUE_PAWN_ENDGAME) *
            (numWhitePawns - numBlackPawns) +
            V(VALUE_KNIGHT_OPENING, VALUE_KNIGHT_ENDGAME) *
            (numWhiteKnights - numBlackKnights) +
            V(VALUE_BISHOP_OPENING, VALUE_BISHOP_ENDGAME) *
            (numWhiteLightSquareBishops + numWhiteDarkSquareBishops -
             numBlackLightSquareBishops - numBlackDarkSquareBishops) +
            V(VALUE_ROOK_OPENING, VALUE_ROOK_ENDGAME) *
            (numWhiteRooks - numBlackRooks) +
            V(VALUE_QUEEN_OPENING, VALUE_QUEEN_ENDGAME) *
            (numWhiteQueens - numBlackQueens);

         mi->phaseIndex = 4 * (numWhiteQueens + numBlackQueens) +
            2 * (numWhiteRooks + numBlackRooks) +
            numWhiteLightSquareBishops + numWhiteDarkSquareBishops +
            numBlackLightSquareBishops + numBlackDarkSquareBishops +
            numWhiteKnights + numBlackKnights;
      }
   }
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

   initializeMaterialInfoTable();

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

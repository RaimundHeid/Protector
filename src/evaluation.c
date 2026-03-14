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

#define dumpPos dumpPosition(position);

#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include "position.h"
#include "fen.h"
#include "io.h"
#include "evaluation.h"
#include "tools.h"
#include "tablebase.h"

#ifndef NDEBUG
PawnHashInfo localPawnHashtable[PAWN_HASHTABLE_SIZE];
KingSafetyHashInfo localKingSafetyHashtable[KINGSAFETY_HASHTABLE_SIZE];
#endif

KingSafetyHashInfo
   kingSafetyHashtable[MAX_THREADS][KINGSAFETY_HASHTABLE_SIZE];

#define MAX_MOVES_KNIGHT 8
#define MAX_MOVES_BISHOP 13
#define MAX_MOVES_ROOK 14
#define MAX_MOVES_QUEEN 27

INT32 KnightMobilityBonus[MAX_MOVES_KNIGHT + 1];
INT32 BishopMobilityBonus[MAX_MOVES_BISHOP + 1];
INT32 RookMobilityBonus[MAX_MOVES_ROOK + 1];
INT32 QueenMobilityBonus[MAX_MOVES_QUEEN + 1];

const INT32 PstPawn[64] = {
   V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0),      /* rank 1 */
   V(-10, -3), V(-2, -3), V(1, -3), V(5, -3), V(5, -3), V(1, -3), V(-2, -3), V(-10, -3),        /* rank 2 */
   V(-10, -3), V(-2, -3), V(3, -3), V(14, -3), V(14, -3), V(3, -3), V(-2, -3), V(-10, -3),      /* rank 3 */
   V(-10, -3), V(-2, -3), V(6, -3), V(22, -3), V(22, -3), V(6, -3), V(-2, -3), V(-10, -3),      /* rank 4 */
   V(-10, -3), V(-2, -3), V(6, -3), V(14, -3), V(14, -3), V(6, -3), V(-2, -3), V(-10, -3),      /* rank 5 */
   V(-10, -3), V(-2, -3), V(3, -3), V(5, -3), V(5, -3), V(3, -3), V(-2, -3), V(-10, -3),        /* rank 6 */
   V(-10, -3), V(-2, -3), V(1, -3), V(5, -3), V(5, -3), V(1, -3), V(-2, -3), V(-10, -3),        /* rank 7 */
   V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0), V(0, 0),      /* rank 8 */
};

const INT32 PstKnight[64] = {
   V(-48, -42), V(-37, -32), V(-28, -22), V(-24, -17), V(-24, -17), V(-28, -22), V(-37, -32), V(-48, -42),      /* rank 1 */
   V(-33, -32), V(-24, -22), V(-13, -11), V(-8, -6), V(-8, -6), V(-13, -11), V(-24, -22), V(-33, -32),  /* rank 2 */
   V(-18, -22), V(-8, -11), V(0, -2), V(4, 1), V(4, 1), V(0, -2), V(-8, -11), V(-18, -22),      /* rank 3 */
   V(-8, -17), V(0, -6), V(9, 1), V(14, 7), V(14, 7), V(9, 1), V(0, -6), V(-8, -17),    /* rank 4 */
   V(-3, -17), V(4, -6), V(14, 1), V(19, 7), V(19, 7), V(14, 1), V(4, -6), V(-3, -17),  /* rank 5 */
   V(-3, -22), V(4, -11), V(14, -2), V(19, 1), V(19, 1), V(14, -2), V(4, -11), V(-3, -22),      /* rank 6 */
   V(-18, -32), V(-8, -22), V(0, -11), V(4, -6), V(4, -6), V(0, -11), V(-8, -22), V(-18, -32),  /* rank 7 */
   V(-70, -42), V(-24, -32), V(-13, -22), V(-8, -17), V(-8, -17), V(-13, -22), V(-24, -32), V(-70, -42) /* rank 8 */
};

const INT32 PstBishop[64] = {
   V(-15, -21), V(-15, -15), V(-13, -12), V(-11, -9), V(-11, -9), V(-13, -12), V(-15, -15), V(-15, -21),        /* rank 1 */
   V(-6, -15), V(0, -9), V(-1, -6), V(0, -3), V(0, -3), V(-1, -6), V(0, -9), V(-6, -15),        /* rank 2 */
   V(-5, -12), V(-1, -6), V(3, -3), V(1, 0), V(1, 0), V(3, -3), V(-1, -6), V(-5, -12),  /* rank 3 */
   V(-3, -9), V(0, -3), V(1, 0), V(6, 0), V(6, 0), V(1, 0), V(0, -3), V(-3, -9),        /* rank 4 */
   V(-3, -9), V(0, -3), V(1, 0), V(6, 0), V(6, 0), V(1, 0), V(0, -3), V(-3, -9),        /* rank 5 */
   V(-5, -12), V(-1, -6), V(3, -3), V(1, 0), V(1, 0), V(3, -3), V(-1, -6), V(-5, -12),  /* rank 6 */
   V(-6, -15), V(0, -9), V(-1, -6), V(0, -3), V(0, -3), V(-1, -6), V(0, -9), V(-6, -15),        /* rank 7 */
   V(-6, -21), V(-6, -15), V(-5, -12), V(-3, -9), V(-3, -9), V(-5, -12), V(-6, -15), V(-6, -21) /* rank 8 */
};

const INT32 PstRook[64] = {
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1),  /* rank 1 */
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1),  /* rank 2 */
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1),  /* rank 3 */
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1),  /* rank 4 */
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1),  /* rank 5 */
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1),  /* rank 6 */
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1),  /* rank 7 */
   V(-4, 1), V(-2, 1), V(0, 1), V(0, 1), V(0, 1), V(0, 1), V(-2, 1), V(-4, 1)   /* rank 8 */
};

const INT32 PstQueen[64] = {
   V(3, -31), V(3, -21), V(3, -16), V(3, -11), V(3, -11), V(3, -16), V(3, -21), V(3, -31),      /* rank 1 */
   V(3, -21), V(3, -11), V(3, -7), V(3, -2), V(3, -2), V(3, -7), V(3, -11), V(3, -21),  /* rank 2 */
   V(3, -16), V(3, -7), V(3, -2), V(3, 2), V(3, 2), V(3, -2), V(3, -7), V(3, -16),      /* rank 3 */
   V(3, -11), V(3, -2), V(3, 2), V(3, 7), V(3, 7), V(3, 2), V(3, -2), V(3, -11),        /* rank 4 */
   V(3, -11), V(3, -2), V(3, 2), V(3, 7), V(3, 7), V(3, 2), V(3, -2), V(3, -11),        /* rank 5 */
   V(3, -16), V(3, -7), V(3, -2), V(3, 2), V(3, 2), V(3, -2), V(3, -7), V(3, -16),      /* rank 6 */
   V(3, -21), V(3, -11), V(3, -7), V(3, -2), V(3, -2), V(3, -7), V(3, -11), V(3, -21),  /* rank 7 */
   V(3, -31), V(3, -21), V(3, -16), V(3, -11), V(3, -11), V(3, -16), V(3, -21), V(3, -31)       /* rank 8 */
};

const INT32 PstKing[64] = {
   V(106, 0), V(115, 23), V(96, 34), V(76, 46), V(76, 46), V(96, 34), V(115, 23), V(106, 0),    /* rank 1 */
   V(96, 23), V(106, 46), V(85, 58), V(67, 69), V(67, 69), V(85, 58), V(106, 46), V(96, 23),    /* rank 2 */
   V(76, 34), V(85, 58), V(67, 69), V(48, 81), V(48, 81), V(67, 69), V(85, 58), V(76, 34),      /* rank 3 */
   V(67, 46), V(76, 69), V(58, 81), V(38, 93), V(38, 93), V(58, 81), V(76, 69), V(67, 46),      /* rank 4 */
   V(58, 46), V(67, 69), V(48, 81), V(28, 93), V(28, 93), V(48, 81), V(67, 69), V(58, 46),      /* rank 5 */
   V(48, 34), V(58, 58), V(38, 69), V(18, 81), V(18, 81), V(38, 69), V(58, 58), V(48, 34),      /* rank 6 */
   V(38, 23), V(48, 46), V(28, 58), V(9, 69), V(9, 69), V(28, 58), V(48, 46), V(38, 23),        /* rank 7 */
   V(28, 0), V(38, 23), V(18, 34), V(0, 46), V(0, 46), V(18, 34), V(38, 23), V(28, 0)   /* rank 8 */
};

/* -------------------------------------------------------------------------- */

static const int KNIGHT_BONUS_ATTACK = 17;
static const int BISHOP_BONUS_ATTACK = 16;
static const int ROOK_BONUS_ATTACK = 26;
static const int QUEEN_BONUS_ATTACK = 40;

INT32 mvImpact[16];

const int KINGSAFETY_PAWN_MALUS_DEFENDER[3][8] = {
   {30, 0, 5, 15, 20, 25, 25, 25},      /* towards nearest border */
   {55, 0, 15, 40, 50, 55, 55, 55},
   {30, 0, 10, 20, 25, 30, 30, 30}      /* towards center */
};

const int KINGSAFETY_PAWN_BONUS_DEFENDER_DIAG[4][8] = {
   {10, 0, 2, 4, 6, 8, 10, 10},
   {8, 0, 2, 4, 6, 7, 8, 8},
   {6, 0, 2, 3, 4, 5, 6, 6},
   {4, 0, 1, 2, 3, 4, 4, 4}
};

const int KINGSAFETY_PAWN_BONUS_ATTACKER[3][8] = {
   {5, 0, 40, 15, 5, 0, 0, 0},  /* towards nearest border */
   {10, 0, 50, 20, 10, 0, 0, 0},
   {10, 0, 50, 20, 10, 0, 0, 0} /* towards center */
};

#define OWN_COLOR_WEIGHT_DIV 256
#define OWN_COLOR_WEIGHT_KINGSAFETY 296

#define PAWN_EVAL_WEIGHT_OP 256
#define PAWN_EVAL_WEIGHT_EG 258
#define PASSED_PAWN_WEIGHT_OP 98
#define PASSED_PAWN_WEIGHT_EG 132
#define CHAIN_BONUS_WEIGHT_OP 86
#define CHAIN_BONUS_WEIGHT_EG 85
#define KS_PAWNSTRUCTURE_ATTACK_WEIGHT 14
#define KS_PAWNSTRUCTURE_WEIGHT 59
#define HOMELAND_SECURITY_WEIGHT 19
#define KING_SAFETY_MALUS_DIM (500)

#define BISHOP_PIN_OP_VAL 23
#define BISHOP_PIN_EG_VAL 5

int KING_SAFETY_MALUS[KING_SAFETY_MALUS_DIM];

/* -------------------------------------------------------------------------- */

int centerDistance[_64_], centerTaxiDistance[_64_];
int attackPoints[16];
Bitboard butterflySquares[_64_];
Bitboard lateralSquares[_64_];
Bitboard companionFiles[_64_];
Bitboard passedPawnRectangle[2][_64_];
Bitboard passedPawnCorridor[2][_64_];
Bitboard candidateDefenders[2][_64_];   /* excludes squares of rank */
Bitboard candidateSupporters[2][_64_];  /* includes squares of rank */
Bitboard pawnOpponents[2][_64_];
Bitboard kingTrapsRook[2][64];
Bitboard rookBlockers[_64_];
Bitboard centralFiles;
Bitboard kingRealm[2][_64_][_64_];
Bitboard attackingRealm[2];
Bitboard homeland[2];
Bitboard troitzkyArea[2];
Bitboard krprkDrawFiles;
Bitboard A1C1, F1H1, A1B1, G1H1;
Bitboard filesBCFG;
KingAttacks kingAttacks[_64_];
int kingChaseMalus[3][_64_];
INT32 piecePieceAttackBonus[16][16];
MaterialInfo materialInfo[MATERIALINFO_TABLE_SIZE];

/* *INDENT-OFF* */
static const int BONUS_KNIGHT_OUTPOST_HR[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  2,  7,  7,  7,  7,  2,  0,
    0,  3, 10, 14, 14, 10,  3,  0,
    0,  2,  7, 10, 10,  7,  2,  0,
    0,  0,  2,  3,  3,  2,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
};

static const int BONUS_BISHOP_OUTPOST_HR[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  2,  3,  3,  3,  3,  2,  0,
    0,  4,  8,  8,  8,  8,  4,  0,
    0,  2,  4,  4,  4,  4,  2,  0,
    0,  0,  2,  2,  2,  2,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,
};
/* *INDENT-ON* */

static int BONUS_KNIGHT_OUTPOST[_64_];
static int BONUS_BISHOP_OUTPOST[_64_];
static INT32 PAWN_CHAIN_BONUS[_64_];

#ifndef NDEBUG
bool debugEval = FALSE;
#endif

int quad(int y_min, int y_max, int rank);
bool squareIsPawnSafe(const EvaluationBase * base,
                      const Color color, const Square square);
bool hasAttackingBishop(const Position * position,
                        const Color attackingColor, const Square square);
void getPawnInfo(const Position * position, EvaluationBase * base);
bool passerWalks(const Position * position,
                 const Square passerSquare, const Color passerColor);
Piece getDiaBatteryPiece(const Position * position,
                         const Bitboard moves,
                         const Square attackerSquare, const Color kingColor);
Square getPinningPiece(const Position * position,
                       const EvaluationBase * base,
                       const Square pieceSquare, const Color pieceColor);

void addEvalBonusForColor(EvaluationBase * base, const Color color,
                          const INT32 bonus)
{
   if (color == WHITE)
   {
      base->balance += bonus;
   }
   else
   {
      base->balance -= bonus;
   }
}

void addEvalMalusForColor(EvaluationBase * base, const Color color,
                          const INT32 bonus)
{
   if (color == WHITE)
   {
      base->balance -= bonus;
   }
   else
   {
      base->balance += bonus;
   }
}

Color getWinningColor(const Position * position, const int value)
{
   if (position->activeColor == WHITE)
   {
      return (value >= 0 ? WHITE : BLACK);
   }
   else
   {
      return (value <= 0 ? WHITE : BLACK);
   }
}

int getWhiteBishopBlockingIndex(const Position * position,
                                const Bitboard bishopSquares)
{
   const Bitboard ownPawns =
      position->piecesOfType[WHITE_PAWN] & bishopSquares;
   const Bitboard blockingPieces = position->piecesOfColor[BLACK] | ownPawns;
   const Bitboard blockedPawns = (blockingPieces >> 8) & ownPawns;

   return getNumberOfSetSquares(ownPawns) +
      getNumberOfSetSquares(ownPawns & extendedCenter) +
      getNumberOfSetSquares(blockedPawns);
}

int getBlackBishopBlockingIndex(const Position * position,
                                const Bitboard bishopSquares)
{
   const Bitboard ownPawns =
      position->piecesOfType[BLACK_PAWN] & bishopSquares;
   const Bitboard blockingPieces = position->piecesOfColor[WHITE] | ownPawns;
   const Bitboard blockedPawns = (blockingPieces << 8) & ownPawns;

   return getNumberOfSetSquares(ownPawns) +
      getNumberOfSetSquares(ownPawns & extendedCenter) +
      getNumberOfSetSquares(blockedPawns);
}

Bitboard getPromotablePawns(const Position * position, const Color color)
{
   const Color oppColor = opponent(color);
   const Square oppKing = position->king[oppColor];
   const bool lightSquaredBishop = (bool)
      (EMPTY_BITBOARD !=
       (lightSquares & position->piecesOfType[BISHOP | color]));
   const Bitboard pawns = position->piecesOfType[PAWN | color];
   Bitboard supporters;
   File excludeFile;
   Rank promotionRank;
   Square promotionSquare;

   if (color == WHITE)
   {
      excludeFile = (lightSquaredBishop ? FILE_H : FILE_A);
      promotionRank = RANK_8;
   }
   else
   {
      excludeFile = (lightSquaredBishop ? FILE_A : FILE_H);
      promotionRank = RANK_1;
   }

   promotionSquare = getSquare(excludeFile, promotionRank);
   supporters = companionFiles[promotionSquare] & pawns;

   if (distance(oppKing, promotionSquare) <= 1 &&
       supporters == EMPTY_BITBOARD)
   {
      return pawns & ~squaresOfFile[excludeFile];
   }
   else
   {
      return pawns;
   }
}

bool oppositeColoredBishops(const Position * position)
{
   if (getPieceCount(position, (Piece) WHITE_BISHOP_DARK) +
       getPieceCount(position, (Piece) WHITE_BISHOP_LIGHT) == 1 &&
       getPieceCount(position, (Piece) BLACK_BISHOP_DARK) +
       getPieceCount(position, (Piece) BLACK_BISHOP_LIGHT) == 1)
   {
      const Bitboard bishops =
         position->piecesOfType[WHITE_BISHOP] |
         position->piecesOfType[BLACK_BISHOP];

      return (bool) ((lightSquares & bishops) != EMPTY_BITBOARD &&
                     (darkSquares & bishops) != EMPTY_BITBOARD);
   }
   else
   {
      return FALSE;
   }
}

bool passiveKingStopsPawn(const Square kingSquare,
                          const Square pawnSquare, const Color pawnColor)
{
   return testSquare(pawnOpponents[pawnColor][pawnSquare], kingSquare);
}

bool passiveKingOnFileStopsPawn(const Square kingSquare,
                                const Square pawnSquare,
                                const Color pawnColor)
{
   return file(kingSquare) == file(pawnSquare) &&
      testSquare(pawnOpponents[pawnColor][pawnSquare], kingSquare);
}

int getKnnkpChances(const Position * position, const Color color)
{
   Bitboard oppPawns = position->piecesOfType[PAWN | opponent(color)];

   return ((oppPawns & troitzkyArea[color]) == EMPTY_BITBOARD ? 0 : 8);
}

int getKppxKxChances(const Position * position, const Color color)
{
   Bitboard ownPawns = position->piecesOfType[PAWN | color];
   File file;
   int fileCount = 0;

   for (file = FILE_A; file <= FILE_H; file++)
   {
      if ((squaresOfFile[file] & ownPawns) != EMPTY_BITBOARD)
      {
         if (++fileCount > 1)
         {
            return 16;
         }
      }
   }

   if (fileCount == 1)
   {
      const Square oppKing = position->king[opponent(color)];
      Square square;

      ITERATE_BITBOARD(&ownPawns, square)
      {
         if (passiveKingOnFileStopsPawn(oppKing, square, color) == FALSE)
         {
            return (distance(oppKing, square) <= 2 ? 14 : 16);
         }
      }

      return (position->piecesOfType[KNIGHT | color] !=
              EMPTY_BITBOARD ? 8 : 4);
   }

   return 16;
}

int getKpxKpxChances(const Position * position, const EvaluationBase * base,
                     const Color color)
{
   if (base->passedPawns[WHITE] != EMPTY_BITBOARD ||
       base->passedPawns[BLACK] != EMPTY_BITBOARD)
   {
      return 16;
   }
   else
   {
      const int chancesByPawnWidth[8] = { 0, 5, 9, 13, 15, 16, 16, 16 };
      const int width = getWidth(position->piecesOfType[PAWN | color]);
      const int chances = chancesByPawnWidth[width];

      if (chances < 16)
      {
         const Color oppColor = opponent(color);
         const int kingDistance =
            getMinimumDistance(position->piecesOfType[PAWN | oppColor],
                               position->king[oppColor]);

         return min(16, chances + 2 * (kingDistance - 1));
      }

      return chances;
   }
}

int getKrppkrChances(const Position * position, const Color color)
{
   const Color oppColor = opponent(color);
   Bitboard pawns = position->piecesOfType[(Piece) (PAWN | color)];
   const Square oppKing = position->king[oppColor];
   Square square;
   Bitboard files = EMPTY_BITBOARD;

   assert(getNumberOfSetSquares(pawns) == 2);

   if (colorRank(color, oppKing) == RANK_8)
   {
      return 16;
   }

   ITERATE_BITBOARD(&pawns, square)
   {
      const File pawnFile = file(square);

      files |= minValue[getSquare(pawnFile, RANK_1)];

      if (passiveKingStopsPawn(oppKing, square, color) == FALSE)
      {
         const int fileDiff = ((file(oppKing) > pawnFile) ? (file(oppKing) - pawnFile) : (pawnFile - file(oppKing)));
         const int rankDiff =
            colorRank(color, square) - colorRank(color, oppKing);

         if ((fileDiff > 2) || (fileDiff == 2 && rankDiff >= 0))
         {
            return 16;
         }
      }
   }

   if (files == A1C1 || files == F1H1 || getNumberOfSetSquares(files) == 1)
   {
      return 4;
   }

   return 16;
}

int getKrpkrChances(const Position * position, const Color color)
{
   const Color oppColor = opponent(color);
   Bitboard pawns = position->piecesOfType[(Piece) (PAWN | color)];
   const Square oppKing = position->king[oppColor];
   Square square = getLastSquare(&pawns);

   assert(getNumberOfSetSquares(pawns) == 0);

   if (passiveKingStopsPawn(oppKing, square, color))
   {
      if (testSquare(krprkDrawFiles, square))
      {
         return 2;
      }
      else
      {
         const Square king = position->king[color];
         const Rank kingRank = colorRank(color, king);
         const Rank pawnRank = colorRank(color, square);

         if (kingRank < RANK_6 || pawnRank < RANK_6)
         {
            return 2;
         }
      }
   }

   return 16;
}

int getKqppkqChances(const Position * position, const Color color)
{
   const Color oppColor = opponent(color);
   Bitboard pawns = position->piecesOfType[(Piece) (PAWN | color)];
   const Square oppKing = position->king[oppColor];
   Square square;
   Bitboard files = EMPTY_BITBOARD;

   assert(getNumberOfSetSquares(pawns) == 2);

   ITERATE_BITBOARD(&pawns, square)
   {
      if (passiveKingStopsPawn(oppKing, square, color) == FALSE)
      {
         return 16;
      }

      files |= minValue[getSquare(file(square), RANK_1)];
   }

   if (files == A1B1 || files == G1H1 || getNumberOfSetSquares(files) == 1)
   {
      return 4;
   }

   return 16;
}

int getKqpkqChances(const Position * position, const Color color)
{
   const Color oppColor = opponent(color);
   Bitboard pawns = position->piecesOfType[(Piece) (PAWN | color)];
   const Square oppKing = position->king[oppColor];
   Square square = getLastSquare(&pawns);
   const File pawnFile = file(square);
   const Rank pawnRank = colorRank(color, square);
   const int distDiff = distance(oppKing, square) -
      distance(position->king[color], square);
   const int distDiv = (distDiff <= 0 ? 2 : 1);

   assert(getNumberOfSetSquares(pawns) == 0);

   if (pawnRank <= RANK_6 && (pawnFile <= FILE_B || pawnFile >= FILE_G))
   {
      return (passiveKingStopsPawn(oppKing, square, color) ? 1 : 4 / distDiv);
   }
   else
   {
      return (passiveKingStopsPawn(oppKing, square, color) ?
              4 : 16 / distDiv);
   }
}

int getKpkChances(const Position * position, const Color color)
{
   const Color oppColor = opponent(color);
   Bitboard pawns = position->piecesOfType[(Piece) (PAWN | color)];

   if ((pawns & nonA & nonH) != EMPTY_BITBOARD)
   {
      return 16;
   }
   else
   {
      const Square oppKing = position->king[oppColor];
      Square square;

      ITERATE_BITBOARD(&pawns, square)
      {
         if (passiveKingStopsPawn(oppKing, square, color) == FALSE)
         {
            return 16;
         }
      }

      return 0;                 /* king holds pawn(s) */
   }
}

int getKbpkChances(const Position * position, const Color color)
{
   const Color oppColor = opponent(color);
   const bool oppColors = oppositeColoredBishops(position);
   const int max = (oppColors ? 8 : 16);
   const Bitboard promotablePawns = getPromotablePawns(position, color);
   const int numPromotablePawns = getNumberOfSetSquares(promotablePawns);
   const int numDefenders =
      (oppColors ?
       getNumberOfSetSquares(position->piecesOfType[BISHOP | oppColor]) : 0);

   return (numPromotablePawns > numDefenders ? max : 0);
}

int specialPositionChances(const Position * position,
                           const EvaluationBase * base,
                           const SpecialEvalType type, const Color color)
{
   switch (type)
   {
   case Se_KpK:
      return getKpkChances(position, color);

   case Se_KbpK:
      return getKbpkChances(position, color);

   case Se_KrpKr:
      return getKrpkrChances(position, color);

   case Se_KrppKr:
      return getKrppkrChances(position, color);

   case Se_KqpKq:
      return getKqpkqChances(position, color);

   case Se_KqppKq:
      return getKqppkqChances(position, color);

   case Se_KnnKp:
      return getKnnkpChances(position, color);

   case Se_KppxKx:
      return getKppxKxChances(position, color);

   case Se_KpxKpx:
      return getKpxKpxChances(position, base, color);

   default:
      return 16;
   }
}

int getChances(const Position * position,
               const EvaluationBase * base, const Color winningColor)
{
   const MaterialInfo *mi = base->materialInfo;
   int chances = 16;

   if (numberOfNonPawnPieces(position, winningColor) <= 4 &&
       getPieceCount(position, WHITE_QUEEN) <= 1 &&
       getPieceCount(position, BLACK_QUEEN) <= 1)
   {
      if (winningColor == WHITE)
      {
         if (mi->specialEvalWhite != Se_None)
         {
            const int specialChances =
               specialPositionChances(position, base, mi->specialEvalWhite,
                                      WHITE);

            chances = min(specialChances, mi->chancesWhite);
         }
         else
         {
            chances = mi->chancesWhite;
         }
      }
      else
      {
         if (mi->specialEvalBlack != Se_None)
         {
            const int specialChances =
               specialPositionChances(position, base, mi->specialEvalBlack,
                                      BLACK);

            chances = min(specialChances, mi->chancesBlack);
         }
         else
         {
            chances = mi->chancesBlack;
         }
      }
   }

   return chances;
}

bool hasBishopPair(const Position * position, const Color color)
{
   const Bitboard *bishops =
      &position->piecesOfType[(Piece) (BISHOP | color)];

   return (bool) ((lightSquares & *bishops) != EMPTY_BITBOARD &&
                  (darkSquares & *bishops) != EMPTY_BITBOARD);
}

int phaseValue(const INT32 value, const Position * position,
               EvaluationBase * base)
{
   const INT32 materialValue = base->materialBalance +
      base->materialInfo->materialBalance;
   const int materialOpeningValue = getOpeningValue(materialValue);
   const int materialEndgameValue = getEndgameValue(materialValue);
   const int chancesWhite = getChances(position, base, WHITE);
   const int chancesBlack = getChances(position, base, BLACK);
   const int chances = max(chancesWhite, chancesBlack);
   const int openingValue = materialOpeningValue + getOpeningValue(value);
   const int endgameValue = (materialEndgameValue * chances) / 16 +
      (getEndgameValue(value) * (16 + chances)) / 32;

   return (openingValue * (256 - base->materialInfo->phaseIndex) +
           endgameValue * base->materialInfo->phaseIndex) / 256;
}

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
   const int rookSaldo = pawnCountWhite * numWhiteRooks -
      pawnCountBlack * numBlackRooks;
   const int pieceCountSaldo =
      (numberOfNonPawnPieces(position, WHITE) - numWhiteRooks -
       getPieceCount(position, WHITE_QUEEN)) -
      (numberOfNonPawnPieces(position, BLACK) - numBlackRooks -
       getPieceCount(position, BLACK_QUEEN));

   if (hasBishopPair(position, WHITE))
   {
      balance += bishopPairBonus;
   }

   if (hasBishopPair(position, BLACK))
   {
      balance -= bishopPairBonus;
   }

   balance += knightSaldo * knightBonus - rookSaldo * rookMalus;

   if (numWhiteRooks >= 2)
   {
      balance -= rookPairMalus + rookQueenMalus;
   }
   else if (numWhiteRooks + getPieceCount(position, WHITE_QUEEN) >= 2)
   {
      balance -= rookQueenMalus;
   }

   if (numBlackRooks >= 2)
   {
      balance += rookPairMalus + rookQueenMalus;
   }
   else if (numBlackRooks + getPieceCount(position, BLACK_QUEEN) >= 2)
   {
      balance += rookQueenMalus;
   }

   if (pieceCountSaldo > 0)
   {
      balance += pieceUpBonus;
   }
   else if (pieceCountSaldo < 0)
   {
      balance -= pieceUpBonus;
   }

   return balance;
}

/**
 * Calculate a rough value of the specified position,
 * based on the current pst-values and the specified evaluation base.
 *
 * @return the value of the specified position
 */
INT32 positionalBalance(const Position * position, EvaluationBase * base)
{
   static const INT32 tempoBonus[2] = {
      V(VALUE_TEMPO_OPENING, VALUE_TEMPO_ENDGAME),
      V(-VALUE_TEMPO_OPENING, -VALUE_TEMPO_ENDGAME)
   };
   const INT32 balance = position->balance + base->balance +
      tempoBonus[position->activeColor];
   const int value = phaseValue(balance, position, base);

   return (position->activeColor == WHITE ? value : -value);
}

/**
 * Check if the specified color can win the specified position.
 *
 * @return FALSE if the specified color doesn't have sufficient material
 *         left to win the position
 */
bool hasWinningPotential(Position * position, Color color)
{
   return (bool) (position->numberOfPieces[color] > 1);
}

/**
 * Get the king safety hash value for the given king square.
 */
Bitboard calculateKingPawnSafetyHashKey(const Position * position,
                                        const Color color)
{
   const int mask[2] =
      { WHITE_00 | WHITE_000 | 16, BLACK_00 | BLACK_000 | 32 };
   const int index = (position->castlingRights | 48) & mask[color];

   return position->pawnHashKey ^
      GENERATED_KEYTABLE[color][position->king[color]] ^
      GENERATED_KEYTABLE[2][index];
}

int getPawnWidth(const Position * position, const Color color)
{
   const Bitboard tmp = position->piecesOfType[(Piece) (PAWN | color)] |
      minValue[position->king[opponent(color)]];

   return getWidth(tmp);
}

int getPassedPawnWidth(const Position * position,
                       const EvaluationBase * base, const Color color)
{
   const Bitboard tmp = base->passedPawns[color] |
      minValue[position->king[opponent(color)]];

   return getWidth(tmp);
}

int quad(int y_min, int y_max, int rank)
{
   const int bonusPerRank[8] = { 0, 0, 0, 26, 77, 154, 256, 0 };

   return y_min + ((y_max - y_min) * bonusPerRank[rank] + 128) / 256;
}

bool squareIsPawnSafe(const EvaluationBase * base,
                      const Color color, const Square square)
{
   return testSquare(base->pawnAttackableSquares[opponent(color)],
                     square) == FALSE;
}

bool hasAttackingBishop(const Position * position,
                        const Color attackingColor, const Square square)
{
   const Bitboard attackers =
      ((lightSquares & minValue[square]) != EMPTY_BITBOARD ?
       lightSquares : darkSquares);

   return (bool)
      ((attackers & position->piecesOfType[BISHOP | attackingColor]) !=
       EMPTY_BITBOARD);
}

void getPawnInfo(const Position * position, EvaluationBase * base)
{
   const Bitboard white = position->piecesOfType[WHITE_PAWN];
   const Bitboard black = position->piecesOfType[BLACK_PAWN];
   Bitboard whiteLateralSquares, blackLateralSquares;
   Bitboard whiteSwamp, blackSwamp;
   Bitboard pawnAttackableSquaresWhite, pawnAttackableSquaresBlack;
   register Bitboard tmp1, tmp2;

   /* Calculate upward and downward realms */
   tmp1 = (white << 8) | (white << 16) | (white << 24);
   tmp1 |= (tmp1 << 24);
   tmp2 = (white >> 8) | (white >> 16) | (white >> 24);
   tmp2 |= (tmp2 >> 24);

   base->doubledPawns[WHITE] = white & tmp2;
   base->upwardRealm[WHITE] = (tmp1 = tmp1 | white);
   pawnAttackableSquaresWhite = ((tmp1 & nonA) << 7) | ((tmp1 & nonH) << 9);
   base->downwardRealm[WHITE] = tmp2;

   /* Calculate upward and downward realms */
   tmp1 = (black >> 8) | (black >> 16) | (black >> 24);
   tmp1 |= (tmp1 >> 24);
   tmp2 = (black << 8) | (black << 16) | (black << 24);
   tmp2 |= (tmp2 << 24);

   base->doubledPawns[BLACK] = black & tmp2;
   base->upwardRealm[BLACK] = (tmp1 = tmp1 | black);
   pawnAttackableSquaresBlack = ((tmp1 & nonA) >> 9) | ((tmp1 & nonH) >> 7);
   base->downwardRealm[BLACK] = tmp2;

   /* Calculate the squares protected by a pawn */
   whiteLateralSquares = ((white & nonA) >> 1) | ((white & nonH) << 1);
   base->pawnProtectedSquares[WHITE] = whiteLateralSquares << 8;
   blackLateralSquares = ((black & nonA) >> 1) | ((black & nonH) << 1);
   base->pawnProtectedSquares[BLACK] = blackLateralSquares >> 8;

   /* Identify the passed pawns */
   whiteSwamp = base->downwardRealm[BLACK] | base->upwardRealm[WHITE] |
      pawnAttackableSquaresWhite;
   blackSwamp = base->downwardRealm[WHITE] | base->upwardRealm[BLACK] |
      pawnAttackableSquaresBlack;

   base->passedPawns[WHITE] = white & ~blackSwamp;
   base->passedPawns[BLACK] = black & ~whiteSwamp;

   /* Calculate the weak pawns */
   tmp2 = ~(white | black | base->pawnProtectedSquares[BLACK]);
   tmp1 = (whiteLateralSquares & tmp2) >> 8;
   tmp1 |= (tmp1 & squaresOfRank[RANK_3] & tmp2) >> 8;
   base->weakPawns[WHITE] =
      (white & ~(pawnAttackableSquaresWhite | whiteLateralSquares | tmp1));

   tmp2 = ~(white | black | base->pawnProtectedSquares[WHITE]);
   tmp1 = (blackLateralSquares & tmp2) << 8;
   tmp1 |= (tmp1 & squaresOfRank[RANK_6] & tmp2) << 8;
   base->weakPawns[BLACK] =
      (black & ~(pawnAttackableSquaresBlack | blackLateralSquares | tmp1));

   /* Calculate the candidates */
   base->candidatePawns[WHITE] = white & ~base->passedPawns[WHITE] &
      (pawnAttackableSquaresWhite | whiteLateralSquares) &
      ~(base->upwardRealm[BLACK] | base->downwardRealm[WHITE]);

   base->candidatePawns[BLACK] = black & ~base->passedPawns[BLACK] &
      (pawnAttackableSquaresBlack | blackLateralSquares) &
      ~(base->upwardRealm[WHITE] | base->downwardRealm[BLACK]);

#ifdef BONUS_HIDDEN_PASSER
   /* Calculate the hidden candidates */
   base->hiddenCandidatePawns[WHITE] = white & (black >> 8) &
      ~pawnAttackableSquaresBlack & ~(blackLateralSquares) &
      (squaresOfRank[RANK_5] | squaresOfRank[RANK_6]) &
      base->pawnProtectedSquares[WHITE];

   base->hiddenCandidatePawns[BLACK] = black & (white << 8) &
      ~pawnAttackableSquaresWhite & ~(whiteLateralSquares) &
      (squaresOfRank[RANK_4] | squaresOfRank[RANK_3]) &
      base->pawnProtectedSquares[BLACK];
#endif

   tmp1 = black & base->pawnProtectedSquares[BLACK];
   tmp2 = ((tmp1 & nonA) >> 9) & ((tmp1 & nonH) >> 7);
   tmp2 &= ~pawnAttackableSquaresWhite;
   tmp1 = tmp2 | (tmp2 >> 8);
   base->fixedPawns[WHITE] = tmp1 | (tmp1 >> 16) | (tmp1 >> 32);

   tmp1 = white & base->pawnProtectedSquares[WHITE];
   tmp2 = ((tmp1 & nonA) << 7) & ((tmp1 & nonH) << 9);
   tmp2 &= ~pawnAttackableSquaresBlack;
   tmp1 = tmp2 | (tmp2 << 8);
   base->fixedPawns[BLACK] = tmp1 | (tmp1 << 16) | (tmp1 << 32);

#ifdef BONUS_HIDDEN_PASSER
   base->hasPassersOrCandidates[WHITE] = (bool)
      (base->passedPawns[WHITE] != EMPTY_BITBOARD ||
       base->candidatePawns[WHITE] != EMPTY_BITBOARD ||
       base->hiddenCandidatePawns[WHITE] != EMPTY_BITBOARD);

   base->hasPassersOrCandidates[BLACK] = (bool)
      (base->passedPawns[BLACK] != EMPTY_BITBOARD ||
       base->candidatePawns[BLACK] != EMPTY_BITBOARD ||
       base->hiddenCandidatePawns[BLACK] != EMPTY_BITBOARD);
#endif

   tmp1 = (white << 8) & ~(white | black);
   tmp2 = white | (tmp1 & ~base->pawnProtectedSquares[BLACK]);
   tmp2 |= tmp1 & base->pawnProtectedSquares[WHITE];
   tmp1 &= squaresOfRank[RANK_3] & ~base->pawnProtectedSquares[BLACK];
   tmp1 = (tmp1 << 8) & ~(white | black);
   tmp2 |= tmp1 & ~base->pawnProtectedSquares[BLACK];
   tmp2 |= tmp1 & base->pawnProtectedSquares[WHITE];
   base->pawnAttackableSquares[WHITE] =
      ((tmp2 & nonA) << 7) | ((tmp2 & nonH) << 9);

   /*dumpBitboard(base->pawnAttackableSquares[WHITE], "pawnAttackable white");
      dumpPosition(position); */

   tmp1 = (black >> 8) & ~(white | black);
   tmp2 = black | (tmp1 & ~base->pawnProtectedSquares[WHITE]);
   tmp2 |= tmp1 & base->pawnProtectedSquares[BLACK];
   tmp1 &= squaresOfRank[RANK_6] & ~base->pawnProtectedSquares[WHITE];
   tmp1 = (tmp1 >> 8) & ~(white | black);
   tmp2 |= tmp1 & ~base->pawnProtectedSquares[WHITE];
   tmp2 |= tmp1 & base->pawnProtectedSquares[BLACK];
   base->pawnAttackableSquares[BLACK] =
      ((tmp2 & nonA) >> 9) | ((tmp2 & nonH) >> 7);

   base->chainPawns[WHITE] = white &
      (base->pawnProtectedSquares[WHITE] | whiteLateralSquares);
   base->chainPawns[BLACK] = black &
      (base->pawnProtectedSquares[BLACK] | blackLateralSquares);
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

bool captureCreatesPasser(Position * position, const Square captureSquare,
                          const Piece capturingPiece)
{
   const Piece captured = position->piece[captureSquare];
   const Color capturedColor = pieceColor(captured);
   const Color passerColor = opponent(capturedColor);
   Bitboard candidates = (passedPawnCorridor[capturedColor][captureSquare] |
                          candidateDefenders[capturedColor][captureSquare]) &
      position->piecesOfType[PAWN | passerColor];
   bool result = FALSE;

   if (pieceType(capturingPiece) == PAWN &&
       pawnIsPassed(position, captureSquare, passerColor))
   {
      /* dumpSquare(captureSquare);
         dumpPosition(position); */

      return TRUE;
   }

   if (candidates != EMPTY_BITBOARD)
   {
      const Bitboard pawnsOriginal =
         position->piecesOfType[PAWN | capturedColor];
      Square square;

      clearSquare(position->piecesOfType[PAWN | capturedColor],
                  captureSquare);

      ITERATE_BITBOARD(&candidates, square)
      {
         if (pawnIsPassed(position, square, passerColor))
         {
            /*dumpSquare(captureSquare);
               dumpSquare(square);
               dumpBitboard(cc, "cd");
               dumpPosition(position); */

            result = TRUE;
            candidates = EMPTY_BITBOARD;        /* force loop exit */
         }
      }

      position->piecesOfType[PAWN | capturedColor] = pawnsOriginal;
   }

   return result;
}

bool passerWalks(const Position * position,
                 const Square passerSquare, const Color passerColor)
{
   const Square attackerKingSquare = position->king[passerColor];
   const Square defenderKingSquare = position->king[opponent(passerColor)];
   const int attackerDistance = distance(attackerKingSquare, passerSquare);
   const Rank kingRank = colorRank(passerColor, attackerKingSquare);
   const File passerFile = file(passerSquare);

   if (passerFile >= FILE_B && passerFile <= FILE_G)
   {
      if ((kingRank == RANK_6 || kingRank == RANK_7) &&
          kingRank > colorRank(passerColor, passerSquare) &&
          ((file(attackerKingSquare) > passerFile) ? (file(attackerKingSquare) - passerFile) : (passerFile - file(attackerKingSquare))) <= 1 &&
          attackerDistance <= 2)
      {
         if (position->activeColor == passerColor ||
             attackerDistance == 1 ||
             distance(defenderKingSquare, passerSquare) > 1)
         {
            return TRUE;
         }
      }

      /*
         if (kingRank == colorRank(passerColor, passerSquare) + 2 &&
         ((file(attackerKingSquare) > passerFile) ? (file(attackerKingSquare) - passerFile) : (passerFile - file(attackerKingSquare))) <= 1 &&
         attackerDistance <= 2)
         {
         if (position->activeColor == passerColor ||
         attackerDistance == 1 ||
         distance(defenderKingSquare, passerSquare) > 1)
         {
         return TRUE;
         }
         }
       */
   }
   else if ((kingRank == RANK_7 || kingRank == RANK_8) &&
            ((file(attackerKingSquare) > passerFile) ? (file(attackerKingSquare) - passerFile) : (passerFile - file(attackerKingSquare))) == 1 &&
            attackerDistance <= 2)
   {
      if (position->activeColor == passerColor ||
          attackerDistance == 1 ||
          distance(defenderKingSquare, passerSquare) > 1)
      {
         return TRUE;
      }
   }

   return FALSE;
}

#ifdef USE_ORTHO_BATTERY_PIECE
static Piece getOrthoBatteryPiece(const Position * position,
                                  const Bitboard moves,
                                  const Square attackerSquare,
                                  const Color kingColor)
{
   const Square kingSquare = position->king[kingColor];
   Bitboard middlePiece = EMPTY_BITBOARD;

   if (testSquare(generalMoves[ROOK][kingSquare], attackerSquare))
   {
      middlePiece = generalMoves[ROOK][attackerSquare] &
         moves & getMagicRookMoves(kingSquare, position->allPieces);

      if (middlePiece != EMPTY_BITBOARD)
      {
         const Square pieceSquare = getLastSquare(&middlePiece);

         return position->piece[pieceSquare];
      }
   }

   return NO_PIECE;
}
#endif

Piece getDiaBatteryPiece(const Position * position,
                         const Bitboard moves,
                         const Square attackerSquare, const Color kingColor)
{
   const Square kingSquare = position->king[kingColor];
   Bitboard middlePiece = EMPTY_BITBOARD;

   if (testSquare(generalMoves[BISHOP][kingSquare], attackerSquare))
   {
      middlePiece = generalMoves[BISHOP][attackerSquare] &
         moves & getMagicBishopMoves(kingSquare, position->allPieces);

      if (middlePiece != EMPTY_BITBOARD)
      {
         const Square pieceSquare = getLastSquare(&middlePiece);

         return position->piece[pieceSquare];
      }
   }

   return NO_PIECE;
}

Square getPinningPiece(const Position * position,
                       const EvaluationBase * base,
                       const Square pieceSquare, const Color pieceColor)
{
   if (testSquare(base->pinnedCandidatesDia[pieceColor], pieceSquare))
   {
      const Color oppColor = opponent(pieceColor);
      Bitboard pinningCandidates = position->piecesOfType[BISHOP | oppColor] |
         position->piecesOfType[QUEEN | oppColor];

      if (pinningCandidates != EMPTY_BITBOARD)
      {
         pinningCandidates &=
            generalMoves[BISHOP][position->king[pieceColor]];

         if (pinningCandidates != EMPTY_BITBOARD)
         {
            pinningCandidates &=
               getMagicBishopMoves(pieceSquare, position->allPieces);

            if (pinningCandidates != EMPTY_BITBOARD)
            {
               return getFirstSquare(&pinningCandidates);
            }
         }
      }
   }

   if (testSquare(base->pinnedCandidatesOrtho[pieceColor], pieceSquare))
   {
      const Color oppColor = opponent(pieceColor);
      Bitboard pinningCandidates = position->piecesOfType[ROOK | oppColor] |
         position->piecesOfType[QUEEN | oppColor];

      if (pinningCandidates != EMPTY_BITBOARD)
      {
         pinningCandidates &= generalMoves[ROOK][position->king[pieceColor]];

         if (pinningCandidates != EMPTY_BITBOARD)
         {
            pinningCandidates &=
               getMagicRookMoves(pieceSquare, position->allPieces);

            if (pinningCandidates != EMPTY_BITBOARD)
            {
               return getFirstSquare(&pinningCandidates);
            }
         }
      }
   }

   return NO_SQUARE;
}

#ifdef USE_CAN_CASTLE
static bool canCastle(const Position * position, const Color color)
{
   BYTE castlingRights[2] = { WHITE_00 | WHITE_000, BLACK_00 | BLACK_000 };

   return (castlingRights[color] & position->castlingRights) != 0;
}
#endif

#define PERSPECTIVE_WHITE
#include "evaluationc.c"
#undef PERSPECTIVE_WHITE
#include "evaluationc.c"

int getValue(const Position * position,
             EvaluationBase * base,
             PawnHashInfo * pawnHashtable,
             KingSafetyHashInfo * kingsafetyHashtable,
             Accumulator * acc)
{
   return evaluateNnue((Position *)position, acc);
}

static void transposeMatrix(const int human[], int machine[])
{
   int file, rank, i = 0;

   for (rank = RANK_8; rank >= RANK_1; rank--)
   {
      for (file = FILE_A; file <= FILE_H; file++)
      {
         const Square machineSquare = getSquare(file, rank);

         machine[machineSquare] = human[i++];
      }
   }
}

static int pstOpeningValue(INT32 value, const int weight)
{
   return applyWeight(weight, getOpeningValue(value));
}

static int pstEndgameValue(INT32 value, const int weight)
{
   return applyWeight(weight, getEndgameValue(value));
}

static void initializePieceSquareValues(void)
{
   Square sq;

   mvImpact[WHITE_QUEEN] = V(VALUE_QUEEN_OPENING, VALUE_QUEEN_ENDGAME);
   mvImpact[BLACK_QUEEN] = V(-VALUE_QUEEN_OPENING, -VALUE_QUEEN_ENDGAME);
   mvImpact[WHITE_ROOK] = V(VALUE_ROOK_OPENING, VALUE_ROOK_ENDGAME);
   mvImpact[BLACK_ROOK] = V(-VALUE_ROOK_OPENING, -VALUE_ROOK_ENDGAME);
   mvImpact[WHITE_BISHOP] = V(VALUE_BISHOP_OPENING, VALUE_BISHOP_ENDGAME);
   mvImpact[BLACK_BISHOP] = V(-VALUE_BISHOP_OPENING, -VALUE_BISHOP_ENDGAME);
   mvImpact[WHITE_KNIGHT] = V(VALUE_KNIGHT_OPENING, VALUE_KNIGHT_ENDGAME);
   mvImpact[BLACK_KNIGHT] = V(-VALUE_KNIGHT_OPENING, -VALUE_KNIGHT_ENDGAME);
   mvImpact[WHITE_PAWN] = V(VALUE_PAWN_OPENING, VALUE_PAWN_ENDGAME);
   mvImpact[BLACK_PAWN] = V(-VALUE_PAWN_OPENING, -VALUE_PAWN_ENDGAME);

   ITERATE(sq)
   {
      int ov, ev;

      ov = pstOpeningValue(PstPawn[sq], 256);
      ev = pstEndgameValue(PstPawn[sq], 250);
      pieceSquareBonus[WHITE_PAWN][sq] =
         pieceSquareBonus[BLACK_PAWN][getFlippedSquare(sq)] = V(ov, ev);

      ov = pstOpeningValue(PstKnight[sq], 256);
      ev = pstEndgameValue(PstKnight[sq], 256);
      pieceSquareBonus[WHITE_KNIGHT][sq] =
         pieceSquareBonus[BLACK_KNIGHT][getFlippedSquare(sq)] = V(ov, ev);

      ov = pstOpeningValue(PstBishop[sq], 256);
      ev = pstEndgameValue(PstBishop[sq], 256);
      pieceSquareBonus[WHITE_BISHOP][sq] =
         pieceSquareBonus[BLACK_BISHOP][getFlippedSquare(sq)] = V(ov, ev);

      ov = pstOpeningValue(PstRook[sq], 256);
      ev = pstEndgameValue(PstRook[sq], 256);
      pieceSquareBonus[WHITE_ROOK][sq] =
         pieceSquareBonus[BLACK_ROOK][getFlippedSquare(sq)] = V(ov, ev);

      ov = pstOpeningValue(PstQueen[sq], 256);
      ev = pstEndgameValue(PstQueen[sq], 322);
      pieceSquareBonus[WHITE_QUEEN][sq] =
         pieceSquareBonus[BLACK_QUEEN][getFlippedSquare(sq)] = V(ov, ev);

      ov = pstOpeningValue(PstKing[sq], 247);
      ev = pstEndgameValue(PstKing[sq], 244);
      pieceSquareBonus[WHITE_KING][sq] =
         pieceSquareBonus[BLACK_KING][getFlippedSquare(sq)] = V(ov, ev);

#ifdef LOG_ARRAY
      logDebug("V(%d,%d),", ov, ev);

      if ((sq + 1) % 8 == 0)
      {
         logDebug(" /* rank %d */\n", (sq + 1) / 8);
      }
#endif
   }

#ifdef LOG_ARRAY
   getKeyStroke();
#endif
}

static void initializeKingAttacks(void)
{
   Square square;

   ITERATE(square)
   {
      const Bitboard corona = getKingMoves(square);
      KingAttacks *attackInfo = &kingAttacks[square];
      Square attackerSquare;

      attackInfo->diaAttackers = attackInfo->orthoAttackers =
         attackInfo->knightAttackers = attackInfo->pawnAttackers[WHITE] =
         attackInfo->pawnAttackers[BLACK] = EMPTY_BITBOARD;

      ITERATE(attackerSquare)
      {
         attackInfo->attackedByDia[attackerSquare] =
            attackInfo->attackedByOrtho[attackerSquare] = NO_SQUARE;
      }

      ITERATE(attackerSquare)
      {
         Bitboard dia, ortho;
         const Bitboard knight =
            corona & generalMoves[WHITE_KNIGHT][attackerSquare];
         const Bitboard whitePawn =
            corona & generalMoves[WHITE_PAWN][attackerSquare];
         const Bitboard blackPawn =
            corona & generalMoves[BLACK_PAWN][attackerSquare];

         dia = corona & generalMoves[WHITE_BISHOP][attackerSquare];
         ortho = corona & generalMoves[WHITE_ROOK][attackerSquare];

         if (dia != EMPTY_BITBOARD)
         {
            Square attackedSquare;
            int dist = 8;

            setSquare(attackInfo->diaAttackers, attackerSquare);

            ITERATE_BITBOARD(&dia, attackedSquare)
            {
               const int currentDistance =
                  distance(attackerSquare, attackedSquare);

               if (currentDistance < dist)
               {
                  attackInfo->attackedByDia[attackerSquare] = attackedSquare;
                  dist = currentDistance;
               }
            }
         }

         if (ortho != EMPTY_BITBOARD)
         {
            Square attackedSquare;
            int dist = 8;

            setSquare(attackInfo->orthoAttackers, attackerSquare);

            ITERATE_BITBOARD(&ortho, attackedSquare)
            {
               const int currentDistance =
                  distance(attackerSquare, attackedSquare);

               if (currentDistance < dist)
               {
                  attackInfo->attackedByOrtho[attackerSquare] =
                     attackedSquare;
                  dist = currentDistance;
               }
            }
         }

         if (knight != EMPTY_BITBOARD)
         {
            setSquare(attackInfo->knightAttackers, attackerSquare);
         }

         if (whitePawn != EMPTY_BITBOARD)
         {
            setSquare(attackInfo->pawnAttackers[WHITE], attackerSquare);
         }

         if (blackPawn != EMPTY_BITBOARD)
         {
            setSquare(attackInfo->pawnAttackers[BLACK], attackerSquare);
         }
      }
   }
}

#define AVOID_TRADES_WITH_TWO_KNIGHTS   /* */

static void getPieceTradeSignatures(UINT32 * materialSignatureWhite,
                                    UINT32 * materialSignatureBlack)
{
   int numWhiteQueens, numWhiteRooks, numWhiteLightSquareBishops;
   int numWhiteDarkSquareBishops, numWhiteKnights, numWhitePawns;
   int numBlackQueens, numBlackRooks, numBlackLightSquareBishops;
   int numBlackDarkSquareBishops, numBlackKnights, numBlackPawns;
   bool finished = TRUE;
   bool whiteKnightTradesOnly = FALSE, blackKnightTradesOnly = FALSE;
   bool knightTradesOnly = FALSE;
   const UINT32 signature =
      bilateralSignature(*materialSignatureWhite, *materialSignatureBlack);

#ifdef AVOID_TRADES_WITH_TWO_KNIGHTS
   int numWhiteNonPawns, numBlackNonPawns;
#endif

   getPieceCounters(signature, &numWhiteQueens, &numWhiteRooks,
                    &numWhiteLightSquareBishops,
                    &numWhiteDarkSquareBishops, &numWhiteKnights,
                    &numWhitePawns, &numBlackQueens, &numBlackRooks,
                    &numBlackLightSquareBishops,
                    &numBlackDarkSquareBishops, &numBlackKnights,
                    &numBlackPawns);

#ifdef AVOID_TRADES_WITH_TWO_KNIGHTS
   numWhiteNonPawns =
      numWhiteQueens + numWhiteRooks + numWhiteLightSquareBishops +
      numWhiteDarkSquareBishops + numWhiteKnights;
   numBlackNonPawns =
      numBlackQueens + numBlackRooks + numBlackLightSquareBishops +
      numBlackDarkSquareBishops + numBlackKnights;

   if (numWhitePawns + numBlackPawns == 0 && numWhiteKnights == 2 &&
       numWhiteNonPawns == 3 && numWhiteNonPawns - numBlackNonPawns >= 2)
   {
      whiteKnightTradesOnly = TRUE;     /* white will avoid to trade pieces */
   }

   if (numBlackPawns + numWhitePawns == 0 && numBlackKnights == 2 &&
       numBlackNonPawns == 3 && numBlackNonPawns - numWhiteNonPawns >= 2)
   {
      blackKnightTradesOnly = TRUE;     /* black will avoid to trade pieces */
   }

   knightTradesOnly = whiteKnightTradesOnly || blackKnightTradesOnly;
#endif

   if (knightTradesOnly == FALSE && numWhiteQueens > 0 && numBlackQueens > 0)
   {
      numWhiteQueens--;
      numBlackQueens--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (knightTradesOnly == FALSE && numWhiteRooks > 0 && numBlackRooks > 0)
   {
      numWhiteRooks--;
      numBlackRooks--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (knightTradesOnly == FALSE &&
       numWhiteLightSquareBishops > 0 && numBlackLightSquareBishops > 0)
   {
      numWhiteLightSquareBishops--;
      numBlackLightSquareBishops--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (knightTradesOnly == FALSE &&
       numWhiteDarkSquareBishops > 0 && numBlackDarkSquareBishops > 0)
   {
      numWhiteDarkSquareBishops--;
      numBlackDarkSquareBishops--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (knightTradesOnly == FALSE &&
       numWhiteLightSquareBishops > 0 && numBlackDarkSquareBishops > 0)
   {
      numWhiteLightSquareBishops--;
      numBlackDarkSquareBishops--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (knightTradesOnly == FALSE &&
       numWhiteDarkSquareBishops > 0 && numBlackLightSquareBishops > 0)
   {
      numWhiteDarkSquareBishops--;
      numBlackLightSquareBishops--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (numWhiteKnights > 0 && numBlackKnights > 0)
   {
      numWhiteKnights--;
      numBlackKnights--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (whiteKnightTradesOnly == FALSE &&
       numWhiteLightSquareBishops > 0 && numBlackKnights > 0 &&
       numWhiteDarkSquareBishops == 0)
   {
      numWhiteLightSquareBishops--;
      numBlackKnights--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (whiteKnightTradesOnly == FALSE &&
       numWhiteDarkSquareBishops > 0 && numBlackKnights > 0 &&
       numWhiteLightSquareBishops == 0)
   {
      numWhiteDarkSquareBishops--;
      numBlackKnights--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (blackKnightTradesOnly == FALSE &&
       numWhiteKnights > 0 && numBlackLightSquareBishops > 0 &&
       numBlackDarkSquareBishops == 0)
   {
      numWhiteKnights--;
      numBlackLightSquareBishops--;
      finished = FALSE;
      goto calculateSignature;
   }

   if (blackKnightTradesOnly == FALSE &&
       numWhiteKnights > 0 && numBlackDarkSquareBishops > 0 &&
       numBlackLightSquareBishops == 0)
   {
      numWhiteKnights--;
      numBlackDarkSquareBishops--;
      finished = FALSE;
      goto calculateSignature;
   }

 calculateSignature:

   *materialSignatureWhite =
      getSingleMaterialSignature(numWhiteQueens, numWhiteRooks,
                                 numWhiteLightSquareBishops,
                                 numWhiteDarkSquareBishops, numWhiteKnights,
                                 numWhitePawns);
   *materialSignatureBlack =
      getSingleMaterialSignature(numBlackQueens, numBlackRooks,
                                 numBlackLightSquareBishops,
                                 numBlackDarkSquareBishops, numBlackKnights,
                                 numBlackPawns);

   if (finished == FALSE)
   {
      getPieceTradeSignatures(materialSignatureWhite, materialSignatureBlack);
   }
}

static bool hasMaterialForMate(const UINT32 materialSignature,
                               const UINT32 oppMaterialSignature,
                               SpecialEvalType * specialEval,
                               const bool tradePieces,
                               const bool evaluateOppMaterial)
{
   int numQueens, numRooks, numLightSquareBishops, numDarkSquareBishops;
   int numKnights, numPawns;
   int numOppQueens, numOppRooks, numOppLightSquareBishops;
   int numOppDarkSquareBishops, numOppKnights, numOppPawns;
   int numBishops, numOppBishops;
   int numPieces, numOppPieces;
   const UINT32 signature =
      bilateralSignature(materialSignature, oppMaterialSignature);

   if (tradePieces)
   {
      UINT32 ms = materialSignature, mso = oppMaterialSignature;
      SpecialEvalType dummy;

      getPieceTradeSignatures(&ms, &mso);

      return hasMaterialForMate(ms, mso, &dummy, FALSE, evaluateOppMaterial);
   }

   getPieceCounters(signature, &numQueens, &numRooks,
                    &numLightSquareBishops, &numDarkSquareBishops,
                    &numKnights, &numPawns, &numOppQueens, &numOppRooks,
                    &numOppLightSquareBishops, &numOppDarkSquareBishops,
                    &numOppKnights, &numOppPawns);

   numBishops = numLightSquareBishops + numDarkSquareBishops;
   numOppBishops = numOppLightSquareBishops + numOppDarkSquareBishops;
   numPieces = numQueens + numRooks + numBishops + numKnights;
   numOppPieces = numOppQueens + numOppRooks + numOppBishops + numOppKnights;

   if (evaluateOppMaterial && numPawns == 0)
   {
      if (numPieces == 1)
      {
         if (numRooks == 1 && numOppPieces > 0)
         {
            return FALSE;
         }

         if (numQueens == 1)
         {
            if (numOppQueens >= 1 || numOppKnights >= 2)
            {
               return FALSE;
            }

            if (numOppPieces >= 2 && numOppRooks >= 1)
            {
               return FALSE;
            }
         }
      }
      else if (numPieces == 2)
      {
         if (numBishops == 2 &&
             numOppQueens + numOppRooks + numOppBishops > 0)
         {
            return FALSE;
         }

         if (numBishops == 1 && numKnights == 1 && numOppPieces > 0)
         {
            return FALSE;
         }
      }
   }

   if (numQueens + numRooks + numLightSquareBishops + numDarkSquareBishops +
       numKnights == 0 && numPawns > 0)
   {
      *specialEval = Se_KpK;

      return TRUE;
   }

   if (numQueens + numRooks + numLightSquareBishops + numDarkSquareBishops +
       numPawns == 0 && numKnights == 2)
   {
      if (numOppQueens + numOppRooks + numOppLightSquareBishops +
          numOppDarkSquareBishops + numOppKnights == 0 && numOppPawns > 0)
      {
         *specialEval = Se_KnnKp;

         return TRUE;
      }
   }

   if (numPawns <= 3 && numPawns >= numOppPawns &&
       numOppPawns >= numPawns - 1 && numQueens == 0 &&
       numPieces == 1 && numPieces == numOppPieces &&
       (numBishops + numKnights == numOppBishops + numOppKnights ||
        numRooks == numOppRooks))
   {
      *specialEval = Se_KpxKpx;

      return TRUE;
   }

   if (numPawns >= 2 && numOppPawns == 0 &&
       numQueens == 0 && numOppQueens == 0 &&
       numPieces == 1 && numPieces == numOppPieces)
   {
      if (numBishops == 1 ||
          (numKnights == 1 && numOppBishops == 1) ||
          (numRooks == 1 && numRooks == numOppRooks))
      {
         *specialEval = Se_KppxKx;
      }

      return TRUE;
   }

   if (numQueens + numRooks + numKnights == 0 &&
       numLightSquareBishops + numDarkSquareBishops == 1 && numPawns > 0)
   {
      *specialEval = Se_KbpK;

      return TRUE;
   }

   if (numQueens + numBishops + numKnights == 0 && numRooks == 1 &&
       numPawns == 1 && numOppBishops >= 1)
   {
      *specialEval = Se_KrpKb;

      return TRUE;
   }

   if (numQueens + numBishops + numKnights == 0 && numRooks == 1 &&
       numPawns == 1 && numOppRooks >= 1)
   {
      *specialEval = Se_KrpKr;

      return TRUE;
   }

   if (numQueens + numBishops + numKnights == 0 && numRooks == 1 &&
       numPawns == 2 && numOppRooks >= 1)
   {
      *specialEval = Se_KrppKr;

      return TRUE;
   }

   if (numRooks + numBishops + numKnights == 0 && numQueens == 1 &&
       numPawns == 1 && numOppQueens >= 1)
   {
      *specialEval = Se_KqpKq;

      return TRUE;
   }

   if (numRooks + numBishops + numKnights == 0 && numQueens == 1 &&
       numPawns == 2 && numOppQueens >= 1)
   {
      *specialEval = Se_KqppKq;

      return TRUE;
   }

   if (numQueens + numRooks + numPawns > 0 || numKnights >= 3)
   {
      return TRUE;
   }

   if (numLightSquareBishops > 0 && numDarkSquareBishops > 0)
   {
      return TRUE;
   }

   if (numKnights > 0 && numLightSquareBishops + numDarkSquareBishops > 0)
   {
      return TRUE;
   }

   return FALSE;
}

static PieceType getKamikazePiece(const UINT32 ownMaterialSignature,
                                  const UINT32 oppMaterialSignature)
{
   int numQueens, numRooks, numLightSquareBishops;
   int numDarkSquareBishops, numKnights, numPawns;
   int numOppQueens, numOppRooks, numOppLightSquareBishops;
   int numOppDarkSquareBishops, numOppKnights, numOppPawns;
   int ownSignature;
   const UINT32 signature =
      bilateralSignature(ownMaterialSignature, oppMaterialSignature);

   getPieceCounters(signature, &numQueens, &numRooks,
                    &numLightSquareBishops, &numDarkSquareBishops,
                    &numKnights, &numPawns, &numOppQueens, &numOppRooks,
                    &numOppLightSquareBishops, &numOppDarkSquareBishops,
                    &numOppKnights, &numOppPawns);

   ownSignature =
      getSingleMaterialSignature(numQueens, numRooks,
                                 numLightSquareBishops,
                                 numDarkSquareBishops, numKnights,
                                 numPawns - 1);

   if (numOppRooks > 0)
   {
      const int oppSignature =
         getSingleMaterialSignature(numOppQueens, numOppRooks - 1,
                                    numOppLightSquareBishops,
                                    numOppDarkSquareBishops,
                                    numOppKnights, numOppPawns);

      if (hasMaterialForMate(ownSignature, oppSignature, 0, TRUE, FALSE) ==
          FALSE)
      {
         return ROOK;
      }
   }

   if (numOppLightSquareBishops > 0)
   {
      const int oppSignature =
         getSingleMaterialSignature(numOppQueens, numOppRooks,
                                    numOppLightSquareBishops - 1,
                                    numOppDarkSquareBishops,
                                    numOppKnights, numOppPawns);

      if (hasMaterialForMate(ownSignature, oppSignature, 0, TRUE, FALSE) ==
          FALSE)
      {
         return BISHOP;
      }
   }

   if (numOppDarkSquareBishops > 0)
   {
      const int oppSignature =
         getSingleMaterialSignature(numOppQueens, numOppRooks,
                                    numOppLightSquareBishops,
                                    numOppDarkSquareBishops - 1,
                                    numOppKnights, numOppPawns);

      if (hasMaterialForMate(ownSignature, oppSignature, 0, TRUE, FALSE) ==
          FALSE)
      {
         return BISHOP;
      }
   }

   if (numOppKnights > 0)
   {
      const int oppSignature =
         getSingleMaterialSignature(numOppQueens, numOppRooks,
                                    numOppLightSquareBishops,
                                    numOppDarkSquareBishops,
                                    numOppKnights - 1,
                                    numOppPawns);

      if (hasMaterialForMate(ownSignature, oppSignature, 0, TRUE, FALSE) ==
          FALSE)
      {
         return KNIGHT;
      }
   }

   return NO_PIECETYPE;
}

static UINT8 getWinningChances(const UINT32 materialSignature,
                               const UINT32 oppMaterialSignature)
{
   int numQueens, numRooks, numLightSquareBishops;
   int numDarkSquareBishops, numKnights, numPawns;
   int numOppQueens, numOppRooks, numOppLightSquareBishops;
   int numOppDarkSquareBishops, numOppKnights, numOppPawns;
   int numPieces;
   int numOppBishops, numOppMinors, numOppSliders;
   bool oppositeColoredBishops;
   const UINT32 signature =
      bilateralSignature(materialSignature, oppMaterialSignature);
   PieceType kamikazePiece = getKamikazePiece(materialSignature,
                                              oppMaterialSignature);

   getPieceCounters(signature, &numQueens, &numRooks,
                    &numLightSquareBishops, &numDarkSquareBishops,
                    &numKnights, &numPawns, &numOppQueens, &numOppRooks,
                    &numOppLightSquareBishops, &numOppDarkSquareBishops,
                    &numOppKnights, &numOppPawns);
   numPieces = numQueens + numRooks + numLightSquareBishops +
      numDarkSquareBishops + numKnights;
   numOppBishops = numOppLightSquareBishops + numOppDarkSquareBishops;
   numOppMinors = numOppBishops + numOppKnights;
   numOppSliders = numOppQueens + numOppRooks + numOppBishops;
   oppositeColoredBishops = (bool)
      (numLightSquareBishops + numDarkSquareBishops > 0 &&
       ((numOppLightSquareBishops > 0 && numLightSquareBishops == 0) ||
        (numOppDarkSquareBishops > 0 && numDarkSquareBishops == 0)));

   if (numPieces == 0)
   {
      if (numPawns <= 1 && numOppSliders > 0)
      {
         return 0;
      }

      if (numPawns <= 1 && numOppKnights > 0)
      {
         return 4;
      }

      if (numPawns == 2)
      {
         return (numOppSliders >= 2 ? 2 : 8);
      }
   }

   if (numPieces == 1)
   {
      if (oppositeColoredBishops)
      {
         const int pawnDiff = min(3, abs(numPawns - numOppPawns));

         return (UINT8) (numPawns > 1 ? 8 + 2 * pawnDiff : 0);
      }

      if (numPawns == 1)        /* One piece, one pawn: */
      {
         if (numQueens > 0 && numOppRooks >= 2)
         {
            return 1;
         }

         if (numQueens > 0 && numOppRooks + numOppMinors >= 2)
         {
            return 12;          /* usually won, but difficult */
         }

         if (kamikazePiece != NO_PIECETYPE)
         {
            switch (kamikazePiece)
            {
            case ROOK:
               return 1;
            case BISHOP:
               return 2;
            case KNIGHT:
               return 4;
            default:
               break;
            }
         }
      }
   }
   else if (numPieces == 2)     /* has more than one piece: */
   {
      if (numPawns <= 1)
      {
         if (numRooks == 2 && numOppQueens > 0)
         {
            return (numPawns == 0 ? 1 : 2);
         }

         if (kamikazePiece != NO_PIECETYPE)
         {
            switch (kamikazePiece)
            {
            case ROOK:
               return 1;
            case BISHOP:
               return 2;
            case KNIGHT:
               return 4;
            default:
               break;
            }
         }
      }

      if (numQueens == 0 && numRooks <= 1 && numRooks == numOppRooks &&
          oppositeColoredBishops)
      {
         const int pawnDiff = min(3, abs(numPawns - numOppPawns));

         return (UINT8) (12 + pawnDiff);
      }
   }

   return 16;
}

static UINT8 getWinningChancesWithoutPawn(UINT32 materialSignature,
                                          UINT32 oppMaterialSignature)
{
   int numQueens, numRooks, numLightSquareBishops;
   int numDarkSquareBishops, numKnights, numPawns;
   int numOppQueens, numOppRooks, numOppLightSquareBishops;
   int numOppDarkSquareBishops, numOppKnights, numOppPawns;
   int numPieces, numOppPieces;
   int numOppBishops, numOppMinors, numOppSliders;
   const UINT32 signature =
      bilateralSignature(materialSignature, oppMaterialSignature);
   int pieceCountDiff;

   getPieceCounters(signature, &numQueens, &numRooks,
                    &numLightSquareBishops, &numDarkSquareBishops,
                    &numKnights, &numPawns, &numOppQueens, &numOppRooks,
                    &numOppLightSquareBishops, &numOppDarkSquareBishops,
                    &numOppKnights, &numOppPawns);
   numPieces = numQueens + numRooks + numLightSquareBishops +
      numDarkSquareBishops + numKnights;
   numOppBishops = numOppLightSquareBishops + numOppDarkSquareBishops;
   numOppMinors = numOppBishops + numOppKnights;
   numOppSliders = numOppQueens + numOppRooks + numOppBishops;
   numOppPieces = numOppSliders + numOppKnights;
   pieceCountDiff = numPieces - numOppPieces;

   if (numPieces == 0)
   {
      return 0;
   }

   if (numPieces == 1)
   {
      if (numQueens > 0 && numOppRooks > 0 && numOppRooks + numOppMinors >= 2)
      {
         return 1;
      }

      if (numQueens > 0 && numOppKnights >= 2)
      {
         return 1;
      }

      if (numRooks > 0 && numOppQueens + numOppRooks > 0)
      {
         return 1;
      }

      if (numRooks > 0 && numOppMinors > 0)
      {
         return (numOppMinors == 1 ? 2 : 1);
      }

      if (numLightSquareBishops + numDarkSquareBishops + numKnights > 0)
      {
         return 0;
      }
   }
   else if (numPieces == 2)
   {
      if (pieceCountDiff <= 1)
      {
         if (numOppRooks > 0 && numOppRooks >= numRooks)
         {
            return 0;
         }
      }
   }
   else if (numPieces <= 3)
   {
      if (numQueens == 0 && numOppQueens > 0)
      {
         if (numRooks <= 1)
         {
            return 1;
         }
         else
         {
            return 8;           /* hard to win */
         }
      }

      if (numRooks + numQueens == 0 &&
          numLightSquareBishops + numDarkSquareBishops <= 1 &&
          numOppRooks + numOppQueens >= 1)
      {
         return (numOppQueens >= 1 ? 1 : 2);
      }
   }

   if (numLightSquareBishops == 1 && numDarkSquareBishops == 1)
   {
      if (numOppPieces == 1 && numOppRooks == 1)
      {
         return (numPieces == 2 ? 1 : 12);
      }

      if (numOppKnights > 0)
      {
         return 8;              /* hard to win sometimes */
      }
   }

   return 16;
}

static void testMaterialSignatureNew(const int numWhiteQueens,
                                     const int numWhiteRooks,
                                     const int numWhiteLightSquareBishops,
                                     const int numWhiteDarkSquareBishops,
                                     const int numWhiteKnights,
                                     const int numWhitePawns,
                                     const int numBlackQueens,
                                     const int numBlackRooks,
                                     const int numBlackLightSquareBishops,
                                     const int numBlackDarkSquareBishops,
                                     const int numBlackKnights,
                                     const int numBlackPawns)
{
   int calculatedNumWhiteQueens;
   int calculatedNumWhiteRooks;
   int calculatedNumWhiteLightSquareBishops;
   int calculatedNumWhiteDarkSquareBishops;
   int calculatedNumWhiteKnights;
   int calculatedNumWhitePawns;
   int calculatedNumBlackQueens;
   int calculatedNumBlackRooks;
   int calculatedNumBlackLightSquareBishops;
   int calculatedNumBlackDarkSquareBishops;
   int calculatedNumBlackKnights;
   int calculatedNumBlackPawns;

   if (numWhiteRooks <= 2 && numWhiteKnights <= 2 &&
       numBlackRooks <= 2 && numBlackKnights <= 2)
   {
      const UINT32 signature = getMaterialSignature(numWhiteQueens,
                                                    numWhiteRooks,
                                                    numWhiteLightSquareBishops,
                                                    numWhiteDarkSquareBishops,
                                                    numWhiteKnights,
                                                    numWhitePawns,
                                                    numBlackQueens,
                                                    numBlackRooks,
                                                    numBlackLightSquareBishops,
                                                    numBlackDarkSquareBishops,
                                                    numBlackKnights,
                                                    numBlackPawns);

      getPieceCounters(signature,
                       &calculatedNumWhiteQueens, &calculatedNumWhiteRooks,
                       &calculatedNumWhiteLightSquareBishops,
                       &calculatedNumWhiteDarkSquareBishops,
                       &calculatedNumWhiteKnights,
                       &calculatedNumWhitePawns, &calculatedNumBlackQueens,
                       &calculatedNumBlackRooks,
                       &calculatedNumBlackLightSquareBishops,
                       &calculatedNumBlackDarkSquareBishops,
                       &calculatedNumBlackKnights, &calculatedNumBlackPawns);

      assert(calculatedNumWhiteQueens == numWhiteQueens);
      assert(calculatedNumWhiteRooks == numWhiteRooks);
      assert(calculatedNumWhiteLightSquareBishops ==
             numWhiteLightSquareBishops);
      assert(calculatedNumWhiteDarkSquareBishops ==
             numWhiteDarkSquareBishops);
      assert(calculatedNumWhiteKnights == numWhiteKnights);
      assert(calculatedNumWhitePawns == numWhitePawns);
      assert(calculatedNumBlackQueens == numBlackQueens);
      assert(calculatedNumBlackRooks == numBlackRooks);
      assert(calculatedNumBlackLightSquareBishops ==
             numBlackLightSquareBishops);
      assert(calculatedNumBlackDarkSquareBishops ==
             numBlackDarkSquareBishops);
      assert(calculatedNumBlackKnights == numBlackKnights);
      assert(calculatedNumBlackPawns == numBlackPawns);
   }
}

static int calculatePhase(UINT32 signature)
{
   int numWhiteQueens;
   int numWhiteRooks;
   int numWhiteLightSquareBishops;
   int numWhiteDarkSquareBishops;
   int numWhiteKnights;
   int numWhitePawns;
   int numBlackQueens;
   int numBlackRooks;
   int numBlackLightSquareBishops;
   int numBlackDarkSquareBishops;
   int numBlackKnights;
   int numBlackPawns;
   int whiteWeight, blackWeight, basicPhase;

   getPieceCounters(signature,
                    &numWhiteQueens, &numWhiteRooks,
                    &numWhiteLightSquareBishops,
                    &numWhiteDarkSquareBishops,
                    &numWhiteKnights, &numWhitePawns,
                    &numBlackQueens, &numBlackRooks,
                    &numBlackLightSquareBishops,
                    &numBlackDarkSquareBishops,
                    &numBlackKnights, &numBlackPawns);

   whiteWeight =
      9 * numWhiteQueens + 5 * numWhiteRooks +
      3 * numWhiteLightSquareBishops + 3 * numWhiteDarkSquareBishops +
      3 * numWhiteKnights;
   blackWeight =
      9 * numBlackQueens + 5 * numBlackRooks +
      3 * numBlackLightSquareBishops + 3 * numBlackDarkSquareBishops +
      3 * numBlackKnights;

   basicPhase = (whiteWeight + blackWeight <= PIECE_WEIGHT_ENDGAME ?
                 PHASE_MAX : max(0, PHASE_MAX - whiteWeight - blackWeight));

   return (basicPhase * 256 + (PHASE_MAX / 2)) / PHASE_MAX;
}

static INT32 calculateMaterialBalance(UINT32 signature)
{
   const INT32 bishopPairBonus =
      V(VALUE_BISHOP_PAIR_OPENING, VALUE_BISHOP_PAIR_ENDGAME);
   static const INT32 knightBonus = V(0, 5);
   static const INT32 rookMalus = V(5, 0);
   static const INT32 rookPairMalus = V(17, 25);
   static const INT32 rookQueenMalus = V(8, 12);
   static const INT32 pieceUpBonus =
      V(DEFAULTVALUE_PIECE_UP_OPENING, DEFAULTVALUE_PIECE_UP_ENDGAME);

   int numWhiteQueens;
   int numWhiteRooks;
   int numWhiteLightSquareBishops;
   int numWhiteDarkSquareBishops;
   int numWhiteKnights;
   int numWhitePawns;
   int numBlackQueens;
   int numBlackRooks;
   int numBlackLightSquareBishops;
   int numBlackDarkSquareBishops;
   int numBlackKnights;
   int numBlackPawns;
   int pawnCountWhite, pawnCountBlack;
   int knightSaldo, rookSaldo, pieceCountSaldo;
   INT32 balance = 0;

   getPieceCounters(signature,
                    &numWhiteQueens, &numWhiteRooks,
                    &numWhiteLightSquareBishops,
                    &numWhiteDarkSquareBishops,
                    &numWhiteKnights,
                    &numWhitePawns, &numBlackQueens,
                    &numBlackRooks,
                    &numBlackLightSquareBishops,
                    &numBlackDarkSquareBishops,
                    &numBlackKnights, &numBlackPawns);

   pawnCountWhite = numWhitePawns - 5;
   pawnCountBlack = numBlackPawns - 5;
   knightSaldo = pawnCountWhite * numWhiteKnights -
      pawnCountBlack * numBlackKnights;
   rookSaldo = pawnCountWhite * numWhiteRooks -
      pawnCountBlack * numBlackRooks;
   pieceCountSaldo =
      (numWhiteLightSquareBishops +
       numWhiteDarkSquareBishops + numWhiteKnights) -
      (numBlackLightSquareBishops +
       numBlackDarkSquareBishops + numBlackKnights);

   if (numWhiteLightSquareBishops > 0 && numWhiteDarkSquareBishops > 0)
   {
      balance += bishopPairBonus;
   }

   if (numBlackLightSquareBishops > 0 && numBlackDarkSquareBishops > 0)
   {
      balance -= bishopPairBonus;
   }

   balance += knightSaldo * knightBonus - rookSaldo * rookMalus;

   if (numWhiteRooks >= 2)
   {
      balance -= rookPairMalus + rookQueenMalus;
   }
   else if (numWhiteRooks + numWhiteQueens >= 2)
   {
      balance -= rookQueenMalus;
   }

   if (numBlackRooks >= 2)
   {
      balance += rookPairMalus + rookQueenMalus;
   }
   else if (numBlackRooks + numBlackQueens >= 2)
   {
      balance += rookQueenMalus;
   }

   if (pieceCountSaldo > 0)
   {
      balance += pieceUpBonus;
   }
   else if (pieceCountSaldo < 0)
   {
      balance -= pieceUpBonus;
   }

   return balance;
}

static void initializeMaterialInfoTableCore1(const UINT32 signatureWhite,
                                             const UINT32 signatureBlack)
{
   const UINT32 signature =
      bilateralSignature(signatureWhite, signatureBlack);
   SpecialEvalType specialEvalWhite = Se_None;
   SpecialEvalType specialEvalBlack = Se_None;
   const bool whiteMateMat =
      hasMaterialForMate(signatureWhite, signatureBlack, &specialEvalWhite,
                         FALSE, FALSE);
   const bool blackMateMat =
      hasMaterialForMate(signatureBlack, signatureWhite, &specialEvalBlack,
                         FALSE, FALSE);
   int numWhiteQueens, numWhiteRooks, numWhiteLightSquareBishops;
   int numWhiteDarkSquareBishops, numWhiteKnights, numWhitePawns;
   int numBlackQueens, numBlackRooks, numBlackLightSquareBishops;
   int numBlackDarkSquareBishops, numBlackKnights, numBlackPawns;

   /*int criticalCase = FALSE; */

   getPieceCounters(signature, &numWhiteQueens, &numWhiteRooks,
                    &numWhiteLightSquareBishops,
                    &numWhiteDarkSquareBishops, &numWhiteKnights,
                    &numWhitePawns, &numBlackQueens, &numBlackRooks,
                    &numBlackLightSquareBishops,
                    &numBlackDarkSquareBishops, &numBlackKnights,
                    &numBlackPawns);

   testMaterialSignatureNew(numWhiteQueens,
                            numWhiteRooks,
                            numWhiteLightSquareBishops,
                            numWhiteDarkSquareBishops,
                            numWhiteKnights,
                            numWhitePawns,
                            numBlackQueens,
                            numBlackRooks,
                            numBlackLightSquareBishops,
                            numBlackDarkSquareBishops,
                            numBlackKnights, numBlackPawns);

   materialInfo[signature].chancesWhite = (whiteMateMat == FALSE ? 0 : 16);
   materialInfo[signature].chancesBlack = (blackMateMat == FALSE ? 0 : 16);
   materialInfo[signature].specialEvalWhite = specialEvalWhite;
   materialInfo[signature].specialEvalBlack = specialEvalBlack;

   /*if (numWhiteQueens == 0 && numWhiteRooks == 0 &&
      numWhiteLightSquareBishops == 0 &&
      numWhiteDarkSquareBishops == 0 &&
      numWhiteKnights == 0 && numWhitePawns == 1 &&
      numBlackQueens == 0 && numBlackRooks == 0 &&
      numBlackLightSquareBishops == 0 &&
      numBlackDarkSquareBishops == 1 &&
      numBlackKnights == 0 && numBlackPawns == 0)
      {
      criticalCase = TRUE;
      } */

   if (whiteMateMat != FALSE)
   {
      if (numWhitePawns == 0)
      {
         if (hasMaterialForMate(signatureWhite, signatureBlack, 0,
                                TRUE, TRUE) == FALSE)
         {
            materialInfo[signature].chancesWhite = 1;
         }
         else
         {
            materialInfo[signature].chancesWhite =
               getWinningChancesWithoutPawn(signatureWhite, signatureBlack);
         }
      }
      else
      {
         materialInfo[signature].chancesWhite =
            getWinningChances(signatureWhite, signatureBlack);
      }
   }

   if (blackMateMat != FALSE)
   {
      if (numBlackPawns == 0)
      {
         if (hasMaterialForMate(signatureBlack, signatureWhite, 0,
                                TRUE, TRUE) == FALSE)
         {
            materialInfo[signature].chancesBlack = 1;
         }
         else
         {
            materialInfo[signature].chancesBlack =
               getWinningChancesWithoutPawn(signatureBlack, signatureWhite);
         }
      }
      else
      {
         materialInfo[signature].chancesBlack =
            getWinningChances(signatureBlack, signatureWhite);
      }

      /*if (criticalCase){
         logDebug("wc=%d",materialInfo[signature].chancesBlack);
         getKeyStroke();

         materialInfo[signature].chancesBlack=0;
         } */
   }

   materialInfo[signature].materialBalance =
      calculateMaterialBalance(signature);
   materialInfo[signature].phaseIndex = calculatePhase(signature);
}

static void initializeMaterialInfoTable(void)
{
   int whiteSignature, blackSignature;

   for (whiteSignature = 0; whiteSignature < 648; whiteSignature++)
   {
      for (blackSignature = 0; blackSignature < 648; blackSignature++)
      {
         initializeMaterialInfoTableCore1(whiteSignature, blackSignature);
      }
   }
}

/* #define DEBUG_KSTABLE 1 */

static int initializeKingSafetyTable(void)
{
   const double MIN_SLOPE = 1.0;
   const double MAX_SLOPE = 7.0;
   const double MAX_VALUE = 1347.0;
   double t = 0.0 - MIN_SLOPE;
   int i;

   for (i = 0; i < KING_SAFETY_MALUS_DIM; i++)
   {
      t = max(t + MIN_SLOPE,
              min(MAX_VALUE, min(0.025 * i * i, t + MAX_SLOPE)));
      KING_SAFETY_MALUS[i] = (156 * t) / 256;

#ifdef DEBUG_KSTABLE
      logDebug("ksm(%d)=%d\n", i, KING_SAFETY_MALUS[i]);
#endif
   }

#ifdef DEBUG_KSTABLE
   getKeyStroke();
#endif

   return 0;
}

int getLogarithmicValue(const double minValue, const double maxValue,
                        const double numValues, const double valueCount)
{
   const double offset = 1.0;
   const double factor = (maxValue - minValue) / log(numValues + offset);
   const double value = factor * log(valueCount + offset) + minValue;

   return (int) value;
}

double wmbv(const double baseValue)
{
   return baseValue * 1.075;
}

/* #define LOG_MOBILILITY_VALUES */
static void initializeMoveBonusValue(void)
{
   int i, limit;

   limit = MAX_MOVES_QUEEN;
   for (i = 0; i <= limit; i++)
   {
      const int opValue =
         getLogarithmicValue(wmbv(-21.0), wmbv(13.0), limit, i);
      const int egValue =
         getLogarithmicValue(wmbv(-20.0), wmbv(21.0), limit, i);

      QueenMobilityBonus[i] = V(opValue, egValue);

#ifdef LOG_MOBILILITY_VALUES
      logDebug("mQ(%d)=(%d/%d)\n", i, getOpeningValue(QueenMobilityBonus[i]),
               getEndgameValue(QueenMobilityBonus[i]));
#endif
   }

#ifdef LOG_MOBILILITY_VALUES
   logDebug("\n");
#endif

   limit = MAX_MOVES_ROOK;
   for (i = 0; i <= limit; i++)
   {
      const int opValue =
         getLogarithmicValue(wmbv(-24.0), wmbv(19.0), limit, i);
      const int egValue =
         getLogarithmicValue(wmbv(-27.0), wmbv(62.0), limit, i);

      RookMobilityBonus[i] = V(opValue, egValue);

#ifdef LOG_MOBILILITY_VALUES
      logDebug("mR(%d)=(%d/%d)\n", i, getOpeningValue(RookMobilityBonus[i]),
               getEndgameValue(RookMobilityBonus[i]));
#endif
   }

#ifdef LOG_MOBILILITY_VALUES
   logDebug("\n");
#endif

   limit = MAX_MOVES_BISHOP;
   for (i = 0; i <= limit; i++)
   {
      const int opValue =
         getLogarithmicValue(wmbv(-26.0), wmbv(43.0), limit, i);
      const int egValue =
         getLogarithmicValue(wmbv(-24.0), wmbv(40.0), limit, i);

      BishopMobilityBonus[i] = V(opValue, egValue);

#ifdef LOG_MOBILILITY_VALUES
      logDebug("mB(%d)=(%d/%d)\n", i, getOpeningValue(BishopMobilityBonus[i]),
               getEndgameValue(BishopMobilityBonus[i]));
#endif
   }

#ifdef LOG_MOBILILITY_VALUES
   logDebug("\n");
#endif

   limit = MAX_MOVES_KNIGHT;
   for (i = 0; i <= limit; i++)
   {
      const int opValue =
         getLogarithmicValue(wmbv(-33.0), wmbv(22.0), limit, i);
      const int egValue =
         getLogarithmicValue(wmbv(-25.0), wmbv(17.0), limit, i);

      KnightMobilityBonus[i] = V(opValue, egValue);

#ifdef LOG_MOBILILITY_VALUES
      logDebug("mN(%d)=(%d/%d)\n", i, getOpeningValue(KnightMobilityBonus[i]),
               getEndgameValue(KnightMobilityBonus[i]));
#endif
   }

#ifdef LOG_MOBILILITY_VALUES
   getKeyStroke();
#endif
}

static INT32 PPAB(const int opv, const int egv)
{
   const int wopv = (opv * 125) / 256;
   const int wegv = (egv * 133) / 256;

   return V(wopv, wegv);
}

static void initializePawnChainBonus(void)
{
   const int bonusPerFile[8] = { 1, 3, 3, 4, 4, 3, 3, 1 };
   Square square;

   ITERATE(square)
   {
      const File squarefile = file(square);
      const Rank squarerank = rank(square);
      const int bonus = squarerank * (squarerank - 1) * (squarerank - 2) +
         bonusPerFile[squarefile] * (squarerank / 2 + 1);

      PAWN_CHAIN_BONUS[square] = V((bonus * CHAIN_BONUS_WEIGHT_OP) / 256,
                                   (bonus * CHAIN_BONUS_WEIGHT_EG) / 256);
   }
}

int initializeModuleEvaluation(void)
{
   int i;
   Square square, kingsquare, catchersquare;

   centralFiles = squaresOfFile[FILE_D] | squaresOfFile[FILE_E];
   attackingRealm[WHITE] = squaresOfRank[RANK_5] | squaresOfRank[RANK_6] |
      squaresOfRank[RANK_7] | squaresOfRank[RANK_8];
   attackingRealm[BLACK] = squaresOfRank[RANK_4] | squaresOfRank[RANK_3] |
      squaresOfRank[RANK_2] | squaresOfRank[RANK_1];
   filesBCFG = squaresOfFileRange[FILE_B][FILE_C] |
      squaresOfFileRange[FILE_F][FILE_G];

   ITERATE(square)
   {
      Color color;
      Square kingSquare;

      for (color = WHITE; color <= BLACK; color++)
      {
         passedPawnRectangle[color][square] =
            passedPawnCorridor[color][square] =
            candidateDefenders[color][square] =
            candidateSupporters[color][square] =
            pawnOpponents[color][square] = EMPTY_BITBOARD;
      }

      ITERATE(kingSquare)
      {
         kingRealm[WHITE][square][kingSquare] =
            kingRealm[BLACK][square][kingSquare] = EMPTY_BITBOARD;
      }

      kingTrapsRook[WHITE][square] = kingTrapsRook[BLACK][square] =
         EMPTY_BITBOARD;
   }

   setSquare(kingTrapsRook[WHITE][F1], H1);     /* a king on f1 traps a rook on h1 ... */
   setSquare(kingTrapsRook[WHITE][F1], G1);
   setSquare(kingTrapsRook[WHITE][F1], H2);
   setSquare(kingTrapsRook[WHITE][F1], G2);
   setSquare(kingTrapsRook[WHITE][G1], H1);
   setSquare(kingTrapsRook[WHITE][G1], H2);
   setSquare(kingTrapsRook[WHITE][G1], G2);
   setSquare(kingTrapsRook[WHITE][G2], H2);

   setSquare(kingTrapsRook[WHITE][C1], A1);
   setSquare(kingTrapsRook[WHITE][C1], B1);
   setSquare(kingTrapsRook[WHITE][C1], A2);
   setSquare(kingTrapsRook[WHITE][C1], B2);
   setSquare(kingTrapsRook[WHITE][B1], A1);
   setSquare(kingTrapsRook[WHITE][B1], A2);
   setSquare(kingTrapsRook[WHITE][B1], B2);
   setSquare(kingTrapsRook[WHITE][B2], A2);

   setSquare(kingTrapsRook[BLACK][F8], H8);
   setSquare(kingTrapsRook[BLACK][F8], G8);
   setSquare(kingTrapsRook[BLACK][F8], H7);
   setSquare(kingTrapsRook[BLACK][F8], G7);
   setSquare(kingTrapsRook[BLACK][G8], H8);
   setSquare(kingTrapsRook[BLACK][G8], H7);
   setSquare(kingTrapsRook[BLACK][G8], G7);
   setSquare(kingTrapsRook[BLACK][G7], H7);

   setSquare(kingTrapsRook[BLACK][C8], A8);
   setSquare(kingTrapsRook[BLACK][C8], B8);
   setSquare(kingTrapsRook[BLACK][C8], A7);
   setSquare(kingTrapsRook[BLACK][C8], B7);
   setSquare(kingTrapsRook[BLACK][B8], A8);
   setSquare(kingTrapsRook[BLACK][B8], A7);
   setSquare(kingTrapsRook[BLACK][B8], B7);
   setSquare(kingTrapsRook[BLACK][B7], A7);

   ITERATE(square)
   {
      const File squarefile = file(square);
      const Rank squarerank = rank(square);
      int d1 = min(distance(square, D4), distance(square, E4));
      int d2 = min(distance(square, D5), distance(square, E5));
      int td1 = min(taxiDistance(square, D4), taxiDistance(square, E4));
      int td2 = min(taxiDistance(square, D5), taxiDistance(square, E5));

      centerDistance[square] = min(d1, d2);
      centerTaxiDistance[square] = min(td1, td2);
      butterflySquares[square] =
         generalMoves[KING][square] & ~squaresOfFile[squarefile];
      lateralSquares[square] =
         generalMoves[KING][square] & squaresOfRank[squarerank];
      companionFiles[square] =
         ((squaresOfFile[squarefile] & nonA) >> 1) |
         ((squaresOfFile[squarefile] & nonH) << 1);
      rookBlockers[square] = EMPTY_BITBOARD;

      ITERATE(kingsquare)
      {
         const File kingsquarefile = file(kingsquare);
         const Rank kingsquarerank = rank(kingsquare);
         Square targetSquare;

         if (kingsquarerank >= squarerank &&
             distance(square, kingsquare) <= (int) (7 - squarerank))
         {
            setSquare(passedPawnRectangle[WHITE][square], kingsquare);
         }

         if (kingsquarerank <= squarerank &&
             distance(square, kingsquare) <= (int) squarerank)
         {
            setSquare(passedPawnRectangle[BLACK][square], kingsquare);
         }

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

         if (squarerank == kingsquarerank)
         {
            if (squarefile <= FILE_C && kingsquarefile <= FILE_C &&
                kingsquarefile > squarefile)
            {
               setSquare(rookBlockers[square], kingsquare);
            }

            if (squarefile >= FILE_F && kingsquarefile >= FILE_F &&
                kingsquarefile < squarefile)
            {
               setSquare(rookBlockers[square], kingsquare);
            }
         }

         ITERATE(targetSquare)
         {
            if (distance(square, targetSquare) <
                distance(kingsquare, targetSquare))
            {
               const Rank targetrank = rank(targetSquare);

               if (targetrank <= (Rank) (squarerank + 1))
               {
                  setSquare(kingRealm[WHITE][square][kingsquare],
                            targetSquare);
               }

               if (targetrank >= (Rank) (squarerank - 1))
               {
                  setSquare(kingRealm[BLACK][square][kingsquare],
                            targetSquare);
               }
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

            if (rank(catchersquare) <= squarerank)
            {
               setSquare(candidateSupporters[WHITE][square], catchersquare);
            }

            if (rank(catchersquare) < squarerank)
            {
               setSquare(candidateDefenders[BLACK][square], catchersquare);
            }

            if (rank(catchersquare) >= squarerank)
            {
               setSquare(candidateSupporters[BLACK][square], catchersquare);
            }
         }

         if (((file(catchersquare) > squarefile) ? (file(catchersquare) - squarefile) : (squarefile - file(catchersquare))) <= 1)
         {
            if (rank(catchersquare) >= squarerank)
            {
               setSquare(pawnOpponents[WHITE][square], catchersquare);
            }

            if (rank(catchersquare) <= squarerank)
            {
               setSquare(pawnOpponents[BLACK][square], catchersquare);
            }
         }
      }
   }

   ITERATE(square)
   {
      const int dDark =
         min(taxiDistance(square, A1), taxiDistance(square, H8));
      const int dLight =
         min(taxiDistance(square, A8), taxiDistance(square, H1));
      const int dStandard = centerDistance[square];

      kingChaseMalus[DARK][square] = 3 * (7 - dDark) + dStandard;
      kingChaseMalus[LIGHT][square] = 3 * (7 - dLight) + dStandard;
      kingChaseMalus[ALL][square] = 6 - min(dDark, dLight) +
         centerDistance[square];
   }

   /*
      dumpBoardValues(kingChaseMalus[DARK]);
      dumpBoardValues(kingChaseMalus[LIGHT]);
      dumpBoardValues(kingChaseMalus[ALL]);
      getKeyStroke();
    */

   transposeMatrix(BONUS_KNIGHT_OUTPOST_HR, BONUS_KNIGHT_OUTPOST);
   transposeMatrix(BONUS_BISHOP_OUTPOST_HR, BONUS_BISHOP_OUTPOST);

   initializePieceSquareValues();
   initializeKingAttacks();
   initializeKingSafetyTable();

   attackPoints[WHITE_KING] = 0;
   attackPoints[WHITE_QUEEN] = QUEEN_BONUS_ATTACK;
   attackPoints[WHITE_ROOK] = ROOK_BONUS_ATTACK;
   attackPoints[WHITE_BISHOP] = BISHOP_BONUS_ATTACK;
   attackPoints[WHITE_KNIGHT] = KNIGHT_BONUS_ATTACK;
   attackPoints[WHITE_PAWN] = 0;
   attackPoints[BLACK_KING] = 0;
   attackPoints[BLACK_QUEEN] = QUEEN_BONUS_ATTACK;
   attackPoints[BLACK_ROOK] = ROOK_BONUS_ATTACK;
   attackPoints[BLACK_BISHOP] = BISHOP_BONUS_ATTACK;
   attackPoints[BLACK_KNIGHT] = KNIGHT_BONUS_ATTACK;
   attackPoints[BLACK_PAWN] = 0;

   homeland[WHITE] = (squaresOfRank[RANK_2] | squaresOfRank[RANK_3] |
                      squaresOfRank[RANK_4]) &
      (squaresOfFile[FILE_C] | squaresOfFile[FILE_D] |
       squaresOfFile[FILE_E] | squaresOfFile[FILE_F]);

   homeland[BLACK] = getFlippedBitboard(homeland[WHITE]);

   for (i = 0; i < 16; i++)
   {
      int j;

      for (j = 0; j < 16; j++)
      {
         piecePieceAttackBonus[i][j] = 0;
      }
   }

   piecePieceAttackBonus[WHITE_PAWN][BLACK_KNIGHT] = PPAB(56, 70);
   piecePieceAttackBonus[WHITE_PAWN][BLACK_BISHOP] = PPAB(56, 70);
   piecePieceAttackBonus[WHITE_PAWN][BLACK_ROOK] = PPAB(76, 99);
   piecePieceAttackBonus[WHITE_PAWN][BLACK_QUEEN] = PPAB(86, 118);

   piecePieceAttackBonus[WHITE_KNIGHT][BLACK_PAWN] = PPAB(7, 39);
   piecePieceAttackBonus[WHITE_KNIGHT][BLACK_BISHOP] = PPAB(24, 49);
   piecePieceAttackBonus[WHITE_KNIGHT][BLACK_ROOK] = PPAB(41, 100);
   piecePieceAttackBonus[WHITE_KNIGHT][BLACK_QUEEN] = PPAB(41, 100);

   piecePieceAttackBonus[WHITE_BISHOP][BLACK_PAWN] = PPAB(7, 39);
   piecePieceAttackBonus[WHITE_BISHOP][BLACK_KNIGHT] = PPAB(24, 49);
   piecePieceAttackBonus[WHITE_BISHOP][BLACK_ROOK] = PPAB(41, 100);
   piecePieceAttackBonus[WHITE_BISHOP][BLACK_QUEEN] = PPAB(41, 100);

   piecePieceAttackBonus[WHITE_ROOK][BLACK_PAWN] = PPAB(0, 22);
   piecePieceAttackBonus[WHITE_ROOK][BLACK_KNIGHT] = PPAB(15, 49);
   piecePieceAttackBonus[WHITE_ROOK][BLACK_BISHOP] = PPAB(15, 49);
   piecePieceAttackBonus[WHITE_ROOK][BLACK_QUEEN] = PPAB(24, 49);

   piecePieceAttackBonus[WHITE_QUEEN][BLACK_PAWN] = PPAB(15, 39);
   piecePieceAttackBonus[WHITE_QUEEN][BLACK_KNIGHT] = PPAB(15, 39);
   piecePieceAttackBonus[WHITE_QUEEN][BLACK_BISHOP] = PPAB(15, 39);
   piecePieceAttackBonus[WHITE_QUEEN][BLACK_ROOK] = PPAB(15, 39);

   for (i = 0; i < 16; i++)
   {
      int j;

      for (j = 0; j < 16; j++)
      {
         if (pieceColor(i) == BLACK)
         {
            const Color reversedColor = opponent(pieceColor(j));
            PieceType attacker = (PieceType) (pieceType(i) | WHITE);
            PieceType attackedPiece =
               (PieceType) (pieceType(j) | reversedColor);

            piecePieceAttackBonus[i][j] =
               piecePieceAttackBonus[attacker][attackedPiece];
         }
      }
   }

   troitzkyArea[WHITE] =
      passedPawnCorridor[WHITE][A3] | passedPawnCorridor[WHITE][B5] |
      passedPawnCorridor[WHITE][C3] | passedPawnCorridor[WHITE][D3] |
      passedPawnCorridor[WHITE][E3] | passedPawnCorridor[WHITE][F3] |
      passedPawnCorridor[WHITE][G5] | passedPawnCorridor[WHITE][H3];
   troitzkyArea[BLACK] = getFlippedBitboard(troitzkyArea[WHITE]);

   krprkDrawFiles = squaresOfFile[FILE_A] | squaresOfFile[FILE_B] |
      squaresOfFile[FILE_G] | squaresOfFile[FILE_H];
   A1C1 = minValue[A1] | minValue[C1], F1H1 = minValue[F1] | minValue[H1];
   A1B1 = minValue[A1] | minValue[B1], G1H1 = minValue[G1] | minValue[H1];

   initializeMaterialInfoTable();
   initializeMoveBonusValue();
   initializePawnChainBonus();

   return 0;
}

#ifndef NDEBUG
bool flipTest(Position * position,
              PawnHashInfo * pawnHashtable,
              KingSafetyHashInfo * kingsafetyHashtable)
{
   int v1, v2;
   EvaluationBase base;

   initializePosition(position);
   v1 = getValue(position, &base, pawnHashtable, kingsafetyHashtable);

   flipPosition(position);
   initializePosition(position);
   v2 = getValue(position, &base, pawnHashtable, kingsafetyHashtable);

   flipPosition(position);
   initializePosition(position);

   if (v1 != v2)
   {
      const int debugFlag = debugOutput;
      const bool debugEvalFlag = debugEval;

      debugOutput = TRUE;
      debugEval = TRUE;

      logDebug("flip test failed: v1=%d v2=%d\n", v1, v2);
      logPosition(position);
      logDebug("hash: %llu\n", position->hashKey);
      getValue(position, &base, pawnHashtable, kingsafetyHashtable);
      flipPosition(position);
      initializePosition(position);
      logPosition(position);
      getValue(position, &base, pawnHashtable, kingsafetyHashtable);
      flipPosition(position);
      initializePosition(position);

      debugEval = debugEvalFlag;
      debugOutput = debugFlag;
   }

   return (bool) (v1 == v2);
}
#endif

static int testPawnInfoGeneration(void)
{
   Variation variation;
   EvaluationBase base;

   initializeVariation(&variation,
                       "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1");
   getPawnInfo(&variation.singlePosition, &base);

   assert(getNumberOfSetSquares(base.pawnProtectedSquares[WHITE]) == 8);
   assert(testSquare(base.pawnProtectedSquares[WHITE], B4));
   assert(testSquare(base.pawnProtectedSquares[WHITE], A3));
   assert(testSquare(base.pawnProtectedSquares[WHITE], C3));
   assert(testSquare(base.pawnProtectedSquares[WHITE], D4));
   assert(testSquare(base.pawnProtectedSquares[WHITE], F4));
   assert(testSquare(base.pawnProtectedSquares[WHITE], E5));
   assert(testSquare(base.pawnProtectedSquares[WHITE], G5));
   assert(testSquare(base.pawnProtectedSquares[WHITE], G3));

   assert(getNumberOfSetSquares(base.pawnProtectedSquares[BLACK]) == 7);
   assert(testSquare(base.pawnProtectedSquares[BLACK], B3));
   assert(testSquare(base.pawnProtectedSquares[BLACK], D3));
   assert(testSquare(base.pawnProtectedSquares[BLACK], C2));
   assert(testSquare(base.pawnProtectedSquares[BLACK], E2));
   assert(testSquare(base.pawnProtectedSquares[BLACK], E4));
   assert(testSquare(base.pawnProtectedSquares[BLACK], G4));
   assert(testSquare(base.pawnProtectedSquares[BLACK], G6));

   assert(getNumberOfSetSquares(base.passedPawns[WHITE]) == 0);
   assert(getNumberOfSetSquares(base.passedPawns[BLACK]) == 1);
   assert(testSquare(base.passedPawns[BLACK], D3));

   assert(getNumberOfSetSquares(base.weakPawns[WHITE]) == 3);
   assert(testSquare(base.weakPawns[WHITE], B2));
   assert(testSquare(base.weakPawns[WHITE], E3));
   assert(testSquare(base.weakPawns[WHITE], H2));

   assert(getNumberOfSetSquares(base.weakPawns[BLACK]) == 4);
   assert(testSquare(base.weakPawns[BLACK], A4));
   assert(testSquare(base.weakPawns[BLACK], C4));
   assert(testSquare(base.weakPawns[BLACK], F5));
   assert(testSquare(base.weakPawns[BLACK], H7));

   initializeVariation(&variation,
                       "4k3/2p5/p2p4/P2P4/1PP3p1/7p/7P/4K3 w - - 0 1");
   getPawnInfo(&variation.singlePosition, &base);

   assert(getNumberOfSetSquares(base.passedPawns[WHITE]) == 0);
   assert(getNumberOfSetSquares(base.passedPawns[BLACK]) == 0);
   assert(getNumberOfSetSquares(base.weakPawns[WHITE]) == 1);
   assert(testSquare(base.weakPawns[WHITE], H2));

   assert(getNumberOfSetSquares(base.weakPawns[BLACK]) == 3);
   assert(testSquare(base.weakPawns[BLACK], A6));
   assert(testSquare(base.weakPawns[BLACK], C7));
   assert(testSquare(base.weakPawns[BLACK], G4));

   return 0;
}

static int testWeakPawnDetection(void)
{
   Position position;
   Bitboard expectedResult = EMPTY_BITBOARD;
   EvaluationBase base;

   clearPosition(&position);
   position.piece[E1] = WHITE_KING;
   position.piece[E8] = BLACK_KING;
   position.piece[A3] = WHITE_PAWN;
   position.piece[B5] = WHITE_PAWN;
   position.piece[B6] = WHITE_PAWN;
   position.piece[C4] = WHITE_PAWN;
   position.piece[E4] = WHITE_PAWN;
   position.piece[G4] = WHITE_PAWN;
   position.piece[H2] = WHITE_PAWN;
   position.piece[A4] = BLACK_PAWN;
   position.piece[B7] = BLACK_PAWN;
   setSquare(expectedResult, A3);
   setSquare(expectedResult, E4);
   initializePosition(&position);
   getPawnInfo(&position, &base);
   assert(base.weakPawns[WHITE] == expectedResult);
   expectedResult = EMPTY_BITBOARD;
   setSquare(expectedResult, B7);
   assert(base.weakPawns[BLACK] == expectedResult);
   assert(base.candidatePawns[BLACK] == EMPTY_BITBOARD);

   position.piece[C4] = NO_PIECE;
   position.piece[C5] = WHITE_PAWN;
   initializePosition(&position);
   getPawnInfo(&position, &base);
   expectedResult = EMPTY_BITBOARD;
   setSquare(expectedResult, C5);
   assert(base.candidatePawns[WHITE] == expectedResult);

   clearPosition(&position);
   position.piece[E1] = WHITE_KING;
   position.piece[E8] = BLACK_KING;
   position.piece[A3] = WHITE_PAWN;
   position.piece[B5] = WHITE_PAWN;
   position.piece[B6] = WHITE_PAWN;
   position.piece[C4] = WHITE_PAWN;
   position.piece[E4] = WHITE_PAWN;
   position.piece[G4] = WHITE_PAWN;
   position.piece[H2] = WHITE_PAWN;
   position.piece[A4] = BLACK_PAWN;
   position.piece[B7] = BLACK_PAWN;
   position.piece[D6] = BLACK_PAWN;
   expectedResult = EMPTY_BITBOARD;
   setSquare(expectedResult, A3);
   setSquare(expectedResult, E4);
   setSquare(expectedResult, C4);
   initializePosition(&position);
   getPawnInfo(&position, &base);
   assert(base.weakPawns[WHITE] == expectedResult);
   assert(base.candidatePawns[WHITE] == EMPTY_BITBOARD);
   expectedResult = EMPTY_BITBOARD;
   setSquare(expectedResult, B7);
   setSquare(expectedResult, D6);
   assert(base.weakPawns[BLACK] == expectedResult);
   assert(base.candidatePawns[BLACK] == EMPTY_BITBOARD);

   position.piece[G5] = BLACK_PAWN;
   expectedResult = EMPTY_BITBOARD;
   setSquare(expectedResult, A3);
   setSquare(expectedResult, E4);
   setSquare(expectedResult, C4);
   setSquare(expectedResult, H2);
   initializePosition(&position);
   getPawnInfo(&position, &base);
   assert(base.weakPawns[WHITE] == expectedResult);
   assert(base.candidatePawns[WHITE] == EMPTY_BITBOARD);
   expectedResult = EMPTY_BITBOARD;
   setSquare(expectedResult, B7);
   setSquare(expectedResult, D6);
   setSquare(expectedResult, G5);
   assert(base.weakPawns[BLACK] == expectedResult);
   assert(base.candidatePawns[BLACK] == EMPTY_BITBOARD);

   return 0;
}

static int testBaseInitialization(void)
{
   Variation variation;

   initializeVariation(&variation, FEN_GAMESTART);

   assert(testSquare(passedPawnCorridor[WHITE][B4], B6));
   assert(testSquare(passedPawnCorridor[BLACK][B4], B6) == FALSE);
   assert(testSquare(passedPawnCorridor[WHITE][C2], H7) == FALSE);
   assert(testSquare(passedPawnCorridor[BLACK][G6], G2));

#ifndef NDEBUG
   {
      INT32 testBonus = evalBonus(-1, -1);

      assert(getOpeningValue(testBonus) == -1);
      assert(getEndgameValue(testBonus) == -1);
   }
#endif

   return 0;
}

#ifndef NDEBUG
static void initializePawnHashtable(PawnHashInfo * pawnHashtable)
{
   int i;

   for (i = 0; i < PAWN_HASHTABLE_SIZE; i++)
   {
      pawnHashtable[i].hashKey = 0;
   }
}

static void initializeKingsafetyHashtable(KingSafetyHashInfo *
                                          kingsafetyHashtable)
{
   int i;

   for (i = 0; i < KINGSAFETY_HASHTABLE_SIZE; i++)
   {
      kingsafetyHashtable[i].hashKey = 0;
   }
}

static int testFlippings(void)
{
   const char fen1[] =
      "2rr2k1/1b3ppp/pb2p3/1p2P3/1P2BPnq/P1N3P1/1B2Q2P/R4R1K b - - 0 1";
   const char fen2[] = "4k3/2p5/p2p4/P2P4/1PP3p1/7p/7P/4K3 w - - 0 1";
   const char fen3[] = "8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1";
   const char fen4[] =
      "6r1/Q2Pn2k/p1p1P2p/5p2/2PqR1r1/1P6/P6P/5R1K b - - 5 4";
   const char fen5[] =
      "Q4rk1/2bb1ppp/4pn2/pQ5q/3P4/N4N2/5PPP/R1B2RK1 w - a6 0 4";
   Variation variation;

   initializeVariation(&variation, fen1);
   variation.pawnHashtable = localPawnHashtable;
   variation.kingsafetyHashtable = localKingSafetyHashtable;
   initializePawnHashtable(variation.pawnHashtable);
   initializeKingsafetyHashtable(variation.kingsafetyHashtable);
   assert(flipTest(&variation.singlePosition, variation.pawnHashtable,
                   variation.kingsafetyHashtable) != FALSE);

   initializeVariation(&variation, fen2);
   assert(flipTest(&variation.singlePosition, variation.pawnHashtable,
                   variation.kingsafetyHashtable) != FALSE);

   initializeVariation(&variation, fen3);
   assert(flipTest(&variation.singlePosition, variation.pawnHashtable,
                   variation.kingsafetyHashtable) != FALSE);

   initializeVariation(&variation, fen4);
   assert(flipTest(&variation.singlePosition, variation.pawnHashtable,
                   variation.kingsafetyHashtable) != FALSE);

   initializeVariation(&variation, fen5);
   assert(flipTest(&variation.singlePosition, variation.pawnHashtable,
                   variation.kingsafetyHashtable) != FALSE);

   return 0;
}
#endif

int testModuleEvaluation(void)
{
   int result;

   if ((result = testPawnInfoGeneration()) != 0)
   {
      return result;
   }

   if ((result = testWeakPawnDetection()) != 0)
   {
      return result;
   }

   if ((result = testBaseInitialization()) != 0)
   {
      return result;
   }

#ifndef NDEBUG
   if ((result = testFlippings()) != 0)
   {
      return result;
   }
#endif

   return 0;
}

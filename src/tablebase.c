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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tablebase.h"
#include "tbprobe.h"
#include "fen.h"
#include "movegeneration.h"

bool tbAvailable = false;

int initializeModuleTablebase(void)
{
   return 0;
}

int testModuleTablebase(void)
{
   return 0;
}

int initializeTablebase(const char *path)
{
   tbAvailable = tb_init(path);
   return tbAvailable ? 0 : -1;
}

void closeTablebaseFiles(void)
{
   // tb_init handles cleanup if called with NULL or implicitly on exit
}


int probeTablebaseWDL(const Position * position)
{
   if (!tbAvailable) return TABLEBASE_ERROR;

   if (position->castlingRights != 0) return TABLEBASE_ERROR;

   int numPieces = position->numberOfPieces[WHITE] + position->numberOfPieces[BLACK];
   if (numPieces <= 2 || numPieces > TB_LARGEST)
      return TABLEBASE_ERROR;

   unsigned res = tb_probe_wdl(
      position->piecesOfColor[WHITE],
      position->piecesOfColor[BLACK],
      position->piecesOfType[WHITE_KING] | position->piecesOfType[BLACK_KING],
      position->piecesOfType[WHITE_QUEEN] | position->piecesOfType[BLACK_QUEEN],
      position->piecesOfType[WHITE_ROOK] | position->piecesOfType[BLACK_ROOK],
      position->piecesOfType[WHITE_BISHOP] | position->piecesOfType[BLACK_BISHOP],
      position->piecesOfType[WHITE_KNIGHT] | position->piecesOfType[BLACK_KNIGHT],
      position->piecesOfType[WHITE_PAWN] | position->piecesOfType[BLACK_PAWN],
      position->halfMoveClock,
      0, // castling
      (position->enPassantSquare == NO_SQUARE ? 0 : position->enPassantSquare),
      (position->activeColor == WHITE)
   );

   if (res == TB_RESULT_FAILED) return TABLEBASE_ERROR;

   switch (res)
   {
      case TB_WIN: return 2;
      case TB_CURSED_WIN: return 1;
      case TB_DRAW: return 0;
      case TB_BLESSED_LOSS: return -1;
      case TB_LOSS: return -2;
      default: return TABLEBASE_ERROR;
   }
}

int probeTablebase(const Position * position)
{
   int res = probeTablebaseWDL(position);
   if (res == TABLEBASE_ERROR) return TABLEBASE_ERROR;

   switch (res)
   {
      case 2: return -VALUE_MATED - MAX_DEPTH - 1;
      case 1: return 1; // Slight advantage for cursed win
      case 0: return 0;
      case -1: return -1; // Slight disadvantage for blessed loss
      case -2: return VALUE_MATED + MAX_DEPTH + 1;
      default: return TABLEBASE_ERROR;
   }
}

int probeTablebaseDTZ(const Position * position, Move * move)
{
   if (!tbAvailable) return TABLEBASE_ERROR;

   if (position->castlingRights != 0) return TABLEBASE_ERROR;

   int numPieces = position->numberOfPieces[WHITE] + position->numberOfPieces[BLACK];
   if (numPieces <= 2 || numPieces > TB_LARGEST)
      return TABLEBASE_ERROR;

   unsigned res = tb_probe_root(
      position->piecesOfColor[WHITE],
      position->piecesOfColor[BLACK],
      position->piecesOfType[WHITE_KING] | position->piecesOfType[BLACK_KING],
      position->piecesOfType[WHITE_QUEEN] | position->piecesOfType[BLACK_QUEEN],
      position->piecesOfType[WHITE_ROOK] | position->piecesOfType[BLACK_ROOK],
      position->piecesOfType[WHITE_BISHOP] | position->piecesOfType[BLACK_BISHOP],
      position->piecesOfType[WHITE_KNIGHT] | position->piecesOfType[BLACK_KNIGHT],
      position->piecesOfType[WHITE_PAWN] | position->piecesOfType[BLACK_PAWN],
      position->halfMoveClock,
      0, // castling
      (position->enPassantSquare == NO_SQUARE ? 0 : position->enPassantSquare),
      (position->activeColor == WHITE),
      NULL
   );

   if (res == TB_RESULT_FAILED) return TABLEBASE_ERROR;

   if (move != NULL)
   {
      if (res == TB_RESULT_STALEMATE || res == TB_RESULT_CHECKMATE)
      {
         *move = NO_MOVE;
      }
      else
      {
         Square from = TB_GET_FROM(res);
         Square to = TB_GET_TO(res);
         unsigned promotes = TB_GET_PROMOTES(res);
         Piece pieceType = NO_PIECE;
         if (promotes != TB_PROMOTES_NONE)
         {
            switch (promotes)
            {
               case TB_PROMOTES_QUEEN: pieceType = (Piece)QUEEN; break;
               case TB_PROMOTES_ROOK: pieceType = (Piece)ROOK; break;
               case TB_PROMOTES_BISHOP: pieceType = (Piece)BISHOP; break;
               case TB_PROMOTES_KNIGHT: pieceType = (Piece)KNIGHT; break;
            }
         }
         *move = getPackedMove(from, to, pieceType);
      }
   }

   int dtz = TB_GET_DTZ(res);
   int wdl = TB_GET_WDL(res);

   // Return a score based on WDL and DTZ
   if (wdl == TB_WIN) return -VALUE_MATED - MAX_DEPTH - 100 - dtz;
   if (wdl == TB_CURSED_WIN) return 100;
   if (wdl == TB_DRAW) return 0;
   if (wdl == TB_BLESSED_LOSS) return -100;
   if (wdl == TB_LOSS) return VALUE_MATED + MAX_DEPTH + 100 + dtz;

   return TABLEBASE_ERROR;
}

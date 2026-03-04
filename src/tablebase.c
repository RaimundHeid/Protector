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

#include "tablebase.h"
#include "protector.h"
#include "io.h"
#include "tbprobe.h"
#include <assert.h>

#ifdef INCLUDE_TABLEBASE_ACCESS

bool tbAvailable = FALSE;

int setTablebaseCacheSize(unsigned int size)
{
   // Fathom handles its own caching or uses system file cache.
   // Nalimov cache size setting is ignored for Syzygy.
   return 0;
}

int initializeTablebase(const char *path)
{
   if (tb_init(path) != FALSE)
   {
      tbAvailable = (TB_LARGEST > 0);
      return 0;
   }
   return -1;
}

void closeTablebaseFiles(void)
{
   tb_free();
   tbAvailable = FALSE;
}

int probeTablebase(const Position * position)
{
   if (!tbAvailable) return TABLEBASE_ERROR;

   unsigned res = tb_probe_wdl(
      position->piecesOfColor[WHITE],
      position->piecesOfColor[BLACK],
      position->piecesOfType[WHITE_KING] | position->piecesOfType[BLACK_KING],
      position->piecesOfType[WHITE_QUEEN] | position->piecesOfType[BLACK_QUEEN],
      position->piecesOfType[WHITE_ROOK] | position->piecesOfType[BLACK_ROOK],
      position->piecesOfType[WHITE_BISHOP] | position->piecesOfType[BLACK_BISHOP],
      position->piecesOfType[WHITE_KNIGHT] | position->piecesOfType[BLACK_KNIGHT],
      position->piecesOfType[WHITE_PAWN] | position->piecesOfType[BLACK_PAWN],
      0, // rule50
      0, // castling
      (position->enPassantSquare == NO_SQUARE ? 0 : position->enPassantSquare),
      (position->activeColor == WHITE)
   );

   if (res == TB_RESULT_FAILED) return TABLEBASE_ERROR;

   switch (res)
   {
      case TB_WIN:
         return -VALUE_MATED - 1; // Win
      case TB_LOSS:
         return VALUE_MATED + 1;  // Loss
      case TB_DRAW:
      case TB_BLESSED_LOSS:
      case TB_CURSED_WIN:
         return 0;                // Drawish
      default:
         return TABLEBASE_ERROR;
   }
}

int initializeModuleTablebase(void)
{
   if (commandlineOptions.tablebasePath != 0)
   {
      return initializeTablebase(commandlineOptions.tablebasePath);
   }
   else
   {
      return 0;
   }
}

int testModuleTablebase(void)
{
   // TODO: Implement Syzygy-specific tests if needed.
   return 0;
}

#endif

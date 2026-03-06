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

#include "fen.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <assert.h>

static int initialized = 0;
static Piece pieceCode[256];
static char pieceToken[16];

static void initialize(void)
{
   if (!initialized)
   {
      int i;

      for (i = 0; i < 256; i++)
      {
         pieceCode[i] = NO_PIECE;
      }

      pieceCode['K'] = WHITE_KING;
      pieceCode['Q'] = WHITE_QUEEN;
      pieceCode['R'] = WHITE_ROOK;
      pieceCode['B'] = WHITE_BISHOP;
      pieceCode['N'] = WHITE_KNIGHT;
      pieceCode['P'] = WHITE_PAWN;
      pieceCode['k'] = BLACK_KING;
      pieceCode['q'] = BLACK_QUEEN;
      pieceCode['r'] = BLACK_ROOK;
      pieceCode['b'] = BLACK_BISHOP;
      pieceCode['n'] = BLACK_KNIGHT;
      pieceCode['p'] = BLACK_PAWN;

      pieceToken[WHITE_KING] = 'K';
      pieceToken[WHITE_QUEEN] = 'Q';
      pieceToken[WHITE_ROOK] = 'R';
      pieceToken[WHITE_BISHOP] = 'B';
      pieceToken[WHITE_KNIGHT] = 'N';
      pieceToken[WHITE_PAWN] = 'P';
      pieceToken[BLACK_KING] = 'k';
      pieceToken[BLACK_QUEEN] = 'q';
      pieceToken[BLACK_ROOK] = 'r';
      pieceToken[BLACK_BISHOP] = 'b';
      pieceToken[BLACK_KNIGHT] = 'n';
      pieceToken[BLACK_PAWN] = 'p';

      initialized = 1;
   }
}

static int setPieces(const char *fen, Position * position)
{
   Square square;
   Rank rank = RANK_8;
   File file = FILE_A;
   Piece p;
   int index = 0;
   int imax = (int) (strlen(fen)) - 1;
   char current = '\0';

   ITERATE(square)
   {
      position->piece[square] = NO_PIECE;
   }

   while (index <= imax && (current = fen[index]) != ' ')
   {
      if (current == '/')
      {
         if (rank == RANK_1) return -1;
         rank--;
         file = FILE_A;
      }
      else if (current >= '1' && current <= '8')
      {
         int emptySquares = current - '0';
         if (file + emptySquares > 8) return -1;
         file = (File) (file + emptySquares);
      }
      else
      {
         p = pieceCode[(unsigned char) current];

         if (p != NO_PIECE)
         {
            if (file > FILE_H || rank < RANK_1) return -1;
            position->piece[getSquare(file, rank)] = p;
            file++;
         }
         else
         {
            return -1;
         }
      }

      index++;
   }

   return (rank == RANK_1 && index <= imax + 1) ? index : -1;
}

static int stringContainsChar(const char *string, char c)
{
   while (*string != '\0')
   {
      if (*(string++) == c)
      {
         return 1;
      }
   }

   return 0;
}

int readFen(const char *fen, Position * position)
{
   int index;
   size_t fenLength = strlen(fen);

   /*
    * Initialize this module.
    */
   initialize();

   /*
    * Get the piece positions.
    */
   if ((index = setPieces(fen, position)) < 0)
   {
      return -1;
   }

   /*
    * Get the active color.
    */
   while (index < fenLength && fen[index] == ' ')
   {
      index++;
   }

   if (index < fenLength && stringContainsChar("bw", fen[index]))
   {
      position->activeColor = (fen[index] == 'w' ? WHITE : BLACK);
      index++;
   }
   else
   {
      return -1;                /* fen format error */
   }

   /*
    * Get the castling rights.
    */
   while (index < fenLength && fen[index] == ' ')
   {
      index++;
   }

   position->castlingRights = NO_CASTLINGS;

   if (index < fenLength && fen[index] == '-')
   {
      index++;
   }
   else
   {
      while (index < fenLength && fen[index] != ' ')
      {
         switch (fen[index++])
         {
         case 'K':
            position->castlingRights |= WHITE_00;
            break;
         case 'Q':
            position->castlingRights |= WHITE_000;
            break;
         case 'k':
            position->castlingRights |= BLACK_00;
            break;
         case 'q':
            position->castlingRights |= BLACK_000;
            break;
         default:
            return -1;             /* fen format error */
         }
      }
   }

   /*
    * Get the en passant square.
    */
   while (index < fenLength && fen[index] == ' ')
   {
      index++;
   }

   if (index < fenLength && fen[index] == '-')
   {
      position->enPassantSquare = NO_SQUARE;
      index++;
   }
   else if (index + 1 < fenLength && fen[index] >= 'a' && fen[index] <= 'h' &&
            (fen[index + 1] == '3' || fen[index + 1] == '6'))
   {
      File file = (File) (fen[index++] - 'a');
      Rank rank = (Rank) (fen[index++] - '1');

      position->enPassantSquare = getSquare(file, rank);
   }
   else
   {
      return -1;                /* fen format error */
   }

   /*
    * Get the half move clock value.
    */
   while (index < fenLength && fen[index] == ' ')
   {
      index++;
   }

   if (index < fenLength)
   {
      char *endptr;
      position->halfMoveClock = (int)strtol(fen + index, &endptr, 10);
      index = (int)(endptr - fen);
   }

   /*
    * Get the move number value.
    */
   while (index < fenLength && fen[index] == ' ')
   {
      index++;
   }

   if (index < fenLength)
   {
      char *endptr;
      position->moveNumber = (int)strtol(fen + index, &endptr, 10);
   }

   return 0;
}

void getFen(const Position * position, char *fen, size_t bufferSize)
{
   int rank, file;
   size_t p = 0;

   /*
    * Initialize this module.
    */
   initialize();

   for (rank = RANK_8; rank >= RANK_1; rank--)
   {
      int emptyCount = 0;

      for (file = FILE_A; file <= FILE_H; file++)
      {
         Square square = getSquare(file, rank);

         if (position->piece[square] != NO_PIECE)
         {
            if (emptyCount > 0)
            {
               if (p < bufferSize - 1) fen[p++] = (char) ('0' + emptyCount);
               emptyCount = 0;
            }

            if (p < bufferSize - 1) fen[p++] = pieceToken[position->piece[square]];
         }
         else
         {
            emptyCount++;
         }
      }

      if (emptyCount > 0)
      {
         if (p < bufferSize - 1) fen[p++] = (char) ('0' + emptyCount);
      }

      if (rank > RANK_1)
      {
         if (p < bufferSize - 1) fen[p++] = '/';
      }
   }

   if (p < bufferSize - 1) fen[p++] = ' ';
   if (p < bufferSize - 1) fen[p++] = (position->activeColor == WHITE ? 'w' : 'b');
   if (p < bufferSize - 1) fen[p++] = ' ';

   if (position->castlingRights)
   {
      if (position->castlingRights & WHITE_00)
      {
         if (p < bufferSize - 1) fen[p++] = 'K';
      }

      if (position->castlingRights & WHITE_000)
      {
         if (p < bufferSize - 1) fen[p++] = 'Q';
      }

      if (position->castlingRights & BLACK_00)
      {
         if (p < bufferSize - 1) fen[p++] = 'k';
      }

      if (position->castlingRights & BLACK_000)
      {
         if (p < bufferSize - 1) fen[p++] = 'q';
      }
   }
   else
   {
      if (p < bufferSize - 1) fen[p++] = '-';
   }

   if (p < bufferSize - 1) fen[p++] = ' ';

   if (position->enPassantSquare != NO_SQUARE)
   {
      char f = (char) (file(position->enPassantSquare) + 'a');
      char r = (char) (rank(position->enPassantSquare) + '1');

      if (p < bufferSize - 1) fen[p++] = f;
      if (p < bufferSize - 1) fen[p++] = r;
   }
   else
   {
      if (p < bufferSize - 1) fen[p++] = '-';
   }

   fen[p] = '\0';
   
   char buffer[32];
   snprintf(buffer, sizeof(buffer), " %i %i", position->halfMoveClock, position->moveNumber);
   strncat(fen, buffer, bufferSize - strlen(fen) - 1);
}

int initializeModuleFen(void)
{
   return 0;
}

int testModuleFen(void)
{
   char *fen1 = FEN_GAMESTART;
   char *fen2 =
      "rn3rk1/pbppq1pp/1p2pb2/4N2Q/3PN3/3B4/PPP2PPP/R3K2R w KQ - 4 11";
   char *fen3 = "8/8/1R5p/q5pk/PR3pP1/7P/8/7K b - g3 0 1";
   char buffer[256];
   Position position;
   Square square;

   readFen(fen1, &position);
   getFen(&position, buffer, sizeof(buffer));
   assert(strcmp(buffer, fen1) == 0);

   assert(position.activeColor == WHITE);
   assert(position.enPassantSquare == NO_SQUARE);
   assert(position.halfMoveClock == 0);
   assert(position.moveNumber == 1);
   assert(position.castlingRights ==
          (WHITE_00 | WHITE_000 | BLACK_00 | BLACK_000));
   assert(position.piece[A1] == WHITE_ROOK);
   assert(position.piece[H1] == WHITE_ROOK);
   assert(position.piece[B1] == WHITE_KNIGHT);
   assert(position.piece[G1] == WHITE_KNIGHT);
   assert(position.piece[C1] == WHITE_BISHOP);
   assert(position.piece[F1] == WHITE_BISHOP);
   assert(position.piece[D1] == WHITE_QUEEN);
   assert(position.piece[E1] == WHITE_KING);

   assert(position.piece[A8] == BLACK_ROOK);
   assert(position.piece[H8] == BLACK_ROOK);
   assert(position.piece[B8] == BLACK_KNIGHT);
   assert(position.piece[G8] == BLACK_KNIGHT);
   assert(position.piece[C8] == BLACK_BISHOP);
   assert(position.piece[F8] == BLACK_BISHOP);
   assert(position.piece[D8] == BLACK_QUEEN);
   assert(position.piece[E8] == BLACK_KING);

   for (square = A2; square <= H2; square++)
   {
      assert(position.piece[square] == WHITE_PAWN);
      assert(position.piece[square + 8] == NO_PIECE);
      assert(position.piece[square + 16] == NO_PIECE);
      assert(position.piece[square + 24] == NO_PIECE);
      assert(position.piece[square + 32] == NO_PIECE);
      assert(position.piece[square + 40] == BLACK_PAWN);
   }

   readFen(fen2, &position);
   getFen(&position, buffer, sizeof(buffer));
   assert(strcmp(buffer, fen2) == 0);

   readFen(fen3, &position);
   getFen(&position, buffer, sizeof(buffer));
   assert(strcmp(buffer, fen3) == 0);

   return 0;
}


#ifndef POSITION_STRUCT_H
#define POSITION_STRUCT_H

#include "protector.h"
#include "bitboard.h"

typedef struct
{
   Piece piece[_64_];
   Color activeColor;
   BYTE castlingRights;
   Square enPassantSquare;
   int moveNumber, halfMoveClock;

   /**
    * Redundant data
    */
   Bitboard allPieces;
   Bitboard piecesOfColor[2];
   Bitboard piecesOfType[16];
   Square king[2];
   int numberOfPieces[2];
   int numberOfPawns[2];
   UINT64 pieceCount;
   INT32 balance;
   UINT64 hashKey, pawnHashKey;
}
Position;

#endif

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

#ifndef _evaluation_h_
#define _evaluation_h_

#include "position.h"
#include "bitboard.h"
#include "keytable.h"
#include "io.h"

#define MATERIALINFO_TABLE_SIZE ( 648 * 648 )
extern MaterialInfo materialInfo[MATERIALINFO_TABLE_SIZE];
extern Bitboard passedPawnCorridor[2][_64_];
extern Bitboard candidateDefenders[2][_64_];

INT32 materialBalance(const Position * position);
bool hasWinningPotential(Position * position, Color color);
bool hasBishopPair(const Position * position, const Color color);

#include "nnue.h"

/**
 * Calculate the value of the specified position.
 *
 * @return the value of the specified position
 */
int getValue(const Position * position, Accumulator * acc);

/**
 * Check if the pawn at the specified square is a passed pawn.
 */
bool pawnIsPassed(const Position * position, const Square pawnSquare,
                  const Color pawnColor);

/**
 * Flip the given position and check if it yields the same result.
 *
 * @return FALSE if the flipped position yields a diffent result
 */
bool flipTest(Position * position);

/**
 * Initialize this module.
 *
 * @return 0 if no errors occurred.
 */
int initializeModuleEvaluation(void);

/**
 * Test this module.
 *
 * @return 0 if all tests succeed.
 */
int testModuleEvaluation(void);

#endif

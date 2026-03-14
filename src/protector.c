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
#include <assert.h>
#include "tools.h"
#include "io.h"
#include "protector.h"
#include "bitboard.h"
#include "position.h"
#include "fen.h"
#include "movegeneration.h"
#include "matesearch.h"
#include "search.h"
#include "hash.h"
#include "test.h"
#include "pgn.h"
#include "evaluation.h"
#include "coordination.h"
#include "uci.h"
#include "tablebase.h"
#include "nnue.h"

const char *programVersionNumber = "2.0.0";

int _distance[_64_][_64_];
int _horizontalDistance[_64_][_64_];
int _verticalDistance[_64_][_64_];
int _taxiDistance[_64_][_64_];
int castlingsOfColor[2];
const int colorSign[2] = { 1, -1 };

CommandlineOptions commandlineOptions;
UINT64 statCount1, statCount2;
int debugOutput = FALSE;

static int initializeModuleProtector(void)
{
   int sq1, sq2;

   ITERATE(sq1)
   {
      ITERATE(sq2)
      {
         _horizontalDistance[sq1][sq2] =
            max(file(sq1), file(sq2)) - min(file(sq1), file(sq2));
         _verticalDistance[sq1][sq2] =
            max(rank(sq1), rank(sq2)) - min(rank(sq1), rank(sq2));
         _distance[sq1][sq2] =
            max(_horizontalDistance[sq1][sq2], _verticalDistance[sq1][sq2]);
         _taxiDistance[sq1][sq2] =
            _horizontalDistance[sq1][sq2] + _verticalDistance[sq1][sq2];
      }
   }

   castlingsOfColor[WHITE] = WHITE_00 | WHITE_000;
   castlingsOfColor[BLACK] = BLACK_00 | BLACK_000;

   if (initializeModuleIo() != 0) return -1;
   if (initializeModuleTools() != 0) return -1;
   if (initializeModuleCoordination() != 0) return -1;
   if (initializeModuleBitboard() != 0) return -1;
   if (initializeModulePosition() != 0) return -1;
   if (initializeModuleFen() != 0) return -1;
   if (initializeModuleMovegeneration() != 0) return -1;
   if (initializeModuleMatesearch() != 0) return -1;
   if (initializeModuleSearch() != 0) return -1;
   if (initializeModuleHash() != 0) return -1;
   if (initializeModuleTest() != 0) return -1;
   if (initializeModulePgn() != 0) return -1;
   if (initializeModuleTablebase() != 0) return -1;
   if (initializeModuleEvaluation() != 0) return -1;
   if (initializeModuleNnue() != 0) return -1;
   if (initializeModuleUci() != 0) return -1;

   return 0;
}

static void reportSuccess(const char *moduleName)
{
   logDebug("Module %s tested successfully.\n", moduleName);
}

static int testModuleProtector(void)
{
   assert(_horizontalDistance[C2][E6] == 2);
   assert(_verticalDistance[C2][E6] == 4);
   assert(_distance[C3][H8] == 5);
   assert(_taxiDistance[B2][F7] == 9);

   if (testModuleIo() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Io");
   }

   if (testModuleTools() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Tools");
   }

   if (testModuleCoordination() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Coordination");
   }

   if (testModuleBitboard() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Bitboard");
   }

   if (testModulePosition() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Position");
   }

   if (testModuleFen() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Fen");
   }

   if (testModuleMovegeneration() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Movegeneration");
   }

   if (testModuleHash() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Hash");
   }

   if (testModuleMatesearch() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Matesearch");
   }

   if (testModuleSearch() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Search");
   }

   if (testModulePgn() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Pgn");
   }

   if (testModuleEvaluation() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Evaluation");
   }

   if (testModuleUci() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Uci");
   }

   if (testModuleTablebase() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Tablebase");
   }

   if (testModuleNnue() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Nnue");
   }

   if (testModuleTest() != 0)
   {
      return -1;
   }
   else
   {
      reportSuccess("Test");
   }

   logDebug("\nModuletest finished successfully.\n");

   return 0;
}

static void parseOptions(int argc, char **argv, CommandlineOptions * options)
{
   int i;

   options->processModuleTest = FALSE;
   options->uciMode = TRUE;
   options->dumpEvaluation = FALSE;
   options->testfile = NULL;
   options->tablebasePath = NULL;

   for (i = 0; i < argc; i++)
   {
      const char *currentArg = argv[i];

      if (strcmp(currentArg, "-m") == 0)
      {
         options->processModuleTest = TRUE;
         options->uciMode = FALSE;
      }

      if (strcmp(currentArg, "-d") == 0)
      {
         options->dumpEvaluation = TRUE;
      }

      if (strcmp(currentArg, "-t") == 0 && i < argc - 1)
      {
         options->testfile = argv[++i];
         options->uciMode = FALSE;
      }

      if (strcmp(currentArg, "-e") == 0 && i < argc - 1)
      {
         options->tablebasePath = argv[++i];
      }

      if (strcmp(currentArg, "-v") == 0)
      {
         printf("Protector %s\n", programVersionNumber);
      }
   }
}

int main(int argc, char **argv)
{
   parseOptions(argc, argv, &commandlineOptions);

   if (initializeModuleProtector() != 0)
   {
      logDebug("Initialization failed. Terminating.\n");
      finalizeModuleCoordination();

      return -1;
   }

   if (commandlineOptions.uciMode)
   {
      acceptGuiCommands();
   }
   else if (commandlineOptions.processModuleTest)
   {
      if (testModuleProtector() != 0)
      {
         logDebug("\n##### Moduletest failed! #####\n");

         finalizeModuleCoordination();

         return -1;
      }
   }

   if (commandlineOptions.testfile != NULL)
   {
      if (processTestsuite(commandlineOptions.testfile) != 0)
      {
         finalizeModuleCoordination();

         return -1;
      }
   }

   logDebug("Main thread terminated.\n");
   finalizeModuleCoordination();

   return 0;
}

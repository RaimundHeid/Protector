
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test.h"
#include "io.h"
#include "pgn.h"
#include "coordination.h"
#include "evaluation.h"

extern bool resetSharedHashtable;

static void handleSearchEvent(int eventId, void *var)
{
   Variation *variation = (Variation *) var;
   long time;
   char *pvMoves;

   switch (eventId)
   {
   case SEARCHEVENT_SEARCH_FINISHED:
      time = variation->finishTimeProcess - variation->startTimeProcess;

      logReport("%lld nodes in %ld msec\n", getNodeCount(), time);

      if (time > 1000)
      {
         logReport("%ld knps\n", getNodeCount() / max(1, time));
      }

      break;

   case SEARCHEVENT_PLY_FINISHED:
   case SEARCHEVENT_NEW_PV:
      pvMoves = getPrincipalVariation(var);
      dumpPv(variation->iteration,
             (getTimestamp() - variation->startTime), pvMoves,
             variation->pv[0].score, variation->nodes,
             variation->startPosition.activeColor);
      free(pvMoves);
      break;

   default:
      break;
   }
}

static bool solveMateProblem(SearchTask * entry)
{
   bool result = TRUE;
   int i;

   completeTask(entry);

   if (entry->solutions.numberOfMoves !=
       entry->calculatedSolutions.numberOfMoves)
   {
      result = FALSE;
   }

   for (i = 0; i < entry->solutions.numberOfMoves; i++)
   {
      if (listContainsMove
          (&entry->calculatedSolutions, entry->solutions.moves[i]) == FALSE)
      {
         result = FALSE;
      }
   }

   return result;
}

static bool solveBestMoveProblem(SearchTask * entry)
{
   completeTask(entry);

   return listContainsMove(&entry->solutions, entry->bestMove);
}

static bool dumpEvaluation(SearchTask * entry)
{
   prepareSearch(entry->variation);
   getValue(&entry->variation->startPosition,
            &entry->variation->plyInfo[0].accumulator);

   return TRUE;
}

int processTestsuite(const char *filename)
{
   PGNFile pgnfile;
   PGNGame *game;
   SearchTask entry;
   Gamemove *gamemove;
   long i;
   UINT64 overallNodes = 0;
   const char *fmt = "\nTestsuite '%s': %d/%d solved, %s nodes\n";
   char ons[32];
   int solved = 0;
   String notSolved = getEmptyString();
   Variation variation;

   if (openPGNFile(&pgnfile, filename) != 0)
   {
      return -1;
   }

   logReport("\nProcessing file '%s' [%ld game(s)]\n", filename,
             pgnfile.numGames);

   statCount1 = statCount2 = 0;
   variation.timeTarget = 60 * 1000;
   variation.timeLimit = 60 * 1000;
   variation.ponderMode = FALSE;
   entry.variation = &variation;

   for (i = 1; i <= pgnfile.numGames; i++)
   {
      game = getGame(&pgnfile, i);

      if (game == 0)
      {
         continue;
      }

      logReport("\n%ld (%ld): %s-%s\n", i, pgnfile.numGames,
                game->white, game->black);
      logPosition(&game->firstMove->position);

      entry.solutions.numberOfMoves = 0;
      gamemove = game->firstMove;

      while (gamemove != 0 &&
             entry.solutions.numberOfMoves < MAX_MOVES_PER_POSITION)
      {
         entry.solutions.moves
            [entry.solutions.numberOfMoves++] =
            getPackedMove(gamemove->from, gamemove->to, gamemove->newPiece);

         gamemove = gamemove->alternativeMove;
      }

      initializeVariation(&variation, game->fen);
      resetSharedHashtable = TRUE;
      variation.handleSearchEvent = &handleSearchEvent;

      char *mateMarker = strstr(game->white, "[#");
      if (mateMarker != NULL)
      {
         entry.type = TASKTYPE_TEST_MATE_IN_N;
         entry.numberOfMoves = atoi(mateMarker + 2);

         if (solveMateProblem(&entry) != FALSE)
         {
            solved++;
         }
         else
         {
            logReport("##### Mate problem %ld NOT solved! #####\n", i);
            appendToString(&notSolved, "%d ", i);
         }

         overallNodes += entry.nodes;
      }
      else
      {
         entry.type = TASKTYPE_TEST_BEST_MOVE;
         entry.numberOfMoves = 0;

         if (commandlineOptions.dumpEvaluation)
         {
            dumpEvaluation(&entry);
         }
         else
         {
            if (solveBestMoveProblem(&entry) != FALSE)
            {
               solved++;
            }
            else
            {
               appendToString(&notSolved, "%d ", i);
            }

            overallNodes += entry.nodes;
         }
      }

      freePgnGame(game);
   }

   formatLongInteger(overallNodes, ons, sizeof(ons));
   logReport(fmt, filename, solved, pgnfile.numGames, ons);
   logReport("Not solved: %s\n", notSolved.buffer);
   deleteString(&notSolved);
   closePGNFile(&pgnfile);

   return 0;
}

/**
 * Initialize this module.
 *
 * @return 0 if no errors occurred.
 */
int initializeModuleTest(void)
{
   return 0;
}


#include "nnue.h"
#include "fen.h"

static int testNnuePlausibility(void) {
    typedef struct {
        const char* fen;
        const char* description;
        int min_eval;
        int max_eval;
    } NnueTestCase;

    NnueTestCase cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Startpos", -50, 50},
        {"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Spanish Opening", -100, 100},
        {"r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "Sicilian Defense", -100, 100},
        {"r1b2rk1/pp1nbppp/2p1pn2/q2p2B1/2PP4/2N1PN2/PPQ2PPP/2R1KB1R w K - 3 9", "QGD Carlsbad (White)", -100, 200},
        {"r4rk1/pp3ppp/2pbbn2/3p4/3P4/2N1PN2/PPQ1BPPP/R4RK1 b - - 5 12", "Equal Middle game (Black)", -500, 300},
        {"r3k2r/pppb1ppp/2n1pn2/8/2PP4/2N2N2/PP2BPPP/R2QK2R w KQkq - 0 1", "White advantage (White)", -100, 800},
        {"2r2rk1/1p1q1ppp/p1p1p3/3p4/2PP4/PP1QP3/5PPP/2R2RK1 b - - 0 1", "Middle heavy (Black)", -100, 100},
        {"8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1", "Endgame Drawn (White)", -50, 50},
        {"8/8/4k3/3p1P2/3P4/4K3/8/8 b - - 0 1", "Endgame White Winning (Black)", -500, -50},
        {"8/8/8/8/8/2k5/2r5/1K1Q4 w - - 0 1", "Queen vs Rook (White)", 20, 1000}
    };

    int result = 0;
    for (int i = 0; i < 10; i++) {
        Variation variation;
        initializeVariation(&variation, cases[i].fen);
        refreshAccumulator(&variation.singlePosition, &variation.plyInfo[variation.ply].accumulator);
        int eval = evaluateNnueWithAccumulator(&variation.singlePosition, &variation.plyInfo[variation.ply].accumulator);
        logDebug("Nnue Test Case %d (%s): eval %d (expected [%d, %d])\n", i, cases[i].description, eval, cases[i].min_eval, cases[i].max_eval);
        if (eval < cases[i].min_eval || eval > cases[i].max_eval) {
            logDebug("Nnue Plausibility failed for case %d: %d not in [%d, %d]\n", i, eval, cases[i].min_eval, cases[i].max_eval);
            result = -1;
        }
    }
    return result;
}

static int testBigNnuePlausibility(void) {
    typedef struct {
        const char* fen;
        const char* description;
        int min_eval;
        int max_eval;
    } NnueTestCase;

    NnueTestCase cases[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Startpos", -100, 100},
        {"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Spanish Opening", -200, 200},
        {"r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "Sicilian Defense", -200, 200},
        {"r1b2rk1/pp1nbppp/2p1pn2/q2p2B1/2PP4/2N1PN2/PPQ2PPP/2R1KB1R w K - 3 9", "QGD Carlsbad (White)", -200, 300},
        {"r4rk1/pp3ppp/2pbbn2/3p4/3P4/2N1PN2/PPQ1BPPP/R4RK1 b - - 5 12", "Equal Middle game (Black)", -600, 400},
        {"r3k2r/pppb1ppp/2n1pn2/8/2PP4/2N2N2/PP2BPPP/R2QK2R w KQkq - 0 1", "White advantage (White)", -200, 1000},
        {"2r2rk1/1p1q1ppp/p1p1p3/3p4/2PP4/PP1QP3/5PPP/2R2RK1 b - - 0 1", "Middle heavy (Black)", -200, 200},
        {"8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1", "Endgame Drawn (White)", -100, 100},
        {"8/8/4k3/3p1P2/3P4/4K3/8/8 b - - 0 1", "Endgame White Winning (Black)", -800, -50},
        {"8/8/8/8/8/2k5/2r5/1K1Q4 w - - 0 1", "Queen vs Rook (White)", 20, 2000}
    };

    int result = 0;
    for (int i = 0; i < 10; i++) {
        Variation variation;
        initializeVariation(&variation, cases[i].fen);
        refreshAccumulator(&variation.singlePosition, &variation.plyInfo[variation.ply].accumulator);
        int eval = evaluateBigNnueWithAccumulator(&variation.singlePosition, &variation.plyInfo[variation.ply].accumulator);
        logDebug("Big Nnue Test Case %d (%s): eval %d (expected [%d, %d])\n", i, cases[i].description, eval, cases[i].min_eval, cases[i].max_eval);
        if (eval < cases[i].min_eval || eval > cases[i].max_eval) {
            logDebug("Big Nnue Plausibility failed for case %d: %d not in [%d, %d]\n", i, eval, cases[i].min_eval, cases[i].max_eval);
            result = -1;
        }
    }
    return result;
}

int testModuleNnue(void)
{
   Variation variation;
   initializeVariation(&variation, FEN_GAMESTART);
   
   // Make some moves and check accumulator consistency
   Move moves[] = {
       getOrdinaryMove(E2, E4),
       getOrdinaryMove(E7, E5),
       getOrdinaryMove(G1, F3),
       getOrdinaryMove(B8, C6),
       getOrdinaryMove(F1, B5),
       getOrdinaryMove(A7, A6),
       getOrdinaryMove(B5, C6) // Capture
   };
   
   for (int i = 0; i < 7; i++) {
       makeMove(&variation, moves[i]);
       
       Accumulator refreshed;
       refreshAccumulator(&variation.singlePosition, &refreshed);
       
       int eval = evaluateNnueWithAccumulator(&variation.singlePosition, &variation.plyInfo[variation.ply].accumulator);
       logDebug("Ply %d eval: %d\n", variation.ply, eval);

       Accumulator *current = &variation.plyInfo[variation.ply].accumulator;
       for (int p = 0; p < 2; p++) {
           for (int j = 0; j < L1_SMALL; j++) {
               if (current->small_v[p][j] != refreshed.small_v[p][j]) {
                   logDebug("Small Accumulator inconsistency at ply %d, perspective %d, index %d: %d != %d\n", 
                            variation.ply, p, j, current->small_v[p][j], refreshed.small_v[p][j]);
                   return -1;
               }
           }
           for (int j = 0; j < L1_BIG; j++) {
               if (current->big_v[p][j] != refreshed.big_v[p][j]) {
                   logDebug("Big Accumulator inconsistency at ply %d, perspective %d, index %d: %d != %d\n", 
                            variation.ply, p, j, current->big_v[p][j], refreshed.big_v[p][j]);
                   return -1;
               }
           }
           for (int j = 0; j < 8; j++) {
               if (current->small_psqtAccumulation[p][j] != refreshed.small_psqtAccumulation[p][j]) {
                   logDebug("Small PSQT Accumulator inconsistency at ply %d, perspective %d, bucket %d: %d != %d\n", 
                            variation.ply, p, j, current->small_psqtAccumulation[p][j], refreshed.small_psqtAccumulation[p][j]);
                   return -1;
               }
               if (current->big_psqtAccumulation[p][j] != refreshed.big_psqtAccumulation[p][j]) {
                   logDebug("Big PSQT Accumulator inconsistency at ply %d, perspective %d, bucket %d: %d != %d\n", 
                            variation.ply, p, j, current->big_psqtAccumulation[p][j], refreshed.big_psqtAccumulation[p][j]);
                   return -1;
               }
           }
       }
   }
   
   // Unmake moves and check consistency
   while (variation.ply > 0) {
       unmakeLastMove(&variation);
       Accumulator refreshed;
       refreshAccumulator(&variation.singlePosition, &refreshed);
       Accumulator *current = &variation.plyInfo[variation.ply].accumulator;
       for (int p = 0; p < 2; p++) {
           for (int j = 0; j < L1_SMALL; j++) {
               if (current->small_v[p][j] != refreshed.small_v[p][j]) {
                   logDebug("Small Accumulator inconsistency after unmake at ply %d, perspective %d, index %d: %d != %d\n", 
                            variation.ply, p, j, current->small_v[p][j], refreshed.small_v[p][j]);
                   return -1;
               }
           }
           for (int j = 0; j < L1_BIG; j++) {
               if (current->big_v[p][j] != refreshed.big_v[p][j]) {
                   logDebug("Big Accumulator inconsistency after unmake at ply %d, perspective %d, index %d: %d != %d\n", 
                            variation.ply, p, j, current->big_v[p][j], refreshed.big_v[p][j]);
                   return -1;
               }
           }
           for (int j = 0; j < 8; j++) {
               if (current->small_psqtAccumulation[p][j] != refreshed.small_psqtAccumulation[p][j]) {
                   logDebug("Small PSQT Accumulator inconsistency after unmake at ply %d, perspective %d, bucket %d: %d != %d\n", 
                            variation.ply, p, j, current->small_psqtAccumulation[p][j], refreshed.small_psqtAccumulation[p][j]);
                   return -1;
               }
               if (current->big_psqtAccumulation[p][j] != refreshed.big_psqtAccumulation[p][j]) {
                   logDebug("Big PSQT Accumulator inconsistency after unmake at ply %d, perspective %d, bucket %d: %d != %d\n", 
                            variation.ply, p, j, current->big_psqtAccumulation[p][j], refreshed.big_psqtAccumulation[p][j]);
                   return -1;
               }
           }
       }
   }

   int res = testNnuePlausibility();
   if (res != 0) return res;

   res = testBigNnuePlausibility();
   if (res != 0) return res;

   typedef struct {
       const char* fen;
       const char* description;
       int min_eval;
       int max_eval;
   } ValueTestCase;

   ValueTestCase cases[] = {
       {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Startpos White", -10, 80},
       {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "Startpos Black", -10, 80},
       {"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Spanish Opening White", -10, 100},
       {"r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 3", "Spanish Opening Black", -10, 100},
       {"r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 1 2", "Sicilian Defense White", -10, 100},
       {"r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "Sicilian Defense Black", -10, 100},
       {"r1b2rk1/pp1nbppp/2p1pn2/q2p2B1/2PP4/2N1PN2/PPQ2PPP/2R1KB1R w K - 3 9", "QGD Carlsbad White", -50, 200},
       {"r1b2rk1/pp1nbppp/2p1pn2/q2p2B1/2PP4/2N1PN2/PPQ2PPP/2R1KB1R b K - 3 9", "QGD Carlsbad Black", -50, 200},
       {"8/8/4k3/3p4/3P4/4K3/8/8 w - - 0 1", "Endgame Drawn White", -50, 50},
       {"8/8/4k3/3p4/3P4/4K3/8/8 b - - 0 1", "Endgame Drawn Black", -50, 50},
       {"8/8/4k3/3p1P2/3P4/4K3/8/8 w - - 0 1", "Endgame White Winning White", 50, 500},
       {"8/8/4k3/3p1P2/3P4/4K3/8/8 b - - 0 1", "Endgame White Winning Black", -500, -50},
       {"8/8/8/8/8/2k5/2r5/1K1Q4 w - - 0 1", "Queen vs Rook White", 150, 1000},
       {"8/8/8/8/8/2k5/2r5/1K1Q4 b - - 0 1", "Queen vs Rook Black", -1000, -150}
   };

   for (int i = 0; i < 14; i++) {
       Variation v;
       initializeVariation(&v, cases[i].fen);
       int eval = getValue(&v.singlePosition, &v.plyInfo[v.ply].accumulator);
       logDebug("Value Test Case %d (%s): eval %d (expected [%d, %d])\n", i, cases[i].description, eval, cases[i].min_eval, cases[i].max_eval);
       if (eval < cases[i].min_eval || eval > cases[i].max_eval) {
           logReport("Value Plausibility failed for case %d (%s): %d not in [%d, %d]\n", i, cases[i].description, eval, cases[i].min_eval, cases[i].max_eval);
           return -1;
       }

       // Symmetry check: score(pos) should be equal to score(flipped_pos)
       // since getValue returns score relative to side to move.
       Position flipped;
       memcpy(&flipped, &v.singlePosition, sizeof(Position));
       flipPosition(&flipped);
       initializePosition(&flipped); // Update redundant data after flip
       int evalFlipped = getValue(&flipped, NULL);
       if (eval != evalFlipped) {
           logReport("Value Symmetry failed for case %d (%s): %d != %d\n", i, cases[i].description, eval, evalFlipped);
           return -1;
       }
   }

   return 0;
}

int testModuleTest(void)
{
   int result;

   if ((result = processTestsuite("moduletest.pgn")) != 0)
   {
      if (commandlineOptions.engineDirectory[0] != '\0')
      {
         char path[2048];
         snprintf(path, sizeof(path), "%s/%s", commandlineOptions.engineDirectory, "moduletest.pgn");
         result = processTestsuite(path);
      }
   }

   return result;
}

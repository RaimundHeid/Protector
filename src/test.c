
#include "test.h"

#include "coordination.h"
#include "evaluation.h"
#include "io.h"
#include "pgn.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern bool resetSharedHashtable;

static void handleSearchEvent(int eventId, void *var)
{
    Variation *variation = (Variation *)var;
    long time;
    char *pvMoves;

    switch (eventId) {
    case SEARCHEVENT_SEARCH_FINISHED:
        time = variation->finishTimeProcess - variation->startTimeProcess;

        logReport("%lld nodes in %ld msec\n", getNodeCount(), time);

        if (time > 1000) {
            logReport("%ld knps\n", getNodeCount() / max(1, time));
        }

        break;

    case SEARCHEVENT_PLY_FINISHED:
    case SEARCHEVENT_NEW_PV:
        pvMoves = getPrincipalVariation(var);
        dumpPv(variation->iteration, (getTimestamp() - variation->startTime), pvMoves, variation->pv[0].score,
               variation->nodes, variation->startPosition.activeColor);
        free(pvMoves);
        break;

    default:
        break;
    }
}

static bool solveMateProblem(SearchTask *entry)
{
    bool result = TRUE;
    int i;

    completeTask(entry);

    if (entry->solutions.numberOfMoves != entry->calculatedSolutions.numberOfMoves) {
        result = FALSE;
    }

    for (i = 0; i < entry->solutions.numberOfMoves; i++) {
        if (listContainsMove(&entry->calculatedSolutions, entry->solutions.moves[i]) == FALSE) {
            result = FALSE;
        }
    }

    return result;
}

static bool solveBestMoveProblem(SearchTask *entry)
{
    completeTask(entry);

    return listContainsMove(&entry->solutions, entry->bestMove);
}

static bool dumpEvaluation(SearchTask *entry)
{
    prepareSearch(entry->variation);
    getValue(&entry->variation->startPosition, &entry->variation->plyInfo[0].accumulator, 0);

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
    Variation *variation = calloc(1, sizeof(Variation));

    if (openPGNFile(&pgnfile, filename) != 0) {
        free(variation);
        return -1;
    }

    logReport("\nProcessing file '%s' [%ld game(s)]\n", filename, pgnfile.numGames);

    statCount1 = statCount2 = 0;
    variation->timeTarget = 60 * 1000;
    variation->timeLimit = 60 * 1000;
    variation->ponderMode = FALSE;
    entry.variation = variation;

    for (i = 1; i <= pgnfile.numGames; i++) {
        game = getGame(&pgnfile, i);

        if (game == 0) {
            continue;
        }

        logReport("\n%ld (%ld): %s-%s\n", i, pgnfile.numGames, game->white, game->black);

        initializeVariation(variation, game->fen);

        if (game->firstMove != NULL) {
            logPosition(&game->firstMove->position);
        } else {
            logPosition(&variation->singlePosition);
        }

        entry.solutions.numberOfMoves = 0;
        gamemove = game->firstMove;

        while (gamemove != 0 && entry.solutions.numberOfMoves < MAX_MOVES_PER_POSITION) {
            entry.solutions.moves[entry.solutions.numberOfMoves++] =
                getPackedMove(gamemove->from, gamemove->to, gamemove->newPiece);

            gamemove = gamemove->alternativeMove;
        }

        resetSharedHashtable = TRUE;
        variation->handleSearchEvent = &handleSearchEvent;

        char *mateMarker = strstr(game->white, "[#");
        if (mateMarker != NULL) {
            entry.type = TASKTYPE_TEST_MATE_IN_N;
            entry.numberOfMoves = atoi(mateMarker + 2);

            if (solveMateProblem(&entry) != FALSE) {
                solved++;
            } else {
                logReport("##### Mate problem %ld NOT solved! #####\n", i);
                appendToString(&notSolved, "%d ", i);
            }

            overallNodes += entry.nodes;
        } else {
            entry.type = TASKTYPE_TEST_BEST_MOVE;
            entry.numberOfMoves = 0;

            if (commandlineOptions.dumpEvaluation) {
                dumpEvaluation(&entry);
            } else {
                if (solveBestMoveProblem(&entry) != FALSE) {
                    solved++;
                } else {
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
    free(variation);

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

int testModuleTest(void)
{
    int result;

    if ((result = processTestsuite("moduletest.pgn")) != 0) {
        if (commandlineOptions.engineDirectory[0] != '\0') {
            char path[2048];
            snprintf(path, sizeof(path), "%s/%s", commandlineOptions.engineDirectory, "moduletest.pgn");
            result = processTestsuite(path);
        }
    }

    return result;
}

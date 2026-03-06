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

#include "coordination.h"
#include "protector.h"
#include "search.h"
#include "matesearch.h"
#include "io.h"
#include "hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <time.h>

/* #define DEBUG_COORDINATION */

#define SEARCH_THREAD_STACK_SIZE (2 * 1024 * 1024)
#define INITIAL_HASHTABLE_SIZE_MB 16
#define BYTES_PER_MB (1024ULL * 1024ULL)

static pthread_t searchThread[MAX_THREADS];
static bool searchThreadStarted[MAX_THREADS];
static pthread_t timer;
static bool timerStarted = FALSE;
static pthread_mutex_t guiSearchMutex = PTHREAD_MUTEX_INITIALIZER;

static int numThreads = 1;
static SearchTask dummyTask;
static SearchTask *currentTask = &dummyTask;
static Variation variations[MAX_THREADS];
static Hashtable sharedHashtable;
static PawnHashInfo pawnHashtable[MAX_THREADS][PAWN_HASHTABLE_SIZE];

Hashtable *getSharedHashtable(void)
{
   return &sharedHashtable;
}

int setNumberOfThreads(int _numThreads)
{
   numThreads = max(1, min(MAX_THREADS, _numThreads));

   return numThreads;
}

int getNumberOfThreads(void)
{
   return numThreads;
}

UINT64 getNodeCount(void)
{
   int threadCount;
   UINT64 totalNodes = 0;

   /* numThreads is only changed between searches, so reading it here is safe */
   for (threadCount = 0; threadCount < numThreads; threadCount++)
   {
      totalNodes += variations[threadCount].nodes;
   }

   return totalNodes;
}

Variation *getCurrentVariation(void)
{
   return &variations[0];
}

void getGuiSearchMutex(void)
{
#ifdef DEBUG_COORDINATION
   logDebug("aquiring search lock...\n");
#endif

   pthread_mutex_lock(&guiSearchMutex);

#ifdef DEBUG_COORDINATION
   logDebug("search lock aquired...\n");
#endif
}

void releaseGuiSearchMutex(void)
{
   pthread_mutex_unlock(&guiSearchMutex);

#ifdef DEBUG_COORDINATION
   logDebug("search lock released...\n");
#endif
}

static void performTaskSpecificSearch(Variation * currentVariation)
{
   switch (currentTask->type)
   {
   case TASKTYPE_BEST_MOVE:
      currentTask->bestMove = search(currentVariation, NULL);
      break;

   case TASKTYPE_TEST_BEST_MOVE:
      currentTask->bestMove =
         search(currentVariation, &currentTask->solutions);
      break;

   case TASKTYPE_MATE_IN_N:
      searchForMate(currentVariation,
                    &currentTask->calculatedSolutions,
                    currentTask->numberOfMoves);
      break;

   case TASKTYPE_TEST_MATE_IN_N:
      searchForMate(currentVariation,
                    &currentTask->calculatedSolutions,
                    currentTask->numberOfMoves);
      break;

   default:
      logDebug("Unknown task type: %d\n", currentTask->type);
      break;
   }
}

static int startSearch(Variation * currentVariation)
{
   currentVariation->searchStatus = SEARCH_STATUS_RUNNING;

#ifdef DEBUG_COORDINATION
   logDebug("Search with thread #%d started.\n",
            currentVariation->threadNumber);
#endif

   performTaskSpecificSearch(currentVariation);

   currentTask->nodes = getNodeCount();

   if (currentVariation->threadNumber == 0)
   {
      int threadCount;

      for (threadCount = 1; threadCount < numThreads; threadCount++)
      {
         variations[threadCount].terminate = TRUE;
      }

      /* Timer cancellation is kept for immediate response, 
         but the thread is properly joined in waitForSearchTermination to avoid leaks. */
      if (timerStarted)
      {
         pthread_cancel(timer);
      }
   }

#ifdef DEBUG_COORDINATION
   logDebug("Search thread #%d terminated.\n",
            currentVariation->threadNumber);
#endif

   currentVariation->searchStatus = SEARCH_STATUS_FINISHED;

   return 0;
}

long getElapsedTime(void)
{
   return getTimestamp() - variations[0].startTime;
}

static void *executeSearch(void *arg)
{
   Variation *currentVariation = arg;

   startSearch(currentVariation);

   return 0;
}

static void *watchTime(void *arg)
{
   const Variation *currentVariation = arg;
   const long timeLimit = currentVariation->timeLimit;
   struct timespec requested, remaining;

   requested.tv_sec = timeLimit / 1000;
   requested.tv_nsec = 1000000 * (timeLimit - 1000 * requested.tv_sec);

   if (nanosleep(&requested, &remaining) != -1)
   {
      getGuiSearchMutex();
      prepareSearchAbort();
      releaseGuiSearchMutex();
   }

   return 0;
}

int startTimerThread(const SearchTask * task)
{
   if (task->variation->timeLimit > 0 && task->variation->ponderMode == FALSE)
   {
      if (pthread_create(&timer, NULL, &watchTime, task->variation) == 0)
      {
         timerStarted = TRUE;
#ifdef DEBUG_COORDINATION
         logDebug("Timer thread started.\n");
#endif
      }
      else
      {
         logDebug("### Timer thread could not be started. ###\n");
         return -1;
      }
   }
   return 0;
}

int scheduleTask(SearchTask * task)
{
   const unsigned long startTime = getTimestamp();
   int threadCount;
   pthread_attr_t attr;
   int result = 0;

   sharedHashtable.entriesUsed = 0;

   pthread_attr_init(&attr);
   pthread_attr_setstacksize(&attr, SEARCH_THREAD_STACK_SIZE);

   if (startTimerThread(task) != 0)
   {
      pthread_attr_destroy(&attr);
      return -1;
   }

   currentTask = task;

   for (threadCount = 0; threadCount < numThreads; threadCount++)
   {
      Variation *currentVariation = &variations[threadCount];

      *currentVariation = *(currentTask->variation);
      currentVariation->searchStatus = SEARCH_STATUS_TERMINATE;
      currentVariation->bestBaseMove = NO_MOVE;
      currentVariation->terminate = FALSE;
      currentVariation->pawnHashtable = &(pawnHashtable[threadCount][0]);
      currentVariation->kingsafetyHashtable =
         &(kingSafetyHashtable[threadCount][0]);
      currentVariation->threadNumber = threadCount;
      currentVariation->startTime = startTime;

      if (pthread_create(&searchThread[threadCount], &attr,
                         &executeSearch, currentVariation) == 0)
      {
         searchThreadStarted[threadCount] = TRUE;
#ifdef DEBUG_COORDINATION
         logDebug("Search thread #%d created.\n", threadCount);
#endif
      }
      else
      {
         logDebug("### Search thread #%d could not be started. ###\n",
                  threadCount);
         result = -1;
         break;
      }
   }

   pthread_attr_destroy(&attr);
   return result;
}

void waitForSearchTermination(void)
{
   int threadCount;

   /* Join search threads */
   for (threadCount = 0; threadCount < numThreads; threadCount++)
   {
      if (searchThreadStarted[threadCount])
      {
         pthread_join(searchThread[threadCount], NULL);
         searchThreadStarted[threadCount] = FALSE;
#ifdef DEBUG_COORDINATION
         logDebug("Task %d joined.\n", threadCount);
#endif
      }
   }

   /* Join timer thread */
   if (timerStarted)
   {
      pthread_join(timer, NULL);
      timerStarted = FALSE;
#ifdef DEBUG_COORDINATION
      logDebug("Timer thread joined.\n");
#endif
   }
}

void completeTask(SearchTask * task)
{
   if (scheduleTask(task) == 0)
   {
#ifdef DEBUG_COORDINATION
      logDebug("Task scheduled. Waiting for completion.\n");
#endif
      waitForSearchTermination();
   }
}

void prepareSearchAbort(void)
{
   int threadCount;

   for (threadCount = 0; threadCount < numThreads; threadCount++)
   {
      variations[threadCount].terminate = TRUE;
   }
}

void unsetPonderMode(void)
{
   int threadCount;

   for (threadCount = 0; threadCount < numThreads; threadCount++)
   {
      variations[threadCount].ponderMode = FALSE;
   }
}

void setTimeLimit(unsigned long timeTarget, unsigned long timeLimit)
{
   int threadCount;

   for (threadCount = 0; threadCount < numThreads; threadCount++)
   {
      Variation *currentVariation = &variations[threadCount];

      currentVariation->timeTarget = timeTarget;
      currentVariation->timeLimit = timeLimit;
   }
}

bool setHashtableSizeInMb(unsigned int size)
{
   const UINT64 tablesize = (UINT64)size * BYTES_PER_MB;

   if (setHashtableSize(&sharedHashtable, tablesize))
   {
      resetHashtable(&sharedHashtable);

      return TRUE;
   }

   return FALSE;
}

int initializeModuleCoordination(void)
{
   int threadCount;

   initializeHashtable(&sharedHashtable);

   if (!setHashtableSize(&sharedHashtable, INITIAL_HASHTABLE_SIZE_MB * BYTES_PER_MB))
   {
      return -1;
   }

   resetHashtable(&sharedHashtable);

   for (threadCount = 0; threadCount < MAX_THREADS; threadCount++)
   {
      variations[threadCount].searchStatus = SEARCH_STATUS_FINISHED;
      searchThreadStarted[threadCount] = FALSE;
   }

   return 0;
}

void finalizeModuleCoordination(void)
{
   pthread_mutex_destroy(&guiSearchMutex);
   finalizeHashtable(&sharedHashtable);
}

int testModuleCoordination(void)
{
   return 0;
}

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

#include "uci.h"
#include "coordination.h"
#include "tools.h"
#include "io.h"
#include "fen.h"
#include "pgn.h"
#include "tablebase.h"
#include "hash.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <ctype.h>

static SearchTask task;
static Variation variation;
static PGNGame game;
static UciStatus status;
bool resetSharedHashtable = FALSE;
static int numUciMoves = 1000;
const int valueRangePct = 10;
int drawScore = 0;
int numPvs = 1;
const int DRAW_SCORE_MAX = 10000;

#define MIN_SEND_DEPTH_DEFAULT 12
#define MIN_SEND_TIME_DEFAULT 1000
#define MAX_SEND_HEIGHT_DEFAULT 24
#define MAX_ENTRIES_PS_DEFAULT 4

int minPvHashEntrySendDepth = MIN_SEND_DEPTH_DEFAULT;
int minPvHashEntrySendTime = MIN_SEND_TIME_DEFAULT;
int maxPvHashEntrySendHeight = MAX_SEND_HEIGHT_DEFAULT;
int maxPvHashEntriesSendPerSecond = MAX_ENTRIES_PS_DEFAULT;
int pvHashEntriesSendInterval = 1000 / MAX_ENTRIES_PS_DEFAULT;

const char *USN_NT = "Threads";
const char *USN_QO = "Value Queen Opening";
const char *USN_QE = "Value Queen Endgame";
const char *USN_RO = "Value Rook Opening";
const char *USN_RE = "Value Rook Endgame";
const char *USN_BO = "Value Bishop Opening";
const char *USN_BE = "Value Bishop Endgame";
const char *USN_NO = "Value Knight Opening";
const char *USN_NE = "Value Knight Endgame";
const char *USN_PO = "Value Pawn Opening";
const char *USN_PE = "Value Pawn Endgame";
const char *USN_BPO = "Value Bishop pair Opening";
const char *USN_BPE = "Value Bishop pair Endgame";
const char *USN_DS = "Draw Score";
const char *USN_PV = "MultiPV";

const char *USN_SD = "Min PVHashEntry Send Depth";
const char *USN_ST = "Min PVHashEntry Send Time";
const char *USN_SH = "Max PVHashEntry Send Height";
const char *USN_PS = "Max PVHashEntries Send Per Second";

/* #define DEBUG_GUI_PROTOCOL */
/* #define DEBUG_GUI_PROTOCOL_BRIEF */
/* #define DEBUG_GUI_CONVERSATION */

static Move readUciMove(const char *buffer)
{
   const Square from = getSquare(buffer[0] - 'a', buffer[1] - '1');
   const Square to = getSquare(buffer[2] - 'a', buffer[3] - '1');
   Piece newPiece = NO_PIECE;

   switch (buffer[4])
   {
   case 'q':
   case 'Q':
      newPiece = (Piece) (QUEEN);
      break;

   case 'r':
   case 'R':
      newPiece = (Piece) (ROOK);
      break;

   case 'b':
   case 'B':
      newPiece = (Piece) (BISHOP);
      break;

   case 'n':
   case 'N':
      newPiece = (Piece) (KNIGHT);
      break;

   default:
      newPiece = (Piece) (NO_PIECE);
   }

   return getPackedMove(from, to, newPiece);
}

static const char *getUciToken(const char *uciString, const char *tokenName)
{
   const size_t tokenNameLength = strlen(tokenName);
   const char *current = uciString;

#ifdef   DEBUG_GUI_PROTOCOL
   logDebug("find >%s< in >%s<\n", tokenName, uciString);
#endif

   while (current != NULL && *current != '\0')
   {
      const char *tokenHit = strstr(current, tokenName);

      if (tokenHit == NULL)
      {
         return NULL;
      }

      const char nextChar = *(tokenHit + tokenNameLength);

      if ((tokenHit == uciString || isspace((unsigned char)*(tokenHit - 1))) &&
          (nextChar == '\0' || isspace((unsigned char)nextChar)))
      {
         return tokenHit;
      }

      current = tokenHit + 1;
   }

   return NULL;
}

static void getNextUciToken(const char *uciString, char *buffer, size_t bufferSize)
{
   const char *start = uciString, *end;
   unsigned int tokenLength;

   while (*start != '\0' && isspace((unsigned char)*start))
   {
      start++;
   }

   if (*start == '\0')
   {
      strncpy(buffer, "", bufferSize);
      buffer[bufferSize - 1] = '\0';

      return;
   }

   assert(*start != '\0' && isspace((unsigned char)*start) == FALSE);

   end = start + 1;

   while (*end != '\0' && isspace((unsigned char)*end) == FALSE)
   {
      end++;
   }

   assert(*end == '\0' || isspace((unsigned char)*end));

   tokenLength = (unsigned int) (end - start);

   unsigned int copyLength = (unsigned int) min(tokenLength, (unsigned int)bufferSize - 1);
   strncpy(buffer, start, copyLength);
   buffer[copyLength] = '\0';
}

static long getLongUciValue(const char *uciString, const char *name,
                            int defaultValue)
{
   long value;
   char valueBuffer[256];
   const char *nameStart = getUciToken(uciString, name);

   if (nameStart == 0)
   {
      value = defaultValue;
   }
   else
   {
      getNextUciToken(nameStart + strlen(name), valueBuffer, sizeof(valueBuffer));
      value = atol(valueBuffer);
   }

#ifdef   DEBUG_GUI_PROTOCOL
   logDebug("get uci long value; %s = %ld\n", name, value);
#endif

   return value;
}

static void getStringUciValue(const char *uciString, const char *name,
                              char *stringValue, size_t bufferSize)
{
   const char *nameStart = getUciToken(uciString, name);

   if (nameStart == 0)
   {
      stringValue[0] = 0;
   }
   else
   {
      getNextUciToken(nameStart + strlen(name), stringValue, bufferSize);
   }

#ifdef   DEBUG_GUI_PROTOCOL
   logDebug("get uci string value; %s = %ld\n", name, stringValue);
#endif
}

static void getUciNamedValue(const char *uciString, char *name, size_t nameSize, char *value, size_t valueSize)
{
   const char *nameTagStart = getUciToken(uciString, "name");
   const char *valueTagStart = getUciToken(uciString, "value");

   name[0] = value[0] = '\0';

   if (nameTagStart != 0 && valueTagStart != 0 &&
       nameTagStart < valueTagStart)
   {
      const int nameLength = (int) (valueTagStart - 1 - (nameTagStart + 5));

      unsigned int copyLength = (unsigned int) min(nameLength, (int)nameSize - 1);
      strncpy(name, nameTagStart + 5, copyLength);
      name[copyLength] = '\0';
      getNextUciToken(valueTagStart + 6, value, valueSize);

      trim(name);
      trim(value);

      /* logDebug("name=<%s> value=<%s>\n", name, value); */
   }
}

/******************************************************************************
 *
 * Get the specified move in uci format.
 *
 ******************************************************************************/
static void getGuiMoveString(const Move move, char *buffer, size_t bufferSize)
{
   char from[16], to[16];

   getSquareName(getFromSquare(move), from);
   getSquareName(getToSquare(move), to);

   if (getNewPiece(move) == NO_PIECE)
   {
      snprintf(buffer, bufferSize, "%s%s", from, to);
   }
   else
   {
      const int pieceIndex = getLimitedValue(0, 15, getNewPiece(move));

      snprintf(buffer, bufferSize, "%s%s%c", from, to, pieceSymbol[pieceIndex]);
   }
}

/******************************************************************************
 *
 * Get the specified param from the specified uci command string.
 *
 ******************************************************************************/
static void getTokenByNumber(const char *command, int paramNumber,
                             char *buffer)
{
   int paramCount = 0;
   char currentChar;
   bool escapeMode = FALSE;
   char *pbuffer = buffer;

   while ((currentChar = *command++) != '\0' && paramCount <= paramNumber)
   {
      if (currentChar == '{')
      {
         escapeMode = TRUE;
      }
      else if (currentChar == '}')
      {
         escapeMode = FALSE;
      }

      if (isspace((unsigned char)currentChar) && escapeMode == FALSE)
      {
         paramCount++;

         while (isspace((unsigned char)*command))
         {
            command++;
         }
      }

      if (paramCount == paramNumber)
      {
         *pbuffer++ = currentChar;
      }
   }

   *pbuffer = '\0';
   trim(buffer);
}

/******************************************************************************
 *
 * Send the specified command via stdout to uci.
 *
 ******************************************************************************/
static void sendToUci(const char *fmt, ...)
{
   va_list args;
   char buffer[4096];

   va_start(args, fmt);
   vsnprintf(buffer, sizeof(buffer), fmt, args);
   va_end(args);

   fprintf(stdout, "%s\n", buffer);
   fflush(stdout);

#ifdef DEBUG_GUI_CONVERSATION
   logDebug("### sent to uci: %s\n", buffer);
#endif
}

/******************************************************************************
 *
 * Send the specified command via stdout to uci.
 *
 ******************************************************************************/
static void sendToUciNonDebug(const char *fmt, ...)
{
   va_list args;
   char buffer[4096];

   va_start(args, fmt);
   vsnprintf(buffer, sizeof(buffer), fmt, args);
   va_end(args);

   fprintf(stdout, "%s\n", buffer);
   fflush(stdout);

#ifdef DEBUG_GUI_CONVERSATION
   logDebug("### sent to uci: %s\n", buffer);
#endif
}

/******************************************************************************
 *
 * Determine the calculation time for the next task in milliseconds.
 *
 ******************************************************************************/
static int getCalculationTime(TimecontrolData * data)
{
   if (data->restTime < 0)
   {
      return 2 * data->incrementTime;
   }

   if (data->movesToGo > 0)
   {
      return data->restTime / data->movesToGo + data->incrementTime;
   }
   else
   {
      const int movesToGo = max(32, 58 - data->numberOfMovesPlayed);

      return (data->incrementTime > 0 ?
              data->incrementTime + data->restTime / movesToGo :
              data->restTime / movesToGo);
   }
}

/******************************************************************************
 *
 * Determine the calculation time for the next task in milliseconds.
 *
 ******************************************************************************/
static int getMaximumCalculationTime(TimecontrolData * data)
{
   if (data->restTime < 0)
   {
      return data->incrementTime;
   }

   if (data->movesToGo > 0)
   {
      const int standardTime = data->restTime / data->movesToGo;
      const int maxTime = data->restTime / 2;

      return min(7 * standardTime, maxTime);
   }
   else
   {
      const int maxTime1 = (100 * data->restTime) / 256;
      const int maxTime2 = 7 * getCalculationTime(data);

      return min(maxTime1, maxTime2);
   }
}

/******************************************************************************
 *
 * Determine the calculation time for the next task in milliseconds.
 *
 ******************************************************************************/
static int determineCalculationTime(bool targetTime)
{
   if (status.operationMode == UCI_OPERATIONMODE_ANALYSIS)
   {
      return 0;
   }
   else
   {
      TimecontrolData *tcd = &status.timecontrolData[status.engineColor];

      initializeVariationFromGame(&variation, &game);
      tcd->numberOfMovesPlayed = variation.singlePosition.moveNumber - 1;

      if (targetTime)
      {
         return max(1, 95 * getCalculationTime(tcd) / 100);
      }
      else
      {
         return max(1, 95 * getMaximumCalculationTime(tcd) / 100);
      }
   }
}

/******************************************************************************
 *
 * Start the calculation of the current position.
 *
 ******************************************************************************/
static void startCalculation(void)
{
   variation.timeTarget = determineCalculationTime(TRUE);
   variation.timeLimit = determineCalculationTime(FALSE);
   setDrawScore(&variation, drawScore, status.engineColor);
   status.bestMoveWasSent = FALSE;

#ifdef DEBUG_GUI_CONVERSATION
   logDebug("Scheduling task. Timelimits: %d/%d\n", variation.timeTarget,
            variation.timeLimit);
#endif

   scheduleTask(&task);
}

static void startPostPonderCalculation(void)
{
   const long elapsedTime = getElapsedTime();
   const long nominalRestTime = variation.timeLimit - elapsedTime;
   const long minimalRestTime = max(1, variation.timeLimit / 4);

   /* logDebug
      ("Preparing post ponder calculation. timelimit: %d elapsed time: %d nomimalRestTime: %d minimal rest time: %d \n",
      variation.timeLimit, elapsedTime, nominalRestTime, minimalRestTime); */

   variation.timeLimit = max(nominalRestTime, minimalRestTime);
   variation.ponderMode = FALSE;

   /* logDebug("Starting post ponder calculation. Timelimits: %d/%d\n",
      variation.timeTarget, variation.timeLimit); */

#ifdef DEBUG_GUI_CONVERSATION
   logDebug("Starting post ponder calculation. Timelimits: %d/%d\n",
            variation.timeTarget, variation.timeLimit);
#endif

   startTimerThread(&task);
}

/******************************************************************************
 *
 * Delete the current ponder result.
 *
 ******************************************************************************/
static void deletePonderResult(void)
{
   status.ponderResultMove = NO_MOVE;
}

/******************************************************************************
 *
 * Get a UCI-compliant pv.
 *
 ******************************************************************************/
static char *getUciPv(const PrincipalVariation * pv, char *buffer, size_t bufferSize)
{
   int i;

   buffer[0] = '\0';

   for (i = 0; i < min(32, pv->length); i++)
   {
      const Move move = (Move) pv->move[i];

      if (move != NO_MOVE)
      {
         char moveBuffer[16];

         if (i > 0)
         {
            strncat(buffer, " ", bufferSize - strlen(buffer) - 1);
         }

         getGuiMoveString(move, moveBuffer, sizeof(moveBuffer));
         strncat(buffer, moveBuffer, bufferSize - strlen(buffer) - 1);
      }
      else
      {
         break;
      }
   }

   return buffer;
}

/******************************************************************************
 *
 * Post a principal variation line.
 *
 ******************************************************************************/
static void postPv(Variation * var, bool sendAnyway)
{
   const char *format =
      "info depth %d seldepth %d time %.0f nodes %llu score %s %s tbhits %lu %s pv %s";
   double time = getTimestamp() - var->startTime;
   char pvBuffer[2048];
   char pvMovesBuffer[1024];
   char scoreBuffer[16];
   char scoreTypeBuffer[32] = "";
   char multiPvBuffer[32] = "";

   if (time >= 250 || sendAnyway)
   {
      const UINT64 nodeCount = getNodeCount();
      const PrincipalVariation *pv = &var->pv[var->pvId];

      if (numPvs > 1)
      {
         snprintf(multiPvBuffer, sizeof(multiPvBuffer), "multipv %d", var->pvId + 1);
      }

      getUciPv(pv, pvMovesBuffer, sizeof(pvMovesBuffer));
      formatUciValue(pv->score, scoreBuffer, sizeof(scoreBuffer));

      if (pv->scoreType == HASHVALUE_LOWER_LIMIT)
      {
         snprintf(scoreTypeBuffer, sizeof(scoreTypeBuffer), "lowerbound");
      }
      else if (pv->scoreType == HASHVALUE_UPPER_LIMIT)
      {
         snprintf(scoreTypeBuffer, sizeof(scoreTypeBuffer), "upperbound");
      }

      snprintf(pvBuffer, sizeof(pvBuffer), format, var->iteration, var->selDepth, time,
              nodeCount, scoreBuffer, scoreTypeBuffer, var->tbHits,
              multiPvBuffer, pvMovesBuffer);

      sendToUciNonDebug("%s", pvBuffer);

      var->numPvUpdates++;
   }
}

/******************************************************************************
 *
 * Post a statistics information about the current search.
 *
 ******************************************************************************/
static void reportBaseMoveUpdate(const Variation * var)
{
   const double time = getTimestamp() - var->startTime;
   char movetext[16];

   if (time >= 500)
   {
      getGuiMoveString(var->currentBaseMove, movetext, sizeof(movetext));

      sendToUciNonDebug
         ("info depth %d seldepth %d currmove %s currmovenumber %d",
          var->iteration, var->selDepth, movetext,
          var->numberOfCurrentBaseMove);
   }
}

/******************************************************************************
 *
 * Post a statistics information about the current search.
 *
 ******************************************************************************/

static void reportStatisticsUpdate(Variation * var)
{
   UINT64 nodeCount = getNodeCount();
   const double time = getTimestamp() - var->startTime;
   const double nps = (nodeCount / max((double) 0.001, (time / 1000.0)));
   const double hashUsage =
      ((double) getSharedHashtable()->entriesUsed * 1000.0) /
      (max((double) 1.0, (double) getSharedHashtable()->tableSize));

   sendToUciNonDebug
      ("info time %0.f nodes %lld nps %.0f hashfull %.0f tbhits %lu",
       time, nodeCount, nps, hashUsage, var->tbHits);
   reportBaseMoveUpdate(var);

   if (var->numPvUpdates == 0)
   {
      postPv(var, FALSE);
   }
}

/******************************************************************************
 *
 * Send a bestmove info to the gui.
 *
 ******************************************************************************/
static void sendBestmoveInfo(Variation * var)
{
   char moveBuffer[8];

   postPv(var, TRUE);

   if (moveIsLegal(&var->startPosition, var->bestBaseMove))
   {
      Variation tmp = *var;

      getGuiMoveString(var->bestBaseMove, moveBuffer, sizeof(moveBuffer));
      status.ponderingMove = (Move) var->completePv.move[1];
      setBasePosition(&tmp, &var->startPosition);
      makeMove(&tmp, var->bestBaseMove);

      if (status.ponderingMove != NO_MOVE &&
          moveIsLegal(&tmp.singlePosition, status.ponderingMove))
      {
         char ponderMoveBuffer[16];

         getGuiMoveString(status.ponderingMove, ponderMoveBuffer, sizeof(ponderMoveBuffer));
         sendToUciNonDebug("bestmove %s ponder %s", moveBuffer,
                              ponderMoveBuffer);
#ifdef DEBUG_GUI_PROTOCOL_BRIEF
         logDebug("bestmove %s ponder %s\n", moveBuffer, ponderMoveBuffer);
#endif
      }
      else
      {
         status.ponderingMove = NO_MOVE;
         sendToUciNonDebug("bestmove %s", moveBuffer);
#ifdef DEBUG_GUI_PROTOCOL_BRIEF
         logDebug("bestmove %s\n", moveBuffer);
#endif
      }

      unmakeLastMove(&tmp);
   }
   else
   {
      getGuiMoveString(var->bestBaseMove, moveBuffer, sizeof(moveBuffer));

#ifdef DEBUG_GUI_CONVERSATION
      logDebug("### Illegal best move %s ###\n", moveBuffer);
      logDebug("### Sending bestmove 0000 ###\n");
#endif

      sendToUciNonDebug("bestmove 0000");
   }

   status.bestMoveWasSent = TRUE;
}

/******************************************************************************
 *
 * Send a hash entry via UCI.
 *
 ******************************************************************************/
void sendHashentry(Hashentry * entry)
{
   /* const char *fmt = "info transkey %llx transdata %llx"; */
   /* kh 2015-09-21 %llx prints only 32 bits on some windows systems */
#if defined(_WIN32) || defined(_WIN64)
   const char *fmt = "info transkey %I64x transdata %I64x";
#else
   const char *fmt = "info transkey %llx transdata %llx";
#endif

   getGuiSearchMutex();
   sendToUci(fmt, entry->key, entry->data);
   releaseGuiSearchMutex();
}

/******************************************************************************
 *
 * Handle events generated by the search engine.
 *
 ******************************************************************************/
static void handleSearchEvent(int eventId, void *var)
{
   Variation *variation = (Variation *) var;

   switch (eventId)
   {
   case SEARCHEVENT_SEARCH_FINISHED:
      if (status.engineIsPondering == FALSE)
      {
         sendBestmoveInfo(variation);
      }
      else
      {
#ifdef DEBUG_GUI_CONVERSATION
         logDebug("Search finished. Engine was pondering.\n");
         logDebug("No best move info sent.\n");
#endif
         postPv(variation, TRUE);
      }

      status.engineIsActive = FALSE;
      break;

   case SEARCHEVENT_PLY_FINISHED:
      postPv(variation, TRUE);
      break;

   case SEARCHEVENT_NEW_BASEMOVE:
      reportBaseMoveUpdate(variation);
      reportStatisticsUpdate(variation);
      break;

   case SEARCHEVENT_STATISTICS_UPDATE:
      reportStatisticsUpdate(variation);
      break;

   case SEARCHEVENT_NEW_PV:
      postPv(variation, TRUE);
      break;

   default:
      break;
   }

}

static int getIntValue(const char *value, int minValue, int defaultValue,
                       int maxValue)
{
   int parsedValue = atoi(value);

   if (parsedValue == 0 && parsedValue < minValue)
   {
      return defaultValue;
   }
   else
   {
      return min(max(parsedValue, minValue), maxValue);
   }
}

static int getValueLimit(int value, int diffPct)
{
   return (value * (100 + diffPct)) / 100;
}

static int getStdIntValue(const char *value, int defaultValue)
{
   const int minValue = getValueLimit(defaultValue, -valueRangePct);
   const int maxValue = getValueLimit(defaultValue, valueRangePct);

   return getIntValue(value, minValue, defaultValue, maxValue);
}

static void sendUciSpinOption(const char *name, const int defaultValue,
                              int minValue, int maxValue)
{
   sendToUciNonDebug("option name %s type spin default %d min %d max %d",
                        name, defaultValue, minValue, maxValue);
}

void addHashentry(Hashentry * entry)
{
   setHashentry(getSharedHashtable(), entry->key, getHashentryValue(entry),
                getHashentryImportance(entry), getHashentryMove(entry),
                getHashentryFlag(entry), getHashentryStaticValue(entry));
}

/******************************************************************************
 *
 * Process the specified UCI command.
 *
 ******************************************************************************/
static int processUciCommand(const char *command)
{
   char buffer[8192];

#ifdef DEBUG_GUI_PROTOCOL_BRIEF
   logDebug("%s", command);
#endif

   getTokenByNumber(command, 0, buffer);

   if (strcmp(buffer, "uci") == 0)
   {
      char nameString[256];

      getGuiSearchMutex();
      snprintf(nameString, sizeof(nameString), "id name Protector %s", programVersionNumber);
      sendToUciNonDebug("%s", nameString);
      sendToUciNonDebug("id author Raimund Heid");
      sendToUciNonDebug
         ("option name Hash type spin default 16 min 8 max 65536");
      sendToUciNonDebug
         ("option name SyzygyPath type string default <empty>");
      sendToUciNonDebug("option name Ponder type check default true");
      sendUciSpinOption(USN_NT, 1, 1, MAX_THREADS);
      sendUciSpinOption(USN_PO, DEFAULTVALUE_PAWN_OPENING,
                        getValueLimit(DEFAULTVALUE_PAWN_OPENING,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_PAWN_OPENING,
                                      valueRangePct));
      sendUciSpinOption(USN_PE, DEFAULTVALUE_PAWN_ENDGAME,
                        getValueLimit(DEFAULTVALUE_PAWN_ENDGAME,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_PAWN_ENDGAME,
                                      valueRangePct));
      sendUciSpinOption(USN_NO, DEFAULTVALUE_KNIGHT_OPENING,
                        getValueLimit(DEFAULTVALUE_KNIGHT_OPENING,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_KNIGHT_OPENING,
                                      valueRangePct));
      sendUciSpinOption(USN_NE, DEFAULTVALUE_KNIGHT_ENDGAME,
                        getValueLimit(DEFAULTVALUE_KNIGHT_ENDGAME,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_KNIGHT_ENDGAME,
                                      valueRangePct));
      sendUciSpinOption(USN_BO, DEFAULTVALUE_BISHOP_OPENING,
                        getValueLimit(DEFAULTVALUE_BISHOP_OPENING,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_BISHOP_OPENING,
                                      valueRangePct));
      sendUciSpinOption(USN_BE, DEFAULTVALUE_BISHOP_ENDGAME,
                        getValueLimit(DEFAULTVALUE_BISHOP_ENDGAME,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_BISHOP_ENDGAME,
                                      valueRangePct));
      sendUciSpinOption(USN_RO, DEFAULTVALUE_ROOK_OPENING,
                        getValueLimit(DEFAULTVALUE_ROOK_OPENING,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_ROOK_OPENING,
                                      valueRangePct));
      sendUciSpinOption(USN_RE, DEFAULTVALUE_ROOK_ENDGAME,
                        getValueLimit(DEFAULTVALUE_ROOK_ENDGAME,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_ROOK_ENDGAME,
                                      valueRangePct));
      sendUciSpinOption(USN_QO, DEFAULTVALUE_QUEEN_OPENING,
                        getValueLimit(DEFAULTVALUE_QUEEN_OPENING,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_QUEEN_OPENING,
                                      valueRangePct));
      sendUciSpinOption(USN_QE, DEFAULTVALUE_QUEEN_ENDGAME,
                        getValueLimit(DEFAULTVALUE_QUEEN_ENDGAME,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_QUEEN_ENDGAME,
                                      valueRangePct));
      sendUciSpinOption(USN_BPO, DEFAULTVALUE_BISHOP_PAIR_OPENING,
                        getValueLimit(DEFAULTVALUE_BISHOP_PAIR_OPENING,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_BISHOP_PAIR_OPENING,
                                      valueRangePct));
      sendUciSpinOption(USN_BPE, DEFAULTVALUE_BISHOP_PAIR_ENDGAME,
                        getValueLimit(DEFAULTVALUE_BISHOP_PAIR_ENDGAME,
                                      -valueRangePct),
                        getValueLimit(DEFAULTVALUE_BISHOP_PAIR_ENDGAME,
                                      valueRangePct));
      sendUciSpinOption(USN_DS, 0, -DRAW_SCORE_MAX, DRAW_SCORE_MAX);
      sendUciSpinOption(USN_PV, 1, 1, MAX_NUM_PV);

#ifdef SEND_HASH_ENTRIES
      sendUciSpinOption(USN_SD, MIN_SEND_DEPTH_DEFAULT, 8, 16);
      sendUciSpinOption(USN_ST, MIN_SEND_TIME_DEFAULT, 100, 5000);
      sendUciSpinOption(USN_SH, MAX_SEND_HEIGHT_DEFAULT, 16, 64);
      sendUciSpinOption(USN_PS, MAX_ENTRIES_PS_DEFAULT, 1, 20);
#endif

      sendToUciNonDebug("uciok");
      releaseGuiSearchMutex();

      return TRUE;
   }

   if (strcmp(buffer, "isready") == 0)
   {
      getGuiSearchMutex();
      sendToUciNonDebug("readyok");
      releaseGuiSearchMutex();

      return TRUE;
   }

   if (strcmp(buffer, "ucinewgame") == 0)
   {
      resetSharedHashtable = TRUE;

      return TRUE;
   }

   if (strcmp(buffer, "setoption") == 0)
   {
      char name[256], value[256];

      getUciNamedValue(command, name, sizeof(name), value, sizeof(value));

      if (strcmp(name, "SyzygyPath") == 0)
      {
         initializeTablebase(value);

         return TRUE;
      }

      if (strcmp(name, "Hash") == 0)
      {
         const unsigned int hashsize = (unsigned int) max(8, atoi(value));

         if (!setHashtableSizeInMb(hashsize))
         {
            sendToUciNonDebug("info string Failed to set hashtable size to %u MB", hashsize);
         }

         return TRUE;
      }

      if (strcmp(name, USN_NT) == 0)
      {
         const unsigned int numThreads =
            (unsigned int) getIntValue(value, 1, 1, MAX_THREADS);

         setNumberOfThreads(numThreads);

         return TRUE;
      }

      if (strcmp(name, USN_PO) == 0)
      {
         VALUE_PAWN_OPENING = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_PAWN_OPENING);

         return TRUE;
      }

      if (strcmp(name, USN_PE) == 0)
      {
         VALUE_PAWN_ENDGAME = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_PAWN_ENDGAME);

         return TRUE;
      }

      if (strcmp(name, USN_NO) == 0)
      {
         VALUE_KNIGHT_OPENING = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_KNIGHT_OPENING);

         return TRUE;
      }

      if (strcmp(name, USN_NE) == 0)
      {
         VALUE_KNIGHT_ENDGAME = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_KNIGHT_ENDGAME);

         return TRUE;
      }

      if (strcmp(name, USN_BO) == 0)
      {
         VALUE_BISHOP_OPENING = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_BISHOP_OPENING);

         return TRUE;
      }

      if (strcmp(name, USN_BE) == 0)
      {
         VALUE_BISHOP_ENDGAME = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_BISHOP_ENDGAME);

         return TRUE;
      }

      if (strcmp(name, USN_RO) == 0)
      {
         VALUE_ROOK_OPENING = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_ROOK_OPENING);

         return TRUE;
      }

      if (strcmp(name, USN_RE) == 0)
      {
         VALUE_ROOK_ENDGAME = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_ROOK_ENDGAME);

         return TRUE;
      }

      if (strcmp(name, USN_QO) == 0)
      {
         VALUE_QUEEN_OPENING = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_QUEEN_OPENING);

         return TRUE;
      }

      if (strcmp(name, USN_QE) == 0)
      {
         VALUE_QUEEN_ENDGAME = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_QUEEN_ENDGAME);

         return TRUE;
      }

      if (strcmp(name, USN_BPO) == 0)
      {
         VALUE_BISHOP_PAIR_OPENING = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_BISHOP_PAIR_OPENING);

         return TRUE;
      }

      if (strcmp(name, USN_BPE) == 0)
      {
         VALUE_BISHOP_PAIR_ENDGAME = (unsigned int)
            getStdIntValue(value, DEFAULTVALUE_BISHOP_PAIR_ENDGAME);

         return TRUE;
      }

      if (strcmp(name, USN_DS) == 0)
      {
         drawScore = getIntValue(value, -DRAW_SCORE_MAX, 0, DRAW_SCORE_MAX);

         return TRUE;
      }

      if (strcmp(name, USN_PV) == 0)
      {
         numPvs = getIntValue(value, 1, 1, MAX_NUM_PV);

         return TRUE;
      }

#ifdef SEND_HASH_ENTRIES
      if (strcmp(name, USN_SD) == 0)
      {
         minPvHashEntrySendDepth =
            getIntValue(value, 0, MIN_SEND_DEPTH_DEFAULT, 1000000);

         return TRUE;
      }

      if (strcmp(name, USN_ST) == 0)
      {
         minPvHashEntrySendTime =
            getIntValue(value, 0, MIN_SEND_TIME_DEFAULT, 1000000);

         return TRUE;
      }

      if (strcmp(name, USN_SH) == 0)
      {
         maxPvHashEntrySendHeight =
            getIntValue(value, 0, MAX_SEND_HEIGHT_DEFAULT, 1000000);

         return TRUE;
      }

      if (strcmp(name, USN_PS) == 0)
      {
         maxPvHashEntriesSendPerSecond =
            getIntValue(value, 1, MAX_ENTRIES_PS_DEFAULT, 1000000);

         pvHashEntriesSendInterval = 1000 / maxPvHashEntriesSendPerSecond;

         return TRUE;
      }
#endif
   }

   if (strcmp(buffer, "position") == 0)
   {
      resetPGNGame(&game);

      if (getUciToken(command, "fen") != 0)
      {
         const char *fenStart = getUciToken(command, "fen") + 3;
         const char *fenEnd = getUciToken(command, "moves");

         if (fenEnd == 0)
         {
            strncpy(game.fen, fenStart, sizeof(game.fen));
            game.fen[sizeof(game.fen) - 1] = '\0';
         }
         else
         {
            const int length = (int) (fenEnd - fenStart - 1);

            unsigned int copyLength = (unsigned int) min(length, (int)sizeof(game.fen) - 1);
            strncpy(game.fen, fenStart, copyLength);
            game.fen[copyLength] = '\0';
         }

         trim(game.fen);
         strncpy(game.setup, "1", sizeof(game.setup));
         game.setup[sizeof(game.setup) - 1] = '\0';

#ifdef   DEBUG_GUI_PROTOCOL
         logDebug("fen set: >%s<\n", game.fen);
#endif
      }

      if (getUciToken(command, "moves") != 0)
      {
         char moveBuffer[8];
         const char *currentMove = getUciToken(command, "moves") + 5;
         bool finished = FALSE;
         int moveCount = 0;

         do
         {
            getNextUciToken(currentMove, moveBuffer, sizeof(moveBuffer));

            if (strlen(moveBuffer) > 0)
            {
               Move move = readUciMove(moveBuffer);

#ifdef   DEBUG_GUI_PROTOCOL
               logDebug("move found: >%s<\n", moveBuffer);
#endif

               if (appendMove(&game, move) == 0)
               {
                  currentMove += strlen(moveBuffer) + 1;
                  moveCount++;
               }
               else
               {
                  finished = TRUE;
               }
            }
            else
            {
               finished = TRUE;
            }
         }
         while (finished == FALSE);

         if (moveCount < numUciMoves - 1)
         {
            resetSharedHashtable = TRUE;
         }

         numUciMoves = moveCount;
      }

      return TRUE;
   }

   if (strcmp(buffer, "stop") == 0)
   {
      getGuiSearchMutex();

      status.engineIsPondering = FALSE;

      if (status.engineIsActive)
      {
#ifdef DEBUG_GUI_CONVERSATION
         logDebug("stopping search...\n");
#endif
         prepareSearchAbort();
         releaseGuiSearchMutex();
         waitForSearchTermination();

         if (status.bestMoveWasSent == FALSE)
         {
            logDebug("### best move was not sent on stop ...\n");
            reportVariation(getCurrentVariation());
            sendBestmoveInfo(getCurrentVariation());
         }

         return TRUE;
      }
      else
      {
         if (status.bestMoveWasSent == FALSE)
         {
#ifdef DEBUG_GUI_CONVERSATION
            logDebug("sending best move info on stop ...\n");
#endif
            sendBestmoveInfo(getCurrentVariation());
         }
      }

      releaseGuiSearchMutex();

      return TRUE;
   }

   if (strcmp(buffer, "ponderhit") == 0)
   {
#ifdef DEBUG_GUI_CONVERSATION
      logDebug("handling ponderhit...\n");
#endif

      getGuiSearchMutex();

      status.engineIsPondering = FALSE;

      if (status.engineIsActive)
      {
         if (getCurrentVariation()->terminateSearchOnPonderhit &&
             getCurrentVariation()->failingLow == FALSE)
         {
#ifdef DEBUG_GUI_CONVERSATION
            logDebug("immediate termination of pondering.\n");
#endif

            prepareSearchAbort();
         }
         else
         {
#ifdef DEBUG_GUI_CONVERSATION
            logDebug("unsetting pondering mode.\n");
#endif

            unsetPonderMode();
            startPostPonderCalculation();
         }
      }
      else
      {
#ifdef DEBUG_GUI_CONVERSATION
         logDebug("Pondering finished prematurely. Sending best move info\n");
#endif

         sendBestmoveInfo(getCurrentVariation());
      }

      releaseGuiSearchMutex();

      return TRUE;
   }

   if (strcmp(buffer, "go") == 0)
   {
      getGuiSearchMutex();
      status.engineIsActive = TRUE;
      task.type = TASKTYPE_BEST_MOVE;

      initializeVariationFromGame(&variation, &game);
      status.engineColor = variation.singlePosition.activeColor;

      if (getUciToken(command, "depth") != 0)
      {
         status.operationMode = UCI_OPERATIONMODE_ANALYSIS;
      }
      else if (getUciToken(command, "nodes") != 0)
      {
         status.operationMode = UCI_OPERATIONMODE_ANALYSIS;
      }
      else if (getUciToken(command, "mate") != 0)
      {
         task.type = TASKTYPE_MATE_IN_N;
         task.numberOfMoves = getLongUciValue(command, "mate", 1);
         status.operationMode = UCI_OPERATIONMODE_ANALYSIS;

#ifdef   DEBUG_GUI_PROTOCOL
         logDebug("Searching mate in %d", task.numberOfMoves);
#endif
      }
      else if (getUciToken(command, "movetime") != 0)
      {
         status.operationMode = UCI_OPERATIONMODE_USERGAME;

         status.timecontrolData[WHITE].restTime =
            status.timecontrolData[BLACK].restTime = -1;
         status.timecontrolData[WHITE].incrementTime =
            status.timecontrolData[BLACK].incrementTime =
            getLongUciValue(command, "movetime", 5000);
      }
      else if (getUciToken(command, "infinite") != 0)
      {
         status.operationMode = UCI_OPERATIONMODE_ANALYSIS;
      }
      else
      {
         const int numMovesPlayed = variation.singlePosition.moveNumber - 1;
         const int movesToGo = (int) getLongUciValue(command, "movestogo", 0);

         status.operationMode = UCI_OPERATIONMODE_USERGAME;

         status.timecontrolData[WHITE].restTime =
            getLongUciValue(command, "wtime", 1000);
         status.timecontrolData[WHITE].incrementTime =
            getLongUciValue(command, "winc", 0);
         status.timecontrolData[BLACK].restTime =
            getLongUciValue(command, "btime", 1000);
         status.timecontrolData[BLACK].incrementTime =
            getLongUciValue(command, "binc", 0);
         status.timecontrolData[WHITE].numberOfMovesPlayed =
            status.timecontrolData[BLACK].numberOfMovesPlayed =
            numMovesPlayed;

         if (movesToGo > 0)
         {
            status.timecontrolData[WHITE].movesToGo =
               status.timecontrolData[BLACK].movesToGo = movesToGo;
         }
         else
         {
            status.timecontrolData[WHITE].movesToGo =
               status.timecontrolData[BLACK].movesToGo = 0;
         }
      }

      if (getUciToken(command, "ponder") == 0)
      {
         status.engineIsPondering = variation.ponderMode = FALSE;
      }
      else
      {
         status.engineIsPondering = variation.ponderMode = TRUE;
         variation.terminateSearchOnPonderhit = FALSE;  /* avoid premature search aborts */
      }

      startCalculation();
      releaseGuiSearchMutex();

      return TRUE;
   }

   if (strcmp(buffer, "settransentry") == 0)
   {
      char value[256];
      Hashentry entry;
      bool entryIsValid = TRUE;

      getStringUciValue(command, "key", value, sizeof(value));

      if (strlen(value) > 0)
      {
         /* printf("key value: %s\n",value); */

         entry.key = getUnsignedLongLongFromHexString(value);
      }
      else
      {
         entryIsValid = FALSE;
      }

      getStringUciValue(command, "data", value, sizeof(value));

      if (strlen(value) > 0)
      {
         /* printf("data value: %s\n",value); */

         entry.data = getUnsignedLongLongFromHexString(value);
      }
      else
      {
         entryIsValid = FALSE;
      }

      if (entryIsValid)
      {
         addHashentry(&entry);
      }
   }

   if (strcmp(buffer, "quit") == 0)
   {
      status.engineIsPondering = FALSE;
      prepareSearchAbort();

      return FALSE;
   }

   return TRUE;
}

/******************************************************************************
 *
 * Read uci's commands from stdin and handle them.
 *
 ******************************************************************************/
void acceptGuiCommands(void)
{
   bool finished = FALSE;
   char command[8192];

   /* signal(SIGINT, SIG_IGN); */

   while (finished == FALSE)
   {
      if (fgets(command, sizeof(command), stdin) == NULL)
      {
         finished = TRUE;
      }
      else
      {
         trim(command);

#ifdef DEBUG_GUI_CONVERSATION
         logDebug("\n### gui command: >%s<\n", command);
#endif

         finished = (bool) (processUciCommand(command) == FALSE);

#ifdef DEBUG_GUI_CONVERSATION
         logDebug(">%s< processed.\n\n", command);
#endif
      }
   }

#ifdef   DEBUG_GUI_PROTOCOL
   logDebug("GUI command processing terminated.\n");
#endif
}

/******************************************************************************
 *
 * (See the header file comment for this function.)
 *
 ******************************************************************************/
int initializeModuleUci(void)
{
   status.operationMode = UCI_OPERATIONMODE_USERGAME;
   status.engineColor = WHITE;
   status.pondering = TRUE;
   status.ponderingMove = NO_MOVE;
   status.engineIsPondering = FALSE;
   status.engineIsActive = FALSE;
   status.bestMoveWasSent = TRUE;
   status.maxPlies = 0;
   status.timecontrolData[WHITE].movesToGo = 0;
   status.timecontrolData[WHITE].incrementTime = 0;
   status.timecontrolData[WHITE].numberOfMovesPlayed = 0;
   status.timecontrolData[WHITE].restTime = 300 * 1000;
   status.timecontrolData[BLACK].movesToGo = 0;
   status.timecontrolData[BLACK].incrementTime =
      status.timecontrolData[WHITE].incrementTime;
   status.timecontrolData[BLACK].numberOfMovesPlayed = 0;
   status.timecontrolData[BLACK].restTime =
      status.timecontrolData[WHITE].restTime;
   deletePonderResult();

   initializePGNGame(&game);
   variation.timeLimit = 5000;
   variation.ponderMode = FALSE;
   variation.handleSearchEvent = &handleSearchEvent;
   task.variation = &variation;
   task.type = TASKTYPE_BEST_MOVE;

   return 0;
}

#ifndef NDEBUG

/******************************************************************************
 *
 * Test parameter parsing.
 *
 ******************************************************************************/
static int testParameterParsing(void)
{
   char buffer[1024];

   getTokenByNumber("protover 2", 1, buffer);
   assert(strcmp("2", buffer) == 0);

   getTokenByNumber("result 1-0 {White mates}", 2, buffer);
   assert(strcmp("{White mates}", buffer) == 0);

   return 0;
}

/******************************************************************************
 *
 * Test time calculation.
 *
 ******************************************************************************/
static int testTimeCalculation(void)
{
   return 0;
}

/******************************************************************************
 *
 * Test the uci tokenizer.
 *
 ******************************************************************************/
static int testUciTokenizer(void)
{
   char buffer[64], name[64], value[64];
   const char *uciString =
      "setoption name\tSyzygyPath    value  \t  C:\\chess\\tablebases   time  641273423";
   const char *trickyUciString =
      "setoption name\tSyzygyPathvalue    value  \t  C:\\chess\\tablebases   time  641273423 tablebases";
   const char *token1 = getUciToken(uciString, "SyzygyPath");
   const char *token2 = getUciToken(uciString, "tablebases");
   const char *token3 = getUciToken(uciString, "name");
   const char *token4 = getUciToken(uciString, "value");
   const char *token5 = getUciToken(trickyUciString, "tablebases");

   assert(strstr(token1, "SyzygyPath") == token1);
   assert(token2 == 0);
   assert(strstr(token3, "name") == token3);

   getNextUciToken(token3 + 4, buffer, sizeof(buffer));
   assert(strcmp(buffer, "SyzygyPath") == 0);

   getNextUciToken(token4 + 5, buffer, sizeof(buffer));
   assert(strcmp(buffer, "C:\\chess\\tablebases") == 0);

   assert(getLongUciValue(uciString, "time", 0) == 641273423);

   assert(strstr(trickyUciString, "423 tablebases") == token5 - 4);

   getUciNamedValue(uciString, name, sizeof(name), value, sizeof(value));
   assert(strcmp(name, "SyzygyPath") == 0);
   assert(strcmp(value, "C:\\chess\\tablebases") == 0);

   getUciNamedValue(trickyUciString, name, sizeof(name), value, sizeof(value));
   assert(strcmp(name, "SyzygyPathvalue") == 0);
   assert(strcmp(value, "C:\\chess\\tablebases") == 0);

   return 0;
}

static int testHashUpdate(void)
{
   Hashtable *hashtable = getSharedHashtable();
   Hashentry *tableHit;
#if defined(_WIN32) || defined(_WIN64)
   const char *transEntryStringFormat = "settransentry key %I64x data %I64x";
#else
   const char *transEntryStringFormat = "settransentry key %llx data %llx";
#endif
   char commandBuffer[256];
   UINT64 hashKey = 15021965ull;
   INT16 value = 2004;
   INT16 staticValue = 2008;
   UINT8 importance = 12;
   UINT16 bestMove = NO_MOVE;
   UINT8 date = 12;
   UINT8 flag = HASHVALUE_EXACT;
   Hashentry entry =
      constructHashEntry(hashKey, value, staticValue, importance,
                         bestMove, date, flag);

   snprintf(commandBuffer, sizeof(commandBuffer), transEntryStringFormat, entry.key, entry.data);
   processUciCommand(commandBuffer);
   tableHit = getHashentry(hashtable, hashKey);

   assert(tableHit != 0);
   assert(getHashentryValue(tableHit) == value);
   assert(getHashentryStaticValue(tableHit) == staticValue);
   assert(getHashentryImportance(tableHit) == importance);
   assert(getHashentryMove(tableHit) == bestMove);
   assert(getHashentryDate(tableHit) == hashtable->date);
   assert(getHashentryFlag(tableHit) == flag);

   return 0;
}

#endif

/******************************************************************************
 *
 * (See the header file comment for this function.)
 *
 ******************************************************************************/
int testModuleUci(void)
{
#ifndef NDEBUG
   int result;

   if ((result = testParameterParsing()) != 0)
   {
      return result;
   }

   if ((result = testTimeCalculation()) != 0)
   {
      return result;
   }

   if ((result = testUciTokenizer()) != 0)
   {
      return result;
   }

   if ((result = testHashUpdate()) != 0)
   {
      return result;
   }
#endif

   return 0;
}


#ifndef NNUE_H
#define NNUE_H

#include "protector.h"

// Small network architecture constants
#define L1_SMALL 128
#define L2_SMALL 15
#define L3_SMALL 32

// Big network architecture constants
#define L1_BIG 1024
#define L2_BIG 31
#define L3_BIG 32

#define LAYER_STACKS 8

typedef struct Position Position;

#define MAX_ACTIVE_FEATURES 32

typedef struct __attribute__((aligned(64))) {
    int16_t small_v[2][L1_SMALL];
    int16_t big_v[2][L1_BIG];
    int16_t big_threat_v[2][L1_BIG];
    int32_t small_psqtAccumulation[2][8];
    int32_t big_psqtAccumulation[2][8];
    int32_t big_threat_psqtAccumulation[2][8];
} Accumulator;

typedef struct {
    int dummy;
} NNUE;

/* Finny table: one cached accumulator entry per (color, king_square) pair.
   Caches the non-threat L1 vectors so refreshes can update incrementally
   instead of recomputing from scratch. */
typedef struct {
    Piece piece[64];           /* board state that produced this entry   */
    int16_t small_v[L1_SMALL]; /* small-net L1 for this perspective      */
    int16_t big_v[L1_BIG];     /* big-net L1 for this perspective        */
    int32_t small_psqt[8];
    int32_t big_psqt[8];
    bool valid; /* false until first computed             */
} FinnyEntry;

typedef struct {
    FinnyEntry entry[2][64]; /* [color][king_square] */
} FinnyTable;

int initializeModuleNnue(void);
int loadNnue(const char *filename);
int evaluateNnueWithAccumulator(Position *pos, Accumulator *acc);
int evaluateBigNnueWithAccumulator(Position *pos, Accumulator *acc);
void evaluateNnueWithAccumulatorFull(Position *pos, Accumulator *acc, int *psqt, int *positional);
void evaluateBigNnueWithAccumulatorFull(Position *pos, Accumulator *acc, int *psqt, int *positional);

int win_rate_scaling(Position *pos);

void resetFinnyTable(FinnyTable *finny);
void refreshAccumulator(Position *pos, Accumulator *acc, FinnyTable *finny);
void updateAccumulator(const Accumulator *prev, Accumulator *next, int added_count, Square *added_sq, Piece *added_pc,
                       int removed_count, Square *removed_sq, Piece *removed_pc, Square *ksq, Position *pos,
                       FinnyTable *finny);
bool kingStaysInSameBucket(Square from, Square to, Color color);

#endif

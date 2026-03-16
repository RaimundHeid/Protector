
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

#include "position_struct.h"

#define MAX_ACTIVE_FEATURES 32

typedef struct {
    int16_t small_v[2][L1_SMALL];
    int16_t big_v[2][L1_BIG];
    int32_t small_psqtAccumulation[2][8];
    int32_t big_psqtAccumulation[2][8];
} Accumulator;

typedef struct {
    int dummy;
} NNUE;

int initializeModuleNnue(void);
int loadNnue(const char* filename);
int evaluateNnueWithAccumulator(Position* pos, Accumulator* acc);

void refreshAccumulator(Position* pos, Accumulator* acc);
void updateAccumulator(Accumulator* prev, Accumulator* next, int added_count, Square* added_sq, Piece* added_pc, int removed_count, Square* removed_sq, Piece* removed_pc, Square* ksq);

#endif

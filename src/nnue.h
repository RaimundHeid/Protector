
#ifndef NNUE_H
#define NNUE_H

#include "protector.h"

// Small network architecture constants
#define L1 128
#define L2 15
#define L3 32

#include "position_struct.h"

#define MAX_ACTIVE_FEATURES 32

typedef struct {
    int16_t v[2][L1];
} Accumulator;

typedef struct {
    int dummy;
    // Accumulator and layers
} NNUE;

int initializeModuleNnue(void);
int loadNnue(const char* filename);
int evaluateNnue(Position* pos);

void refreshAccumulator(Position* pos, Accumulator* acc);
void updateAccumulator(Accumulator* prev, Accumulator* next, int added_count, Square* added_sq, Piece* added_pc, int removed_count, Square* removed_sq, Piece* removed_pc, Square* ksq);

#endif

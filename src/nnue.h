
#ifndef NNUE_H
#define NNUE_H

#include "protector.h"
#include "position.h"

#define MAX_ACTIVE_FEATURES 32

typedef int16_t WeightType;
typedef int32_t PSQTWeightType;
typedef int16_t BiasType;

// Small network architecture constants
#define L1 128
#define L2 15
#define L3 32

#define PSQT_BUCKETS 8
#define LAYER_STACKS 8

typedef struct {
    WeightType weights[L1 * 2 * PSQT_BUCKETS * 11 * 64 / 2]; // Simplified, needs adjustment
    // Actually FeatureTransformer is separate
} FeatureTransformer;

typedef struct {
    int dummy;
    // Accumulator and layers
} NNUE;

int initializeModuleNnue(void);
int loadNnue(const char* filename);
int evaluateNnue(Position* pos);

#endif

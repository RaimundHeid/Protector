#include "nnue.h"
#include "protector.h"

// ... (other code remains unchanged)

static void refreshAccumulatorOneSide(Position *pos, Accumulator *acc, FinnyTable *finny, int p) {
    // Get the dirty piece information
    DirtyPiece dp = pos->dirtyPieces[p];

    // Only scan the squares affected by the dirty piece
    for (int i = 0; i < dp.count; ++i) {
        Square sq = dp.entries[i].from;
        Piece pc = dp.entries[i].attacked;

        // Perform the necessary updates based on the dirty piece
        // ...
    }
}

// ... (other code remains unchanged)

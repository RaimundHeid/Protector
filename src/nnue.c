#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nnue.h"
#include "tools.h"
#include "position.h"
#include "io.h"

// Constants from Stockfish
static const uint32_t NNUE_VERSION = 0x7AF32F20u;

#define FT_INPUT_DIMENSIONS (64 * 11 * 64 / 2) // 22528

// Feature transformer data
static int16_t ft_biases[L1];
static int16_t ft_weights[L1 * FT_INPUT_DIMENSIONS]; 
static int32_t ft_psqt_weights[FT_INPUT_DIMENSIONS * 8];

// Network layers (8 stacks)
static int32_t fc0_biases[LAYER_STACKS][L2 + 1];
static int8_t  fc0_weights[LAYER_STACKS][(L2 + 1) * L1]; 

static int32_t fc1_biases[LAYER_STACKS][L3];
static int8_t  fc1_weights[LAYER_STACKS][L3 * 32]; // Padded from 30 to 32

static int32_t fc2_biases[LAYER_STACKS][1];
static int8_t  fc2_weights[LAYER_STACKS][1 * L3];

static int nnue_loaded = 0;

static const Square OrientTBL[64] = {
  H1, H1, H1, H1, A1, A1, A1, A1,
  H1, H1, H1, H1, A1, A1, A1, A1,
  H1, H1, H1, H1, A1, A1, A1, A1,
  H1, H1, H1, H1, A1, A1, A1, A1,
  H1, H1, H1, H1, A1, A1, A1, A1,
  H1, H1, H1, H1, A1, A1, A1, A1,
  H1, H1, H1, H1, A1, A1, A1, A1,
  H1, H1, H1, H1, A1, A1, A1, A1
};

#define B(v) (v * 11 * 64)
static const int KingBuckets[64] = {
    B(28), B(29), B(30), B(31), B(31), B(30), B(29), B(28),
    B(24), B(25), B(26), B(27), B(27), B(26), B(25), B(24),
    B(20), B(21), B(22), B(23), B(23), B(22), B(21), B(20),
    B(16), B(17), B(18), B(19), B(19), B(18), B(17), B(16),
    B(12), B(13), B(14), B(15), B(15), B(14), B(13), B(12),
    B( 8), B( 9), B(10), B(11), B(11), B(10), B( 9), B( 8),
    B( 4), B( 5), B( 6), B( 7), B( 7), B( 6), B( 5), B( 4),
    B( 0), B( 1), B( 2), B( 3), B( 3), B( 2), B( 1), B( 0),
};
#undef B

static int get_piece_square_index(Piece pc, Color perspective) {
    int type = pieceType(pc);
    int color = pieceColor(pc);
    if (type == KING) return 10 * 64;
    int sf_type;
    if (color == perspective) {
        if (type == PAWN) sf_type = 0;
        else if (type == KNIGHT) sf_type = 2;
        else if (type == BISHOP) sf_type = 4;
        else if (type == ROOK) sf_type = 6;
        else sf_type = 8;
    } else {
        if (type == PAWN) sf_type = 1;
        else if (type == KNIGHT) sf_type = 3;
        else if (type == BISHOP) sf_type = 5;
        else if (type == ROOK) sf_type = 7;
        else sf_type = 9;
    }
    return sf_type * 64;
}

static int get_feature_index(Square s, Piece pc, Square ksq, Color perspective) {
    const int flip = 56 * perspective;
    return (int)(s ^ OrientTBL[ksq] ^ flip) + get_piece_square_index(pc, perspective)
         + KingBuckets[ksq ^ flip];
}

void refreshAccumulator(Position* pos, Accumulator* acc) {
    for (int p = 0; p < 2; p++) {
        memcpy(acc->v[p], ft_biases, sizeof(int16_t) * L1);
        Square ksq = pos->king[p];
        for (Square s = 0; s < 64; s++) {
            Piece pc = pos->piece[s];
            if (pc != NO_PIECE) {
                int idx = get_feature_index(s, pc, ksq, p);
                for (int i = 0; i < L1; i++) {
                    acc->v[p][i] += ft_weights[idx * L1 + i];
                }
            }
        }
    }
}

void updateAccumulator(Accumulator* prev, Accumulator* next, int added_count, Square* added_sq, Piece* added_pc, int removed_count, Square* removed_sq, Piece* removed_pc, Square* ksq) {
    for (int p = 0; p < 2; p++) {
        memcpy(next->v[p], prev->v[p], sizeof(int16_t) * L1);
        for (int j = 0; j < removed_count; j++) {
            int idx = get_feature_index(removed_sq[j], removed_pc[j], ksq[p], p);
            for (int i = 0; i < L1; i++) {
                next->v[p][i] -= ft_weights[idx * L1 + i];
            }
        }
        for (int j = 0; j < added_count; j++) {
            int idx = get_feature_index(added_sq[j], added_pc[j], ksq[p], p);
            for (int i = 0; i < L1; i++) {
                next->v[p][i] += ft_weights[idx * L1 + i];
            }
        }
    }
}

static void read_leb128(FILE* f, void* out, size_t count, size_t element_size) {
    char magic[18] = {0};
    if (fread(magic, 1, 17, f) != 17) return;
    uint32_t bytes_left;
    if (fread(&bytes_left, 4, 1, f) != 1) return;
    uint32_t total_bytes = bytes_left;
    for (size_t i = 0; i < count; i++) {
        int32_t result = 0;
        int shift = 0;
        while (1) {
            uint8_t byte;
            if (fread(&byte, 1, 1, f) != 1) return;
            bytes_left--;
            result |= (int32_t)(byte & 0x7f) << shift;
            shift += 7;
            if (!(byte & 0x80)) {
                if (shift < 32 && (byte & 0x40)) result |= ~((unsigned int)((1 << shift) - 1));
                if (element_size == 2) ((int16_t*)out)[i] = (int16_t)result;
                else if (element_size == 4) ((int32_t*)out)[i] = (int32_t)result;
                else if (element_size == 1) ((int8_t*)out)[i] = (int8_t)result;
                break;
            }
            if (bytes_left == 0 && (byte & 0x80)) return; // Safety break
        }
    }
    if (bytes_left > 0) fseek(f, bytes_left, SEEK_CUR);
}

int initializeModuleNnue(void) {
    return loadNnue("nn-47fc8b7fff06.nnue");
}

int loadNnue(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;
    
    uint32_t version, hash, desc_size;
    if (fread(&version, 4, 1, f) != 1 || version != NNUE_VERSION) { fclose(f); return -2; }
    fread(&hash, 4, 1, f);
    fread(&desc_size, 4, 1, f);
    fseek(f, desc_size, SEEK_CUR);
    
    uint32_t ft_hash;
    fread(&ft_hash, 4, 1, f);
    read_leb128(f, ft_biases, L1, 2);
    read_leb128(f, ft_weights, L1 * FT_INPUT_DIMENSIONS, 2);
    read_leb128(f, ft_psqt_weights, FT_INPUT_DIMENSIONS * 8, 4);
    
    for (int s = 0; s < LAYER_STACKS; s++) {
        uint32_t arch_hash;
        if (fread(&arch_hash, 4, 1, f) != 1) break;
        
        // FC0
        fread(fc0_biases[s], 4, L2 + 1, f);
        fread(fc0_weights[s], 1, (L2 + 1) * L1, f);
        
        // FC1
        fread(fc1_biases[s], 4, L3, f);
        fread(fc1_weights[s], 1, L3 * 32, f);
        
        // FC2
        fread(fc2_biases[s], 4, 1, f);
        fread(fc2_weights[s], 1, 1 * L3, f);
    }
    
    fclose(f);
    nnue_loaded = 1;
    return 0;
}

static inline int32_t clipped_relu(int32_t x) {
    return max(0, min(127, x));
}

static inline int32_t sqr_clipped_relu(int32_t x) {
    int32_t c = max(0, min(127, x));
    return (c * c) >> 7; // Scaled to fit (matching Stockfish >> 19 raw)
}

static int win_rate_scaling(Position* pos) {
    int material = getNumberOfSetSquares(pos->piecesOfType[WHITE_PAWN]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_PAWN])
                 + 3 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_KNIGHT]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_KNIGHT]))
                 + 3 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_BISHOP]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_BISHOP]))
                 + 5 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_ROOK]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_ROOK]))
                 + 9 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_QUEEN]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_QUEEN]));

    // The fitted model only uses data for material counts in [17, 78], and is anchored at count 58.
    double m = max(17.0, min(78.0, (double)material)) / 58.0;

    // Return a = p_a(material), see github.com/official-stockfish/WDL_model
    const double as[] = {-72.32565836, 185.93832038, -144.58862193, 416.44950446};
    double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];

    return (int)a;
}

int evaluateNnueWithAccumulator(Position* pos, Accumulator* acc) {
    if (!nnue_loaded) return 0;

    int num_pieces = 0;
    for (int i = 0; i < 16; i++) num_pieces += getNumberOfSetSquares(pos->piecesOfType[i]);
    int s = min(7, (num_pieces - 2) / 4);

    Color side = pos->activeColor;
    uint8_t transformed[L1]; // 64 per side
    for (int p = 0; p < 2; p++) {
        Color perspective = (p == 0 ? side : !side);
        for (int i = 0; i < L1 / 2; i++) {
            int16_t v0 = acc->v[perspective][i];
            int16_t v1 = acc->v[perspective][L1 / 2 + i];
            int32_t clipped0 = max(0, min(255, v0));
            int32_t clipped1 = max(0, min(255, v1));
            transformed[p * 64 + i] = (uint8_t)((clipped0 * clipped1) / 512);
        }
    }

    int32_t fc0_out[L2 + 1];
    for (int i = 0; i <= L2; i++) {
        fc0_out[i] = fc0_biases[s][i];
        for (int j = 0; j < L1; j++) {
            fc0_out[i] += (int32_t)transformed[j] * fc0_weights[s][i * L1 + j];
        }
    }

    int32_t ac0_out[32] = {0}; // Padded to 32
    for (int i = 0; i < L2; i++) {
        int32_t in = fc0_out[i] >> 6;
        ac0_out[i] = sqr_clipped_relu(in);
        ac0_out[L2 + i] = clipped_relu(in);
    }

    int32_t fc1_out[L3];
    for (int i = 0; i < L3; i++) {
        fc1_out[i] = fc1_biases[s][i];
        for (int j = 0; j < 30; j++) {
            fc1_out[i] += ac0_out[j] * fc1_weights[s][i * 32 + j];
        }
    }

    int32_t ac1_out[L3];
    for (int i = 0; i < L3; i++) {
        ac1_out[i] = clipped_relu(fc1_out[i] >> 6);
    }

    int32_t fc2_out = fc2_biases[s][0];
    for (int i = 0; i < L3; i++) {
        fc2_out += ac1_out[i] * fc2_weights[s][i]; // Only one output, so i * 1 + 0 is just i
    }

    int32_t fwdOut = (int32_t)((int64_t)fc0_out[L2] * (600 * 16) / (127 * 64));
    int32_t outputValue = fc2_out + fwdOut;

    // Internal value matching Stockfish search units
    int v = outputValue / 16;

    // Scale to centipawns using win rate model
    int a = win_rate_scaling(pos);
    return v * 100 / a;
}

int evaluateNnue(Position* pos, Accumulator* acc) {
    return evaluateNnueWithAccumulator(pos, acc);
}

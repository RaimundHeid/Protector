#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "nnue.h"
#include "tools.h"
#include "position.h"
#include "io.h"

// Constants from Stockfish
static const uint32_t NNUE_VERSION = 0x7AF32F20u;

#define FT_INPUT_DIMENSIONS (64 * 11 * 64 / 2) // 22528

// Embedding logic
#if defined(__APPLE__)
#define INCBIN(name, file) \
    __asm__(".text\n" \
            ".global _" #name "_data\n" \
            ".align 4\n" \
            "_" #name "_data:\n" \
            ".incbin \"" file "\"\n" \
            ".global _" #name "_end\n" \
            "_" #name "_end:\n" \
            ".byte 0\n"); \
    extern const uint8_t name ## _data[]; \
    extern const uint8_t name ## _end[];
#elif defined(__linux__)
#define INCBIN(name, file) \
    __asm__(".section .rodata\n" \
            ".global " #name "_data\n" \
            ".align 4\n" \
            #name "_data:\n" \
            ".incbin \"" file "\"\n" \
            ".global " #name "_end\n" \
            #name "_end:\n" \
            ".byte 0\n" \
            ".previous\n"); \
    extern const uint8_t name ## _data[]; \
    extern const uint8_t name ## _end[];
#else
// Fallback
#define INCBIN(name, file) \
    static const uint8_t name ## _data[1] = {0}; \
    static const uint8_t name ## _end[1] = {0};
#endif

INCBIN(nnue_model, "nn-47fc8b7fff06.nnue")

// Feature transformer data
static int16_t ft_biases[L1] __attribute__((aligned(64)));
static int16_t ft_weights[L1 * FT_INPUT_DIMENSIONS] __attribute__((aligned(64))); 
static int32_t ft_psqt_weights[FT_INPUT_DIMENSIONS * 8] __attribute__((aligned(64)));

// Network layers (8 stacks)
static int32_t fc0_biases[LAYER_STACKS][L2 + 1] __attribute__((aligned(64)));
static int8_t  fc0_weights[LAYER_STACKS][(L2 + 1) * L1] __attribute__((aligned(64))); 

static int32_t fc1_biases[LAYER_STACKS][L3] __attribute__((aligned(64)));
static int8_t  fc1_weights[LAYER_STACKS][L3 * 32] __attribute__((aligned(64))); 

static int32_t fc2_biases[LAYER_STACKS][1] __attribute__((aligned(64)));
static int8_t  fc2_weights[LAYER_STACKS][1 * L3] __attribute__((aligned(64)));

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
                const int16_t* weights = &ft_weights[idx * L1];
#if defined(__x86_64__)
                for (int i = 0; i < L1; i += 16) {
                    __m256i v = _mm256_load_si256((__m256i*)&acc->v[p][i]);
                    __m256i w = _mm256_load_si256((__m256i*)&weights[i]);
                    _mm256_store_si256((__m256i*)&acc->v[p][i], _mm256_add_epi16(v, w));
                }
#else
                for (int i = 0; i < L1; i++) {
                    acc->v[p][i] += weights[i];
                }
#endif
            }
        }
    }
}

void updateAccumulator(Accumulator* prev, Accumulator* next, int added_count, Square* added_sq, Piece* added_pc, int removed_count, Square* removed_sq, Piece* removed_pc, Square* ksq) {
    for (int p = 0; p < 2; p++) {
        memcpy(next->v[p], prev->v[p], sizeof(int16_t) * L1);
        for (int j = 0; j < removed_count; j++) {
            int idx = get_feature_index(removed_sq[j], removed_pc[j], ksq[p], p);
            const int16_t* weights = &ft_weights[idx * L1];
#if defined(__x86_64__)
            for (int i = 0; i < L1; i += 16) {
                __m256i v = _mm256_load_si256((__m256i*)&next->v[p][i]);
                __m256i w = _mm256_load_si256((__m256i*)&weights[i]);
                _mm256_store_si256((__m256i*)&next->v[p][i], _mm256_sub_epi16(v, w));
            }
#else
            for (int i = 0; i < L1; i++) {
                next->v[p][i] -= weights[i];
            }
#endif
        }
        for (int j = 0; j < added_count; j++) {
            int idx = get_feature_index(added_sq[j], added_pc[j], ksq[p], p);
            const int16_t* weights = &ft_weights[idx * L1];
#if defined(__x86_64__)
            for (int i = 0; i < L1; i += 16) {
                __m256i v = _mm256_load_si256((__m256i*)&next->v[p][i]);
                __m256i w = _mm256_load_si256((__m256i*)&weights[i]);
                _mm256_store_si256((__m256i*)&next->v[p][i], _mm256_add_epi16(v, w));
            }
#else
            for (int i = 0; i < L1; i++) {
                next->v[p][i] += weights[i];
            }
#endif
        }
    }
}

static const uint8_t* read_data;
static size_t read_pos;

static void mem_read(void* dest, size_t size) {
    memcpy(dest, &read_data[read_pos], size);
    read_pos += size;
}

static void mem_skip(size_t size) {
    read_pos += size;
}

static void read_leb128_mem(void* out, size_t count, size_t element_size) {
    mem_skip(17); // magic
    uint32_t bytes_left;
    mem_read(&bytes_left, 4);
    size_t end_pos = read_pos + bytes_left;
    for (size_t i = 0; i < count; i++) {
        int32_t result = 0;
        int shift = 0;
        while (1) {
            uint8_t byte = read_data[read_pos++];
            result |= (int32_t)(byte & 0x7f) << shift;
            shift += 7;
            if (!(byte & 0x80)) {
                if (shift < 32 && (byte & 0x40)) result |= ~((unsigned int)((1 << shift) - 1));
                if (element_size == 2) ((int16_t*)out)[i] = (int16_t)result;
                else if (element_size == 4) ((int32_t*)out)[i] = (int32_t)result;
                else if (element_size == 1) ((int8_t*)out)[i] = (int8_t)result;
                break;
            }
        }
    }
    read_pos = end_pos;
}

int initializeModuleNnue(void) {
    read_data = nnue_model_data;
    read_pos = 0;
    
    // Simple check to see if we have data
    if (read_data[0] == 0 && (size_t)(nnue_model_end - nnue_model_data) <= 1) return -1;

    uint32_t version, hash, desc_size;
    mem_read(&version, 4);
    if (version != NNUE_VERSION) return -2;
    mem_read(&hash, 4);
    mem_read(&desc_size, 4);
    mem_skip(desc_size);
    
    uint32_t ft_hash;
    mem_read(&ft_hash, 4);
    read_leb128_mem(ft_biases, L1, 2);
    read_leb128_mem(ft_weights, L1 * FT_INPUT_DIMENSIONS, 2);
    read_leb128_mem(ft_psqt_weights, FT_INPUT_DIMENSIONS * 8, 4);
    
    for (int s = 0; s < LAYER_STACKS; s++) {
        uint32_t arch_hash;
        mem_read(&arch_hash, 4);
        
        // FC0
        mem_read(fc0_biases[s], 4 * (L2 + 1));
        mem_read(fc0_weights[s], (L2 + 1) * L1);
        
        // FC1
        mem_read(fc1_biases[s], 4 * L3);
        mem_read(fc1_weights[s], L3 * 32);
        
        // FC2
        mem_read(fc2_biases[s], 4 * 1);
        mem_read(fc2_weights[s], 1 * L3);
    }
    
    nnue_loaded = 1;
    return 0;
}

static inline int32_t clipped_relu(int32_t x) {
    return max(0, min(127, x));
}

static inline int32_t sqr_clipped_relu(int32_t x) {
    int32_t c = max(0, min(127, x));
    return (c * c) >> 7; 
}

static int win_rate_scaling(Position* pos) {
    int material = getNumberOfSetSquares(pos->piecesOfType[WHITE_PAWN]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_PAWN])
                 + 3 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_KNIGHT]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_KNIGHT]))
                 + 3 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_BISHOP]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_BISHOP]))
                 + 5 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_ROOK]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_ROOK]))
                 + 9 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_QUEEN]) + getNumberOfSetSquares(pos->piecesOfType[BLACK_QUEEN]));

    double m = max(17.0, min(78.0, (double)material)) / 58.0;
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
    uint8_t transformed[L1] __attribute__((aligned(64))); 
    for (int p = 0; p < 2; p++) {
        Color perspective = (p == 0 ? side : !side);
#if defined(__x86_64__)
        for (int i = 0; i < L1 / 2; i += 16) {
            __m256i v0 = _mm256_load_si256((__m256i*)&acc->v[perspective][i]);
            __m256i v1 = _mm256_load_si256((__m256i*)&acc->v[perspective][L1 / 2 + i]);
            
            __m256i zero = _mm256_setzero_si256();
            __m256i high = _mm256_set1_epi16(255);
            
            __m256i c0 = _mm256_min_epi16(_mm256_max_epi16(v0, zero), high);
            __m256i c1 = _mm256_min_epi16(_mm256_max_epi16(v1, zero), high);
            
            __m256i prod = _mm256_mullo_epi16(c0, c1);
            __m256i shifted = _mm256_srli_epi16(prod, 9);
            
            __m256i packed = _mm256_packus_epi16(shifted, shifted);
            packed = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
            
            _mm_store_si128((__m128i*)&transformed[p * 64 + i], _mm256_extracti128_si256(packed, 0));
        }
#else
        for (int i = 0; i < L1 / 2; i++) {
            int16_t v0 = acc->v[perspective][i];
            int16_t v1 = acc->v[perspective][L1 / 2 + i];
            int32_t c0 = max(0, min(255, v0));
            int32_t c1 = max(0, min(255, v1));
            transformed[p * 64 + i] = (uint8_t)((c0 * c1) / 512);
        }
#endif
    }

    int32_t fc0_out[L2 + 1];
    for (int i = 0; i <= L2; i++) {
#if defined(__x86_64__)
        __m256i sum = _mm256_setzero_si256();
        const int8_t* weights = &fc0_weights[s][i * L1];
        for (int j = 0; j < L1; j += 32) {
            __m256i t = _mm256_load_si256((__m256i*)&transformed[j]);
            __m256i w = _mm256_load_si256((__m256i*)&weights[j]);
            
            __m256i mad = _mm256_maddubs_epi16(t, w);
            __m256i ones = _mm256_set1_epi16(1);
            __m256i res32 = _mm256_madd_epi16(mad, ones);
            sum = _mm256_add_epi32(sum, res32);
        }
        
        __m128i sum128 = _mm_add_epi32(_mm256_extracti128_si256(sum, 0), _mm256_extracti128_si256(sum, 1));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(0, 0, 0, 1)));
        
        fc0_out[i] = fc0_biases[s][i] + _mm_cvtsi128_si32(sum128);
#else
        fc0_out[i] = fc0_biases[s][i];
        for (int j = 0; j < L1; j++) {
            fc0_out[i] += (int32_t)transformed[j] * fc0_weights[s][i * L1 + j];
        }
#endif
    }

    int32_t ac0_out[32] = {0}; 
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
        fc2_out += ac1_out[i] * fc2_weights[s][i];
    }

    int32_t fwdOut = (int32_t)((int64_t)fc0_out[L2] * (600 * 16) / (127 * 64));
    int32_t outputValue = fc2_out + fwdOut;
    int v = outputValue / 16;
    int a = win_rate_scaling(pos);
    return v * 100 / a;
}

int evaluateNnue(Position* pos, Accumulator* acc) {
    return evaluateNnueWithAccumulator(pos, acc);
}

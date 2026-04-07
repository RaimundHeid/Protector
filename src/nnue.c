#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "io.h"
#include "nnue.h"
#include "position.h"
#include "tools.h"

// ===== SIMD helpers for L1 accumulator operations =====

#if defined(__AVX2__)

static inline void add_weights_int16(int16_t *restrict acc, const int16_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i *)(acc + i));
        __m256i b = _mm256_load_si256((const __m256i *)(w + i));
        _mm256_store_si256((__m256i *)(acc + i), _mm256_add_epi16(a, b));
    }
}

static inline void sub_weights_int16(int16_t *restrict acc, const int16_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i *)(acc + i));
        __m256i b = _mm256_load_si256((const __m256i *)(w + i));
        _mm256_store_si256((__m256i *)(acc + i), _mm256_sub_epi16(a, b));
    }
}

static inline void add_weights_int8_to_int16(int16_t *restrict acc, const int8_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i *)(acc + i));
        __m128i b8 = _mm_load_si128((const __m128i *)(w + i));
        _mm256_store_si256((__m256i *)(acc + i), _mm256_add_epi16(a, _mm256_cvtepi8_epi16(b8)));
    }
}

static inline void sub_weights_int8_to_int16(int16_t *restrict acc, const int8_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 16) {
        __m256i a = _mm256_load_si256((const __m256i *)(acc + i));
        __m128i b8 = _mm_load_si128((const __m128i *)(w + i));
        _mm256_store_si256((__m256i *)(acc + i), _mm256_sub_epi16(a, _mm256_cvtepi8_epi16(b8)));
    }
}

static void transform_small(const int16_t *restrict acc0, const int16_t *restrict acc1, uint8_t *restrict out)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max255 = _mm256_set1_epi16(255);
    const int16_t *sides[2] = {acc0, acc1};
    for (int p = 0; p < 2; p++) {
        const int16_t *acc = sides[p];
        for (int i = 0; i < L1_SMALL / 2; i += 16) {
            __m256i v0 =
                _mm256_min_epi16(_mm256_max_epi16(_mm256_load_si256((const __m256i *)(acc + i)), zero), max255);
            __m256i v1 = _mm256_min_epi16(
                _mm256_max_epi16(_mm256_load_si256((const __m256i *)(acc + L1_SMALL / 2 + i)), zero), max255);
            __m256i prod = _mm256_srli_epi16(_mm256_mullo_epi16(v0, v1), 9);
            __m128i lo = _mm256_castsi256_si128(prod);
            __m128i hi = _mm256_extracti128_si256(prod, 1);
            _mm_store_si128((__m128i *)(out + p * (L1_SMALL / 2) + i), _mm_packus_epi16(lo, hi));
        }
    }
}

static void transform_big(const int16_t *restrict bv0, const int16_t *restrict tv0, const int16_t *restrict bv1,
                          const int16_t *restrict tv1, uint8_t *restrict out)
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i max255 = _mm256_set1_epi16(255);
    const int16_t *bvsides[2] = {bv0, bv1};
    const int16_t *tvsides[2] = {tv0, tv1};
    for (int p = 0; p < 2; p++) {
        const int16_t *bv = bvsides[p];
        const int16_t *tv = tvsides[p];
        for (int i = 0; i < L1_BIG / 2; i += 16) {
            __m256i v0 =
                _mm256_min_epi16(_mm256_max_epi16(_mm256_add_epi16(_mm256_load_si256((const __m256i *)(bv + i)),
                                                                   _mm256_load_si256((const __m256i *)(tv + i))),
                                                  zero),
                                 max255);
            __m256i v1 = _mm256_min_epi16(
                _mm256_max_epi16(_mm256_add_epi16(_mm256_load_si256((const __m256i *)(bv + L1_BIG / 2 + i)),
                                                  _mm256_load_si256((const __m256i *)(tv + L1_BIG / 2 + i))),
                                 zero),
                max255);
            __m256i prod = _mm256_srli_epi16(_mm256_mullo_epi16(v0, v1), 9);
            __m128i lo = _mm256_castsi256_si128(prod);
            __m128i hi = _mm256_extracti128_si256(prod, 1);
            _mm_store_si128((__m128i *)(out + p * (L1_BIG / 2) + i), _mm_packus_epi16(lo, hi));
        }
    }
}

// L2 forward pass helpers (AVX2): uint8 input × int8 weights → int32 output
// Transformed values are in [0,127] so uint8 × int8 uses _mm256_maddubs_epi16 safely.
// n must be a multiple of 32; input and weights must be 32-byte aligned.
static inline int32_t dot_u8s8(const uint8_t *restrict a, const int8_t *restrict b, int n)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 32) {
        __m256i va = _mm256_load_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_load_si256((const __m256i *)(b + i));
        __m256i p = _mm256_maddubs_epi16(va, vb);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, ones));
    }
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

static void fc_u8s8(int32_t *restrict out, const int32_t *restrict bias, const uint8_t *restrict input,
                    const int8_t *restrict weights, int n_in, int n_out)
{
    for (int i = 0; i < n_out; i++)
        out[i] = bias[i] + dot_u8s8(input, weights + i * n_in, n_in);
}

#elif defined(__ARM_NEON)

static inline void add_weights_int16(int16_t *restrict acc, const int16_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 8)
        vst1q_s16(acc + i, vaddq_s16(vld1q_s16(acc + i), vld1q_s16(w + i)));
}

static inline void sub_weights_int16(int16_t *restrict acc, const int16_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 8)
        vst1q_s16(acc + i, vsubq_s16(vld1q_s16(acc + i), vld1q_s16(w + i)));
}

static inline void add_weights_int8_to_int16(int16_t *restrict acc, const int8_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 16) {
        int8x16_t b = vld1q_s8(w + i);
        vst1q_s16(acc + i, vaddq_s16(vld1q_s16(acc + i), vmovl_s8(vget_low_s8(b))));
        vst1q_s16(acc + i + 8, vaddq_s16(vld1q_s16(acc + i + 8), vmovl_s8(vget_high_s8(b))));
    }
}

static inline void sub_weights_int8_to_int16(int16_t *restrict acc, const int8_t *restrict w, int n)
{
    for (int i = 0; i < n; i += 16) {
        int8x16_t b = vld1q_s8(w + i);
        vst1q_s16(acc + i, vsubq_s16(vld1q_s16(acc + i), vmovl_s8(vget_low_s8(b))));
        vst1q_s16(acc + i + 8, vsubq_s16(vld1q_s16(acc + i + 8), vmovl_s8(vget_high_s8(b))));
    }
}

static void transform_small(const int16_t *restrict acc0, const int16_t *restrict acc1, uint8_t *restrict out)
{
    const int16x8_t vmax = vdupq_n_s16(255);
    const int16x8_t vzero = vdupq_n_s16(0);
    const int16_t *sides[2] = {acc0, acc1};
    for (int p = 0; p < 2; p++) {
        const int16_t *acc = sides[p];
        for (int i = 0; i < L1_SMALL / 2; i += 8) {
            int16x8_t v0 = vminq_s16(vmaxq_s16(vld1q_s16(acc + i), vzero), vmax);
            int16x8_t v1 = vminq_s16(vmaxq_s16(vld1q_s16(acc + L1_SMALL / 2 + i), vzero), vmax);
            uint16x8_t prod = vshrq_n_u16(vmulq_u16(vreinterpretq_u16_s16(v0), vreinterpretq_u16_s16(v1)), 9);
            vst1_u8(out + p * (L1_SMALL / 2) + i, vmovn_u16(prod));
        }
    }
}

static void transform_big(const int16_t *restrict bv0, const int16_t *restrict tv0, const int16_t *restrict bv1,
                          const int16_t *restrict tv1, uint8_t *restrict out)
{
    const int16x8_t vmax = vdupq_n_s16(255);
    const int16x8_t vzero = vdupq_n_s16(0);
    const int16_t *bvsides[2] = {bv0, bv1};
    const int16_t *tvsides[2] = {tv0, tv1};
    for (int p = 0; p < 2; p++) {
        const int16_t *bv = bvsides[p];
        const int16_t *tv = tvsides[p];
        for (int i = 0; i < L1_BIG / 2; i += 8) {
            int16x8_t v0 = vminq_s16(vmaxq_s16(vaddq_s16(vld1q_s16(bv + i), vld1q_s16(tv + i)), vzero), vmax);
            int16x8_t v1 = vminq_s16(
                vmaxq_s16(vaddq_s16(vld1q_s16(bv + L1_BIG / 2 + i), vld1q_s16(tv + L1_BIG / 2 + i)), vzero), vmax);
            uint16x8_t prod = vshrq_n_u16(vmulq_u16(vreinterpretq_u16_s16(v0), vreinterpretq_u16_s16(v1)), 9);
            vst1_u8(out + p * (L1_BIG / 2) + i, vmovn_u16(prod));
        }
    }
}

// L2 forward pass helpers (NEON): uint8 input × int8 weights → int32 output
// Transformed values are in [0,127], so reinterpreting uint8 as int8 is safe.
// n must be a multiple of 16.
static inline int32_t dot_u8s8(const uint8_t *restrict a, const int8_t *restrict b, int n)
{
    int32x4_t acc = vdupq_n_s32(0);
    for (int i = 0; i < n; i += 16) {
        int8x8_t va_lo = vreinterpret_s8_u8(vld1_u8(a + i));
        int8x8_t va_hi = vreinterpret_s8_u8(vld1_u8(a + i + 8));
        int8x8_t vb_lo = vld1_s8(b + i);
        int8x8_t vb_hi = vld1_s8(b + i + 8);
        acc = vpadalq_s16(acc, vmull_s8(va_lo, vb_lo));
        acc = vpadalq_s16(acc, vmull_s8(va_hi, vb_hi));
    }
    int32x2_t s2 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    return vget_lane_s32(vpadd_s32(s2, s2), 0);
}

static void fc_u8s8(int32_t *restrict out, const int32_t *restrict bias, const uint8_t *restrict input,
                    const int8_t *restrict weights, int n_in, int n_out)
{
    for (int i = 0; i < n_out; i++)
        out[i] = bias[i] + dot_u8s8(input, weights + i * n_in, n_in);
}

#else

static inline void add_weights_int16(int16_t *restrict acc, const int16_t *restrict w, int n)
{
    for (int i = 0; i < n; i++)
        acc[i] += w[i];
}

static inline void sub_weights_int16(int16_t *restrict acc, const int16_t *restrict w, int n)
{
    for (int i = 0; i < n; i++)
        acc[i] -= w[i];
}

static inline void add_weights_int8_to_int16(int16_t *restrict acc, const int8_t *restrict w, int n)
{
    for (int i = 0; i < n; i++)
        acc[i] += w[i];
}

static inline void sub_weights_int8_to_int16(int16_t *restrict acc, const int8_t *restrict w, int n)
{
    for (int i = 0; i < n; i++)
        acc[i] -= w[i];
}

static void transform_small(const int16_t *restrict acc0, const int16_t *restrict acc1, uint8_t *restrict out)
{
    const int16_t *sides[2] = {acc0, acc1};
    for (int p = 0; p < 2; p++) {
        const int16_t *acc = sides[p];
        for (int i = 0; i < L1_SMALL / 2; i++) {
            int32_t c0 = max(0, min(255, (int32_t)acc[i]));
            int32_t c1 = max(0, min(255, (int32_t)acc[L1_SMALL / 2 + i]));
            out[p * (L1_SMALL / 2) + i] = (uint8_t)((c0 * c1) / 512);
        }
    }
}

static void transform_big(const int16_t *restrict bv0, const int16_t *restrict tv0, const int16_t *restrict bv1,
                          const int16_t *restrict tv1, uint8_t *restrict out)
{
    const int16_t *bvsides[2] = {bv0, bv1};
    const int16_t *tvsides[2] = {tv0, tv1};
    for (int p = 0; p < 2; p++) {
        const int16_t *bv = bvsides[p];
        const int16_t *tv = tvsides[p];
        for (int i = 0; i < L1_BIG / 2; i++) {
            int32_t v0 = (int32_t)bv[i] + tv[i];
            int32_t v1 = (int32_t)bv[L1_BIG / 2 + i] + tv[L1_BIG / 2 + i];
            int32_t c0 = max(0, min(255, v0));
            int32_t c1 = max(0, min(255, v1));
            out[p * (L1_BIG / 2) + i] = (uint8_t)((c0 * c1) / 512);
        }
    }
}

// L2 forward pass helpers (scalar fallback)
static inline int32_t dot_u8s8(const uint8_t *restrict a, const int8_t *restrict b, int n)
{
    int32_t s = 0;
    for (int i = 0; i < n; i++)
        s += (int32_t)a[i] * b[i];
    return s;
}

static void fc_u8s8(int32_t *restrict out, const int32_t *restrict bias, const uint8_t *restrict input,
                    const int8_t *restrict weights, int n_in, int n_out)
{
    for (int i = 0; i < n_out; i++) {
        int32_t s = bias[i];
        for (int j = 0; j < n_in; j++)
            s += (int32_t)input[j] * weights[i * n_in + j];
        out[i] = s;
    }
}

#endif

// ===== End SIMD helpers =====

// Constants from Stockfish
static const uint32_t NNUE_VERSION = 0x7AF32F20u;
#define LEB128_MAGIC_STRING_SIZE (sizeof("COMPRESSED_LEB128") - 1)

#define FT_INPUT_DIMENSIONS (64 * 11 * 64 / 2) // 22528
#define THREAT_INPUT_DIMENSIONS 60144

// Embedding logic
#if defined(__APPLE__)
#define INCBIN(name, file)                                                                                             \
    __asm__(".text\n"                                                                                                  \
            ".global _" #name "_data\n"                                                                                \
            ".align 4\n"                                                                                               \
            "_" #name "_data:\n"                                                                                       \
            ".incbin \"" file "\"\n"                                                                                   \
            ".global _" #name "_end\n"                                                                                 \
            "_" #name "_end:\n"                                                                                        \
            ".byte 0\n");                                                                                              \
    extern const uint8_t name##_data[];                                                                                \
    extern const uint8_t name##_end[];
#elif defined(__linux__)
#define INCBIN(name, file)                                                                                             \
    __asm__(".section .rodata\n"                                                                                       \
            ".global " #name "_data\n"                                                                                 \
            ".align 4\n" #name "_data:\n"                                                                              \
            ".incbin \"" file "\"\n"                                                                                   \
            ".global " #name "_end\n" #name "_end:\n"                                                                  \
            ".byte 0\n"                                                                                                \
            ".previous\n");                                                                                            \
    extern const uint8_t name##_data[];                                                                                \
    extern const uint8_t name##_end[];
#else
// Fallback
#define INCBIN(name, file)                                                                                             \
    static const uint8_t name##_data[1] = {0};                                                                         \
    static const uint8_t name##_end[1] = {0};
#endif

INCBIN(small_nnue_model, "nn-47fc8b7fff06.nnue")
INCBIN(big_nnue_model, "nn-9a0cc2a62c52.nnue")

// Small Feature transformer data
static int16_t small_ft_biases[L1_SMALL] __attribute__((aligned(64)));
static int16_t small_ft_weights[L1_SMALL * FT_INPUT_DIMENSIONS] __attribute__((aligned(64)));
static int32_t small_ft_psqt_weights[FT_INPUT_DIMENSIONS * 8] __attribute__((aligned(64)));

// Small Network layers (8 stacks)
static int32_t small_fc0_biases[LAYER_STACKS][L2_SMALL + 1] __attribute__((aligned(64)));
static int8_t small_fc0_weights[LAYER_STACKS][(L2_SMALL + 1) * L1_SMALL] __attribute__((aligned(64)));

static int32_t small_fc1_biases[LAYER_STACKS][L3_SMALL] __attribute__((aligned(64)));
static int8_t small_fc1_weights[LAYER_STACKS][L3_SMALL * 32] __attribute__((aligned(64)));

static int32_t small_fc2_biases[LAYER_STACKS][1] __attribute__((aligned(64)));
static int8_t small_fc2_weights[LAYER_STACKS][1 * L3_SMALL] __attribute__((aligned(64)));

// Big Feature transformer data
static int16_t big_ft_biases[L1_BIG] __attribute__((aligned(64)));
static int16_t big_ft_weights[L1_BIG * FT_INPUT_DIMENSIONS] __attribute__((aligned(64)));
static int32_t big_ft_psqt_weights[FT_INPUT_DIMENSIONS * 8] __attribute__((aligned(64)));

static int8_t big_ft_threat_weights[L1_BIG * THREAT_INPUT_DIMENSIONS] __attribute__((aligned(64)));
static int32_t big_ft_threat_psqt_weights[THREAT_INPUT_DIMENSIONS * 8] __attribute__((aligned(64)));

typedef struct {
    int cumulativePieceOffset, cumulativeOffset;
} HelperOffsets;

static HelperOffsets helper_offsets[16];
static uint32_t threat_offsets[16][64];
static uint32_t index_lut1[16][16][2];
static uint8_t index_lut2[16][64][64];

// Mapping from piece to Stockfish internal piece types
// W_PAWN=1, W_KNIGHT=2, W_BISHOP=3, W_ROOK=4, W_QUEEN=5, W_KING=6
// B_PAWN=9, B_KNIGHT=10, B_BISHOP=11, B_ROOK=12, B_QUEEN=13, B_KING=14
static int get_sf_piece(Piece pc)
{
    if (pc == NO_PIECE)
        return 0;
    int pt = pieceType(pc);
    int color = pieceColor(pc);
    int sf_pt;
    if (pt == PAWN)
        sf_pt = 1;
    else if (pt == KNIGHT)
        sf_pt = 2;
    else if (pt == BISHOP)
        sf_pt = 3;
    else if (pt == ROOK)
        sf_pt = 4;
    else if (pt == QUEEN)
        sf_pt = 5;
    else if (pt == KING)
        sf_pt = 6;
    else
        return 0;
    return (color == WHITE) ? sf_pt : sf_pt + 8;
}

void initializeThreatLuts(void)
{
    static const int numValidTargets[16] = {0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0};

    static const int tmap[6][6] = {{0, 1, -1, 2, -1, -1}, {0, 1, 2, 3, 4, -1}, {0, 1, 2, 3, -1, -1},
                                   {0, 1, 2, 3, -1, -1},  {0, 1, 2, 3, 4, -1}, {-1, -1, -1, -1, -1, -1}};

    static const int allPieces[] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14};

    memset(helper_offsets, 0, sizeof(helper_offsets));
    memset(threat_offsets, 0, sizeof(threat_offsets));
    memset(index_lut1, 0, sizeof(index_lut1));
    memset(index_lut2, 0, sizeof(index_lut2));

    for (int i = 0; i < 12; i++) {
        int attacker = allPieces[i];
        int aColor = (attacker > 8) ? BLACK : WHITE;
        int aType = attacker & 7;
        PieceType pAttackerType;
        switch (aType) {
        case 1:
            pAttackerType = PAWN;
            break;
        case 2:
            pAttackerType = KNIGHT;
            break;
        case 3:
            pAttackerType = BISHOP;
            break;
        case 4:
            pAttackerType = ROOK;
            break;
        case 5:
            pAttackerType = QUEEN;
            break;
        case 6:
            pAttackerType = KING;
            break;
        default:
            continue;
        }

        for (Square from = A1; from <= H8; from++) {
            Bitboard attacks = 0;
            if (pAttackerType == PAWN) {
                attacks = (aColor == WHITE) ? generalMoves[WHITE_PAWN][from] : generalMoves[BLACK_PAWN][from];
            } else {
                attacks = generalMoves[pAttackerType][from];
            }

            for (Square to = A1; to <= H8; to++) {
                index_lut2[attacker][from][to] = (uint8_t)getNumberOfSetSquares(attacks & (minValue[to] - 1ULL));
            }
        }
    }

    int cumulativeOffset = 0;
    for (int i = 0; i < 12; i++) {
        int attacker = allPieces[i];
        int aColor = (attacker > 8) ? BLACK : WHITE;
        int aType = attacker & 7;
        PieceType pAttackerType;
        switch (aType) {
        case 1:
            pAttackerType = PAWN;
            break;
        case 2:
            pAttackerType = KNIGHT;
            break;
        case 3:
            pAttackerType = BISHOP;
            break;
        case 4:
            pAttackerType = ROOK;
            break;
        case 5:
            pAttackerType = QUEEN;
            break;
        case 6:
            pAttackerType = KING;
            break;
        default:
            continue;
        }

        int cumulativePieceOffset = 0;
        for (Square from = A1; from <= H8; from++) {
            threat_offsets[attacker][from] = cumulativePieceOffset;

            Bitboard attacks = 0;
            if (pAttackerType != PAWN) {
                attacks = generalMoves[pAttackerType][from];
            } else if (rank(from) >= RANK_2 && rank(from) <= RANK_7) {
                attacks = (aColor == WHITE) ? generalMoves[WHITE_PAWN][from] : generalMoves[BLACK_PAWN][from];
            }
            cumulativePieceOffset += getNumberOfSetSquares(attacks);
        }

        helper_offsets[attacker].cumulativePieceOffset = cumulativePieceOffset;
        helper_offsets[attacker].cumulativeOffset = cumulativeOffset;

        cumulativeOffset += numValidTargets[attacker] * cumulativePieceOffset;
    }

    for (int i = 0; i < 12; i++) {
        int attacker = allPieces[i];
        for (int j = 0; j < 12; j++) {
            int attacked = allPieces[j];

            int attackerType = attacker & 7;
            int attackedType = attacked & 7;
            int attackerColor = (attacker > 8) ? BLACK : WHITE;
            int attackedColor = (attacked > 8) ? BLACK : WHITE;

            bool enemy = (attackerColor != attackedColor);
            int m = tmap[attackerType - 1][attackedType - 1];
            bool semi_excluded = (attackerType == attackedType) && (enemy || attackerType != 1);

            uint32_t feature =
                (uint32_t)helper_offsets[attacker].cumulativeOffset +
                (attackedColor * (numValidTargets[attacker] / 2) + m) * helper_offsets[attacker].cumulativePieceOffset;

            bool excluded = (m < 0);
            index_lut1[attacker][attacked][0] = excluded ? THREAT_INPUT_DIMENSIONS : feature;
            index_lut1[attacker][attacked][1] = (excluded || semi_excluded) ? THREAT_INPUT_DIMENSIONS : feature;
        }
    }
}

// Big Network layers (8 stacks)
static int32_t big_fc0_biases[LAYER_STACKS][L2_BIG + 1] __attribute__((aligned(64)));
static int8_t big_fc0_weights[LAYER_STACKS][(L2_BIG + 1) * L1_BIG] __attribute__((aligned(64)));

static int32_t big_fc1_biases[LAYER_STACKS][L3_BIG] __attribute__((aligned(64)));
static int8_t big_fc1_weights[LAYER_STACKS][L3_BIG * 64] __attribute__((aligned(64)));

static int32_t big_fc2_biases[LAYER_STACKS][1] __attribute__((aligned(64)));
static int8_t big_fc2_weights[LAYER_STACKS][1 * L3_BIG] __attribute__((aligned(64)));

static int nnue_loaded = 0;

static const Square OrientTBL[64] = {H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1,
                                     H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1,
                                     H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1,
                                     H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1};

static const Square OrientTBLThreats[64] = {A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1,
                                            A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1,
                                            A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1,
                                            A1, A1, A1, A1, H1, H1, H1, H1, A1, A1, A1, A1, H1, H1, H1, H1};

#define B(v) (v * 704)
static const int KingBuckets[64] = {
    B(28), B(29), B(30), B(31), B(31), B(30), B(29), B(28), B(24), B(25), B(26), B(27), B(27), B(26), B(25), B(24),
    B(20), B(21), B(22), B(23), B(23), B(22), B(21), B(20), B(16), B(17), B(18), B(19), B(19), B(18), B(17), B(16),
    B(12), B(13), B(14), B(15), B(15), B(14), B(13), B(12), B(8),  B(9),  B(10), B(11), B(11), B(10), B(9),  B(8),
    B(4),  B(5),  B(6),  B(7),  B(7),  B(6),  B(5),  B(4),  B(0),  B(1),  B(2),  B(3),  B(3),  B(2),  B(1),  B(0),
};
#undef B

uint32_t make_threat_index(Color perspective, Piece attacker, Square from, Square to, Piece attacked, Square ksq)
{
    int flip = 56 * perspective;
    int orientation = OrientTBLThreats[ksq] ^ flip;

    Square from_oriented = (Square)(from ^ orientation);
    Square to_oriented = (Square)(to ^ orientation);

    int swap = 8 * perspective;
    int sf_attacker = get_sf_piece(attacker) ^ swap;
    int sf_attacked = get_sf_piece(attacked) ^ swap;

    return index_lut1[sf_attacker][sf_attacked][from_oriented < to_oriented] +
           threat_offsets[sf_attacker][from_oriented] + index_lut2[sf_attacker][from_oriented][to_oriented];
}

#define PS_NONE 0
#define PS_W_PAWN (0 * 64)
#define PS_B_PAWN (1 * 64)
#define PS_W_KNIGHT (2 * 64)
#define PS_B_KNIGHT (3 * 64)
#define PS_W_BISHOP (4 * 64)
#define PS_B_BISHOP (5 * 64)
#define PS_W_ROOK (6 * 64)
#define PS_B_ROOK (7 * 64)
#define PS_W_QUEEN (8 * 64)
#define PS_B_QUEEN (9 * 64)
#define PS_KING (10 * 64)

static const int PieceSquareIndex[2][16] = {
    {PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE, PS_NONE, PS_B_PAWN,
     PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE},
    {PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE, PS_NONE, PS_W_PAWN,
     PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE}};

static int get_feature_index(Square s, Piece pc, Square ksq, Color perspective)
{
    const int flip = 56 * perspective;
    int sf_pc = get_sf_piece(pc);

    int orientation = OrientTBL[ksq] ^ flip;
    return (int)(s ^ orientation) + PieceSquareIndex[perspective][sf_pc] + KingBuckets[ksq ^ flip];
}

static void computeThreatAccumulator(Position *pos, Accumulator *acc, int p)
{
    Square ksq = pos->king[p];
    Bitboard occupied = pos->allPieces;
    static const PieceType attackerTypes[] = {PAWN, KNIGHT, BISHOP, ROOK, QUEEN};
    for (int c_idx = 0; c_idx < 2; c_idx++) {
        Color c = (Color)c_idx;
        for (int pt_idx = 0; pt_idx < 5; pt_idx++) {
            PieceType pt = attackerTypes[pt_idx];
            Piece attacker = (Piece)(c | pt);
            Bitboard bb = pos->piecesOfType[attacker];
            while (bb) {
                Square from = getLastSquare(&bb);
                Bitboard attacks;
                if (pt == PAWN) {
                    attacks = (c == WHITE) ? ((shiftLeft(minValue[from]) | shiftRight(minValue[from])) << 8)
                                           : ((shiftLeft(minValue[from]) | shiftRight(minValue[from])) >> 8);
                } else {
                    attacks = getMoves(from, attacker, occupied);
                }
                attacks &= occupied;
                while (attacks) {
                    Square to = getLastSquare(&attacks);
                    Piece attacked = pos->piece[to];
                    uint32_t idx = make_threat_index(p, attacker, from, to, attacked, ksq);
                    if (idx < THREAT_INPUT_DIMENSIONS) {
                        add_weights_int8_to_int16(acc->big_threat_v[p], big_ft_threat_weights + idx * L1_BIG, L1_BIG);
                        for (int i = 0; i < 8; i++)
                            acc->big_threat_psqtAccumulation[p][i] += big_ft_threat_psqt_weights[idx * 8 + i];
                    }
                }
            }
        }
    }
}

/* ===== Incremental threat accumulator update ===== */

#define MAX_DIRTY_THREATS 256

typedef struct {
    Piece attacker;
    Piece attacked;
    uint8_t from;
    uint8_t to;
    bool add;
} ThreatDirty;

typedef struct {
    ThreatDirty entries[MAX_DIRTY_THREATS];
    int count;
    bool overflow;
} ThreatDirtyList;

static inline void pushThreatDirty(ThreatDirtyList *dl, Piece attacker, Square from, Piece attacked, Square to,
                                   bool add)
{
    if (attacker == NO_PIECE || pieceType(attacker) == KING)
        return;
    if (attacked == NO_PIECE || pieceType(attacked) == KING)
        return;
    if (dl->count < MAX_DIRTY_THREATS) {
        ThreatDirty *e = &dl->entries[dl->count++];
        e->attacker = attacker;
        e->attacked = attacked;
        e->from = (uint8_t)from;
        e->to = (uint8_t)to;
        e->add = add;
    } else {
        dl->overflow = TRUE;
    }
}

/* Piece lookup with up to 2 square overrides */
static inline Piece pcAt(Position *pos, Square sq, Square ov1, Piece pc1, Square ov2, Piece pc2)
{
    if (ov1 != NO_SQUARE && sq == ov1)
        return pc1;
    if (ov2 != NO_SQUARE && sq == ov2)
        return pc2;
    return pos->piece[sq];
}

/* Enumerate outgoing threats from pc at sq, using occ; victim lookup via pcAt */
static void enumOutgoing(ThreatDirtyList *dl, Piece pc, Square sq, Bitboard occ, Square ov1, Piece pov1, Square ov2,
                         Piece pov2, bool add, Position *pos)
{
    if (pc == NO_PIECE || pieceType(pc) == KING)
        return;
    Bitboard attacks;
    if (pieceType(pc) == PAWN) {
        Color c = pieceColor(pc);
        attacks = (c == WHITE) ? generalMoves[WHITE_PAWN][sq] : generalMoves[BLACK_PAWN][sq];
    } else {
        attacks = getMoves(sq, pc, occ);
    }
    attacks &= occ;
    while (attacks) {
        Square t = getLastSquare(&attacks);
        pushThreatDirty(dl, pc, sq, pcAt(pos, t, ov1, pov1, ov2, pov2), t, add);
    }
}

/* Enumerate incoming threats to victim_pc at sq from all attackers in occ.
   skip_sq: skip this attacker square to avoid double-counting. */
static void enumIncoming(ThreatDirtyList *dl, Piece victim_pc, Square sq, Bitboard occ, Square ov1, Piece pov1,
                         Square ov2, Piece pov2, Square skip_sq, bool add, Position *pos)
{
    if (victim_pc == NO_PIECE || pieceType(victim_pc) == KING)
        return;
    Bitboard bb;
    Square asq;
    Piece apc;

    /* Knight attackers */
    bb = getKnightMoves(sq) & occ;
    while (bb) {
        asq = getLastSquare(&bb);
        if (asq == skip_sq)
            continue;
        apc = pcAt(pos, asq, ov1, pov1, ov2, pov2);
        if (pieceType(apc) == KNIGHT)
            pushThreatDirty(dl, apc, asq, victim_pc, sq, add);
    }

    /* Pawn attackers: generalMoves[BLACK_PAWN][sq] gives white pawn squares attacking sq */
    bb = generalMoves[BLACK_PAWN][sq] & occ;
    while (bb) {
        asq = getLastSquare(&bb);
        if (asq == skip_sq)
            continue;
        apc = pcAt(pos, asq, ov1, pov1, ov2, pov2);
        if (apc == WHITE_PAWN)
            pushThreatDirty(dl, apc, asq, victim_pc, sq, add);
    }
    bb = generalMoves[WHITE_PAWN][sq] & occ;
    while (bb) {
        asq = getLastSquare(&bb);
        if (asq == skip_sq)
            continue;
        apc = pcAt(pos, asq, ov1, pov1, ov2, pov2);
        if (apc == BLACK_PAWN)
            pushThreatDirty(dl, apc, asq, victim_pc, sq, add);
    }

    /* Ortho slider attackers (rook / queen) */
    bb = getMoves(sq, WHITE_ROOK, occ) & occ;
    while (bb) {
        asq = getLastSquare(&bb);
        if (asq == skip_sq)
            continue;
        apc = pcAt(pos, asq, ov1, pov1, ov2, pov2);
        if (pieceType(apc) == ROOK || pieceType(apc) == QUEEN)
            pushThreatDirty(dl, apc, asq, victim_pc, sq, add);
    }

    /* Diag slider attackers (bishop / queen) */
    bb = getMoves(sq, WHITE_BISHOP, occ) & occ;
    while (bb) {
        asq = getLastSquare(&bb);
        if (asq == skip_sq)
            continue;
        apc = pcAt(pos, asq, ov1, pov1, ov2, pov2);
        if (pieceType(apc) == BISHOP || pieceType(apc) == QUEEN)
            pushThreatDirty(dl, apc, asq, victim_pc, sq, add);
    }
}

/* Slider discovery: vac_sq was just vacated.  For each slider that was attacking
   vac_sq in old_occ, ADD the threat to the first piece now visible beyond vac_sq.
   skip_target: skip revealed squares equal to this (handled elsewhere). */
static void enumDiscoveries(ThreatDirtyList *dl, Square vac_sq, Bitboard old_occ, Bitboard new_occ, Square ov1,
                            Piece pov1, Square skip_target, Position *pos)
{
    Bitboard bb;
    Square asq;
    Piece apc;

    /* Ortho sliders */
    bb = getMoves(vac_sq, WHITE_ROOK, old_occ) & old_occ;
    while (bb) {
        asq = getLastSquare(&bb);
        apc = pcAt(pos, asq, ov1, pov1, NO_SQUARE, NO_PIECE);
        if (pieceType(apc) != ROOK && pieceType(apc) != QUEEN)
            continue;
        if (!(new_occ & minValue[asq]))
            continue; /* slider captured */
        if (pos->piece[asq] != apc)
            continue; /* slider replaced */
        Bitboard revealed = getMoves(asq, WHITE_ROOK, new_occ) & squaresBehind[vac_sq][asq] & new_occ;
        while (revealed) {
            Square vsq = getLastSquare(&revealed);
            if (vsq == skip_target)
                continue;
            pushThreatDirty(dl, apc, asq, pos->piece[vsq], vsq, TRUE);
        }
    }

    /* Diag sliders */
    bb = getMoves(vac_sq, WHITE_BISHOP, old_occ) & old_occ;
    while (bb) {
        asq = getLastSquare(&bb);
        apc = pcAt(pos, asq, ov1, pov1, NO_SQUARE, NO_PIECE);
        if (pieceType(apc) != BISHOP && pieceType(apc) != QUEEN)
            continue;
        if (!(new_occ & minValue[asq]))
            continue;
        if (pos->piece[asq] != apc)
            continue;
        Bitboard revealed = getMoves(asq, WHITE_BISHOP, new_occ) & squaresBehind[vac_sq][asq] & new_occ;
        while (revealed) {
            Square vsq = getLastSquare(&revealed);
            if (vsq == skip_target)
                continue;
            pushThreatDirty(dl, apc, asq, pos->piece[vsq], vsq, TRUE);
        }
    }
}

/* Slider blocking: new_sq was just occupied (quiet move only; new_sq was empty in old_occ).
   For each slider that now attacks new_sq, REMOVE the threat it had to the piece it was
   previously attacking through new_sq.
   skip_target: skip pieces-beyond equal to this (handled elsewhere). */
static void enumBlockings(ThreatDirtyList *dl, Square new_sq, Bitboard old_occ, Bitboard new_occ, Square skip_target,
                          Position *pos)
{
    Bitboard bb;
    Square asq;
    Piece apc;

    /* Ortho sliders */
    bb = getMoves(new_sq, WHITE_ROOK, new_occ) & new_occ;
    while (bb) {
        asq = getLastSquare(&bb);
        apc = pos->piece[asq];
        if (pieceType(apc) != ROOK && pieceType(apc) != QUEEN)
            continue;
        Bitboard blocked = getMoves(asq, WHITE_ROOK, old_occ) & squaresBehind[new_sq][asq] & old_occ;
        while (blocked) {
            Square vsq = getLastSquare(&blocked);
            if (vsq == skip_target)
                continue;
            pushThreatDirty(dl, apc, asq, pos->piece[vsq], vsq, FALSE);
        }
    }

    /* Diag sliders */
    bb = getMoves(new_sq, WHITE_BISHOP, new_occ) & new_occ;
    while (bb) {
        asq = getLastSquare(&bb);
        apc = pos->piece[asq];
        if (pieceType(apc) != BISHOP && pieceType(apc) != QUEEN)
            continue;
        Bitboard blocked = getMoves(asq, WHITE_BISHOP, old_occ) & squaresBehind[new_sq][asq] & old_occ;
        while (blocked) {
            Square vsq = getLastSquare(&blocked);
            if (vsq == skip_target)
                continue;
            pushThreatDirty(dl, apc, asq, pos->piece[vsq], vsq, FALSE);
        }
    }
}

static void buildThreatDirtyList(ThreatDirtyList *dl, Position *pos, int added_cnt, Square *added_sq, Piece *added_pc,
                                 int removed_cnt, Square *removed_sq, Piece *removed_pc)
{
    dl->count = 0;
    dl->overflow = FALSE;

    Square from_sq = removed_sq[0];
    Piece from_pc = removed_pc[0];
    Square to_sq = added_sq[0];
    Piece new_pc = added_pc[0];
    bool is_cap = (removed_cnt > 1);
    Piece cap_pc = is_cap ? removed_pc[1] : NO_PIECE;

    /* Reconstruct old occupancy */
    Bitboard new_occ = pos->allPieces;
    Bitboard old_occ = new_occ | minValue[from_sq];
    if (!is_cap)
        old_occ &= ~minValue[to_sq]; /* to_sq was empty before quiet move */

    /* --- OLD threats to REMOVE --- */

    /* (A) Outgoing from from_pc @ from_sq */
    enumOutgoing(dl, from_pc, from_sq, old_occ, to_sq, cap_pc, NO_SQUARE, NO_PIECE, FALSE, pos);

    /* (B) Incoming to from_pc @ from_sq; skip to_sq (covered by C for captures,
       absent in old_occ for quiet moves) */
    enumIncoming(dl, from_pc, from_sq, old_occ, to_sq, cap_pc, NO_SQUARE, NO_PIECE, to_sq, FALSE, pos);

    /* (C) Outgoing from cap_pc @ to_sq (capture only) */
    if (is_cap) {
        enumOutgoing(dl, cap_pc, to_sq, old_occ, from_sq, from_pc, NO_SQUARE, NO_PIECE, FALSE, pos);
    }

    /* (D) Incoming to cap_pc @ to_sq; skip from_sq (already in A) */
    if (is_cap) {
        enumIncoming(dl, cap_pc, to_sq, old_occ, from_sq, from_pc, NO_SQUARE, NO_PIECE, from_sq, FALSE, pos);
    }

    /* (E) Slider blocking (quiet move only): sliders that passed through to_sq */
    if (!is_cap) {
        enumBlockings(dl, to_sq, old_occ, new_occ, from_sq, /* skip if piece-beyond was at from_sq (covered by B) */
                      pos);
    }

    /* --- NEW threats to ADD --- */

    /* (F) Outgoing from new_pc @ to_sq */
    enumOutgoing(dl, new_pc, to_sq, new_occ, NO_SQUARE, NO_PIECE, NO_SQUARE, NO_PIECE, TRUE, pos);

    /* (G) Incoming to new_pc @ to_sq; from_sq is empty in new_occ, no skip needed */
    enumIncoming(dl, new_pc, to_sq, new_occ, NO_SQUARE, NO_PIECE, NO_SQUARE, NO_PIECE, NO_SQUARE, TRUE, pos);

    /* (H) Slider discovery: from_sq vacated, sliders now see beyond it.
       Skip revealed pieces at to_sq (covered by G). */
    enumDiscoveries(dl, from_sq, old_occ, new_occ, to_sq, cap_pc, to_sq, pos);
}

static void applyThreatDirtyList(const ThreatDirtyList *dl, Accumulator *next, Position *pos)
{
    for (int i = 0; i < dl->count; i++) {
        const ThreatDirty *e = &dl->entries[i];
        for (int p = 0; p < 2; p++) {
            uint32_t idx =
                make_threat_index((Color)p, e->attacker, (Square)e->from, (Square)e->to, e->attacked, pos->king[p]);
            if (idx >= THREAT_INPUT_DIMENSIONS)
                continue;
            if (e->add) {
                add_weights_int8_to_int16(next->big_threat_v[p], big_ft_threat_weights + idx * L1_BIG, L1_BIG);
                for (int j = 0; j < 8; j++)
                    next->big_threat_psqtAccumulation[p][j] += big_ft_threat_psqt_weights[idx * 8 + j];
            } else {
                sub_weights_int8_to_int16(next->big_threat_v[p], big_ft_threat_weights + idx * L1_BIG, L1_BIG);
                for (int j = 0; j < 8; j++)
                    next->big_threat_psqtAccumulation[p][j] -= big_ft_threat_psqt_weights[idx * 8 + j];
            }
        }
    }
}

/* forward declaration — defined after resetFinnyTable */
void refreshAccumulatorOneSide(Position *pos, Accumulator *acc, FinnyTable *finny, int p);

static void applyThreatDirtyListOneSide(const ThreatDirtyList *dl, Accumulator *next, Position *pos, int p)
{
    for (int i = 0; i < dl->count; i++) {
        const ThreatDirty *e = &dl->entries[i];
        uint32_t idx =
            make_threat_index((Color)p, e->attacker, (Square)e->from, (Square)e->to, e->attacked, pos->king[p]);
        if (idx >= THREAT_INPUT_DIMENSIONS)
            continue;
        if (e->add) {
            add_weights_int8_to_int16(next->big_threat_v[p], big_ft_threat_weights + idx * L1_BIG, L1_BIG);
            for (int j = 0; j < 8; j++)
                next->big_threat_psqtAccumulation[p][j] += big_ft_threat_psqt_weights[idx * 8 + j];
        } else {
            sub_weights_int8_to_int16(next->big_threat_v[p], big_ft_threat_weights + idx * L1_BIG, L1_BIG);
            for (int j = 0; j < 8; j++)
                next->big_threat_psqtAccumulation[p][j] -= big_ft_threat_psqt_weights[idx * 8 + j];
        }
    }
}

void updateAccumulatorOneSide(Accumulator *next, int added_count, Square *added_sq, Piece *added_pc,
                               int removed_count, Square *removed_sq, Piece *removed_pc, Square *ksq, Position *pos,
                               FinnyTable *finny, int p)
{
    for (int j = 0; j < removed_count; j++) {
        int idx = get_feature_index(removed_sq[j], removed_pc[j], ksq[p], p);
        if (idx < 0)
            continue;
        sub_weights_int16(next->small_v[p], small_ft_weights + idx * L1_SMALL, L1_SMALL);
        for (int i = 0; i < 8; i++)
            next->small_psqtAccumulation[p][i] -= small_ft_psqt_weights[idx * 8 + i];
        sub_weights_int16(next->big_v[p], big_ft_weights + idx * L1_BIG, L1_BIG);
        for (int i = 0; i < 8; i++)
            next->big_psqtAccumulation[p][i] -= big_ft_psqt_weights[idx * 8 + i];
    }
    for (int j = 0; j < added_count; j++) {
        int idx = get_feature_index(added_sq[j], added_pc[j], ksq[p], p);
        if (idx < 0)
            continue;
        add_weights_int16(next->small_v[p], small_ft_weights + idx * L1_SMALL, L1_SMALL);
        for (int i = 0; i < 8; i++)
            next->small_psqtAccumulation[p][i] += small_ft_psqt_weights[idx * 8 + i];
        add_weights_int16(next->big_v[p], big_ft_weights + idx * L1_BIG, L1_BIG);
        for (int i = 0; i < 8; i++)
            next->big_psqtAccumulation[p][i] += big_ft_psqt_weights[idx * 8 + i];
    }

    ThreatDirtyList dl;
    buildThreatDirtyList(&dl, pos, added_count, added_sq, added_pc, removed_count, removed_sq, removed_pc);
    if (dl.overflow) {
        refreshAccumulatorOneSide(pos, next, finny, p);
    } else {
        applyThreatDirtyListOneSide(&dl, next, pos, p);
    }
    next->computed[p] = TRUE;
}

void resetFinnyTable(FinnyTable *finny)
{
    for (int p = 0; p < 2; p++)
        for (int sq = 0; sq < 64; sq++)
            finny->entry[p][sq].valid = FALSE;
}

void refreshAccumulatorOneSide(Position *pos, Accumulator *acc, FinnyTable *finny, int p)
{
    Square ksq = pos->king[p];
    FinnyEntry *entry = &finny->entry[p][(int)ksq];

    if (entry->valid) {
        /* Incremental update from cached state */
        memcpy(acc->small_v[p], entry->small_v, sizeof(int16_t) * L1_SMALL);
        memcpy(acc->big_v[p], entry->big_v, sizeof(int16_t) * L1_BIG);
        memcpy(acc->big_threat_v[p], entry->big_threat_v, sizeof(int16_t) * L1_BIG);
        memcpy(acc->small_psqtAccumulation[p], entry->small_psqt, sizeof(int32_t) * 8);
        memcpy(acc->big_psqtAccumulation[p], entry->big_psqt, sizeof(int32_t) * 8);
        memcpy(acc->big_threat_psqtAccumulation[p], entry->big_threat_psqt, sizeof(int32_t) * 8);

        int added_count = 0, removed_count = 0;
        Square added_sq[8], removed_sq[8];
        Piece added_pc[8], removed_pc[8];

        for (Square s = A1; s <= H8; s++) {
            Piece cached_pc = entry->piece[s];
            Piece curr_pc = pos->piece[s];
            if (cached_pc == curr_pc)
                continue;

            if (cached_pc != NO_PIECE) {
                int idx = get_feature_index(s, cached_pc, ksq, p);
                if (idx >= 0) {
                    sub_weights_int16(acc->small_v[p], small_ft_weights + idx * L1_SMALL, L1_SMALL);
                    for (int i = 0; i < 8; i++)
                        acc->small_psqtAccumulation[p][i] -= small_ft_psqt_weights[idx * 8 + i];
                    sub_weights_int16(acc->big_v[p], big_ft_weights + idx * L1_BIG, L1_BIG);
                    for (int i = 0; i < 8; i++)
                        acc->big_psqtAccumulation[p][i] -= big_ft_psqt_weights[idx * 8 + i];
                }
                if (removed_count < 8) {
                    removed_sq[removed_count] = s;
                    removed_pc[removed_count++] = cached_pc;
                }
            }
            if (curr_pc != NO_PIECE) {
                int idx = get_feature_index(s, curr_pc, ksq, p);
                if (idx >= 0) {
                    add_weights_int16(acc->small_v[p], small_ft_weights + idx * L1_SMALL, L1_SMALL);
                    for (int i = 0; i < 8; i++)
                        acc->small_psqtAccumulation[p][i] += small_ft_psqt_weights[idx * 8 + i];
                    add_weights_int16(acc->big_v[p], big_ft_weights + idx * L1_BIG, L1_BIG);
                    for (int i = 0; i < 8; i++)
                        acc->big_psqtAccumulation[p][i] += big_ft_psqt_weights[idx * 8 + i];
                }
                if (added_count < 8) {
                    added_sq[added_count] = s;
                    added_pc[added_count++] = curr_pc;
                }
            }
        }

        /* Update threat accumulator incrementally from cached state if delta is a simple move */
        bool use_incremental_threat = FALSE;
        if (added_count == 1 && removed_count == 1) {
            use_incremental_threat = TRUE;
        } else if (added_count == 1 && removed_count == 2) {
            /* Normal capture: one of the removed pieces was at the added square */
            if (removed_sq[0] == added_sq[0] || removed_sq[1] == added_sq[0]) {
                use_incremental_threat = TRUE;
                /* Ensure removed_sq[0] is the 'from' square and removed_sq[1] is the 'to' square for buildThreatDirtyList */
                if (removed_sq[0] == added_sq[0]) {
                    Square tmp_sq = removed_sq[0]; Piece tmp_pc = removed_pc[0];
                    removed_sq[0] = removed_sq[1]; removed_pc[0] = removed_pc[1];
                    removed_sq[1] = tmp_sq; removed_pc[1] = tmp_pc;
                }
            }
        }

        if (use_incremental_threat) {
            ThreatDirtyList dl;
            buildThreatDirtyList(&dl, pos, added_count, added_sq, added_pc, removed_count, removed_sq, removed_pc);
            if (dl.overflow) {
                memset(acc->big_threat_v[p], 0, sizeof(int16_t) * L1_BIG);
                memset(acc->big_threat_psqtAccumulation[p], 0, sizeof(int32_t) * 8);
                computeThreatAccumulator(pos, acc, p);
            } else {
                applyThreatDirtyListOneSide(&dl, acc, pos, p);
            }
        } else if (added_count > 0 || removed_count > 0) {
            /* Complex delta (castling, en-passant, etc.): recompute from scratch */
            memset(acc->big_threat_v[p], 0, sizeof(int16_t) * L1_BIG);
            memset(acc->big_threat_psqtAccumulation[p], 0, sizeof(int32_t) * 8);
            computeThreatAccumulator(pos, acc, p);
        }
    } else {
        /* Full refresh from biases */
        memcpy(acc->small_v[p], small_ft_biases, sizeof(int16_t) * L1_SMALL);
        memcpy(acc->big_v[p], big_ft_biases, sizeof(int16_t) * L1_BIG);
        memset(acc->big_threat_v[p], 0, sizeof(int16_t) * L1_BIG);
        memset(acc->small_psqtAccumulation[p], 0, sizeof(int32_t) * 8);
        memset(acc->big_psqtAccumulation[p], 0, sizeof(int32_t) * 8);
        memset(acc->big_threat_psqtAccumulation[p], 0, sizeof(int32_t) * 8);

        for (Square s = A1; s <= H8; s++) {
            Piece pc = pos->piece[s];
            if (pc != NO_PIECE) {
                int idx = get_feature_index(s, pc, ksq, p);
                if (idx >= 0) {
                    add_weights_int16(acc->small_v[p], small_ft_weights + idx * L1_SMALL, L1_SMALL);
                    for (int i = 0; i < 8; i++)
                        acc->small_psqtAccumulation[p][i] += small_ft_psqt_weights[idx * 8 + i];
                    add_weights_int16(acc->big_v[p], big_ft_weights + idx * L1_BIG, L1_BIG);
                    for (int i = 0; i < 8; i++)
                        acc->big_psqtAccumulation[p][i] += big_ft_psqt_weights[idx * 8 + i];
                }
            }
        }
        computeThreatAccumulator(pos, acc, p);
    }

    /* Update Finny cache with the freshly computed state */
    memcpy(entry->piece, pos->piece, sizeof(Piece) * 64);
    memcpy(entry->small_v, acc->small_v[p], sizeof(int16_t) * L1_SMALL);
    memcpy(entry->big_v, acc->big_v[p], sizeof(int16_t) * L1_BIG);
    memcpy(entry->big_threat_v, acc->big_threat_v[p], sizeof(int16_t) * L1_BIG);
    memcpy(entry->small_psqt, acc->small_psqtAccumulation[p], sizeof(int32_t) * 8);
    memcpy(entry->big_psqt, acc->big_psqtAccumulation[p], sizeof(int32_t) * 8);
    memcpy(entry->big_threat_psqt, acc->big_threat_psqtAccumulation[p], sizeof(int32_t) * 8);
    entry->valid = TRUE;
    acc->computed[p] = TRUE;
}

void refreshAccumulator(Position *pos, Accumulator *acc, FinnyTable *finny)
{
    for (int p = 0; p < 2; p++)
        refreshAccumulatorOneSide(pos, acc, finny, p);
}

bool kingStaysInSameBucket(Square from, Square to, Color color)
{
    int flip = 56 * color;
    return KingBuckets[from ^ flip] == KingBuckets[to ^ flip] && OrientTBL[from] == OrientTBL[to];
}

void updateAccumulator(const Accumulator *prev, Accumulator *next, int added_count, Square *added_sq, Piece *added_pc,
                       int removed_count, Square *removed_sq, Piece *removed_pc, Square *ksq, Position *pos,
                       FinnyTable *finny)
{
    /* Single contiguous copy instead of 6 separate memcpy calls per side */
    memcpy(next, prev, sizeof(Accumulator));

    for (int p = 0; p < 2; p++) {
        for (int j = 0; j < removed_count; j++) {
            int idx = get_feature_index(removed_sq[j], removed_pc[j], ksq[p], p);
            if (idx < 0)
                continue;
            sub_weights_int16(next->small_v[p], small_ft_weights + idx * L1_SMALL, L1_SMALL);
            for (int i = 0; i < 8; i++)
                next->small_psqtAccumulation[p][i] -= small_ft_psqt_weights[idx * 8 + i];
            sub_weights_int16(next->big_v[p], big_ft_weights + idx * L1_BIG, L1_BIG);
            for (int i = 0; i < 8; i++)
                next->big_psqtAccumulation[p][i] -= big_ft_psqt_weights[idx * 8 + i];
        }
        for (int j = 0; j < added_count; j++) {
            int idx = get_feature_index(added_sq[j], added_pc[j], ksq[p], p);
            if (idx < 0)
                continue;
            add_weights_int16(next->small_v[p], small_ft_weights + idx * L1_SMALL, L1_SMALL);
            for (int i = 0; i < 8; i++)
                next->small_psqtAccumulation[p][i] += small_ft_psqt_weights[idx * 8 + i];
            add_weights_int16(next->big_v[p], big_ft_weights + idx * L1_BIG, L1_BIG);
            for (int i = 0; i < 8; i++)
                next->big_psqtAccumulation[p][i] += big_ft_psqt_weights[idx * 8 + i];
        }
    }

    ThreatDirtyList dl;
    buildThreatDirtyList(&dl, pos, added_count, added_sq, added_pc, removed_count, removed_sq, removed_pc);
    if (dl.overflow) {
        refreshAccumulator(pos, next, finny);
    } else {
        applyThreatDirtyList(&dl, next, pos);
    }
    next->computed[0] = TRUE;
    next->computed[1] = TRUE;
}

static const uint8_t *read_data;
static const uint8_t *read_data_end;
static size_t read_pos;

static void mem_read(void *dest, size_t size)
{
    if (read_pos + size > (size_t)(read_data_end - read_data)) {
        memset(dest, 0, size);
        read_pos = (size_t)(read_data_end - read_data);
        return;
    }
    memcpy(dest, &read_data[read_pos], size);
    read_pos += size;
}

static void mem_skip(size_t size)
{
    read_pos += size;
    if (read_pos > (size_t)(read_data_end - read_data)) {
        read_pos = (size_t)(read_data_end - read_data);
    }
}

static void read_leb128_mem(void *out, size_t count, size_t element_size)
{
    if (read_pos + 17 > (size_t)(read_data_end - read_data))
        return;
    mem_skip(17); // magic
    uint32_t bytes_left;
    mem_read(&bytes_left, 4);
    size_t end_pos = read_pos + bytes_left;
    for (size_t i = 0; i < count; i++) {
        int32_t result = 0;
        int shift = 0;
        while (read_pos < (size_t)(read_data_end - read_data)) {
            uint8_t byte = read_data[read_pos++];
            result |= (int32_t)(byte & 0x7f) << shift;
            shift += 7;
            if (!(byte & 0x80)) {
                if (shift < 32 && (byte & 0x40))
                    result |= ~((unsigned int)((1 << shift) - 1));
                if (element_size == 2)
                    ((int16_t *)out)[i] = (int16_t)result;
                else if (element_size == 4)
                    ((int32_t *)out)[i] = (int32_t)result;
                else if (element_size == 1)
                    ((int8_t *)out)[i] = (int8_t)result;
                break;
            }
        }
    }
    read_pos = end_pos;
}

/* Unused functions removed */

int initializeModuleNnue(void)
{
    // Load small net
    read_data = small_nnue_model_data;
    read_data_end = small_nnue_model_end;
    read_pos = 0;

    if (read_data[0] == 0 && (size_t)(read_data_end - read_data) <= 1)
        return -1;

    uint32_t version, hash, desc_size;
    mem_read(&version, 4);
    if (version != NNUE_VERSION)
        return -2;
    mem_read(&hash, 4);
    mem_read(&desc_size, 4);
    mem_skip(desc_size);

    uint32_t ft_hash;
    mem_read(&ft_hash, 4);
    read_leb128_mem(small_ft_biases, L1_SMALL, 2);
    read_leb128_mem(small_ft_weights, L1_SMALL * FT_INPUT_DIMENSIONS, 2);
    read_leb128_mem(small_ft_psqt_weights, FT_INPUT_DIMENSIONS * 8, 4);

    for (int s = 0; s < LAYER_STACKS; s++) {
        uint32_t arch_hash;
        mem_read(&arch_hash, 4);

        // FC0
        mem_read(small_fc0_biases[s], 4 * (L2_SMALL + 1));
        mem_read(small_fc0_weights[s], (L2_SMALL + 1) * L1_SMALL);

        // FC1
        mem_read(small_fc1_biases[s], 4 * L3_SMALL);
        mem_read(small_fc1_weights[s], L3_SMALL * 32);

        // FC2
        mem_read(small_fc2_biases[s], 4 * 1);
        mem_read(small_fc2_weights[s], 1 * L3_SMALL);
    }

    // Load big net
    read_data = big_nnue_model_data;
    read_data_end = big_nnue_model_end;
    read_pos = 0;

    if (read_data[0] == 0 && (size_t)(read_data_end - read_data) <= 1)
        return -1;

    mem_read(&version, 4);
    if (version != NNUE_VERSION)
        return -2;
    mem_read(&hash, 4);
    mem_read(&desc_size, 4);
    mem_skip(desc_size);

    mem_read(&ft_hash, 4);
    read_leb128_mem(big_ft_biases, L1_BIG, 2);

    // threatWeights are raw little-endian int8_t
    mem_read(big_ft_threat_weights, L1_BIG * THREAT_INPUT_DIMENSIONS);
    read_leb128_mem(big_ft_weights, L1_BIG * FT_INPUT_DIMENSIONS, 2);

    // threatPsqtWeights and psqtWeights are combined in one LEB128 block
    static int32_t combined_psqt[THREAT_INPUT_DIMENSIONS * 8 + FT_INPUT_DIMENSIONS * 8];
    read_leb128_mem(combined_psqt, THREAT_INPUT_DIMENSIONS * 8 + FT_INPUT_DIMENSIONS * 8, 4);
    memcpy(big_ft_threat_psqt_weights, combined_psqt, THREAT_INPUT_DIMENSIONS * 8 * 4);
    memcpy(big_ft_psqt_weights, combined_psqt + THREAT_INPUT_DIMENSIONS * 8, FT_INPUT_DIMENSIONS * 8 * 4);

    for (int s = 0; s < LAYER_STACKS; s++) {
        uint32_t arch_hash;
        mem_read(&arch_hash, 4);

        // FC0
        mem_read(big_fc0_biases[s], 4 * (L2_BIG + 1));
        mem_read(big_fc0_weights[s], (L2_BIG + 1) * L1_BIG);

        // FC1
        mem_read(big_fc1_biases[s], 4 * L3_BIG);
        mem_read(big_fc1_weights[s], L3_BIG * 64);

        // FC2
        mem_read(big_fc2_biases[s], 4 * 1);
        mem_read(big_fc2_weights[s], 1 * L3_BIG);
    }

    nnue_loaded = 1;
    initializeThreatLuts();
    return 0;
}

static inline int32_t clipped_relu(int32_t x)
{
    return (x < 0) ? 0 : (x > 127) ? 127 : x;
}

static inline int32_t sqr_clipped_relu(int32_t x)
{
    long long v = (long long)x * x;
    v >>= 19;
    return (v > 127) ? 127 : (int32_t)v;
}

int win_rate_scaling(Position *pos)
{
    int material = getNumberOfSetSquares(pos->piecesOfType[WHITE_PAWN]) +
                   getNumberOfSetSquares(pos->piecesOfType[BLACK_PAWN]) +
                   3 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_KNIGHT]) +
                        getNumberOfSetSquares(pos->piecesOfType[BLACK_KNIGHT])) +
                   3 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_BISHOP]) +
                        getNumberOfSetSquares(pos->piecesOfType[BLACK_BISHOP])) +
                   5 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_ROOK]) +
                        getNumberOfSetSquares(pos->piecesOfType[BLACK_ROOK])) +
                   9 * (getNumberOfSetSquares(pos->piecesOfType[WHITE_QUEEN]) +
                        getNumberOfSetSquares(pos->piecesOfType[BLACK_QUEEN]));

    double m = max(17.0, min(78.0, (double)material)) / 58.0;
    const double as[] = {-72.32565836, 185.93832038, -144.58862193, 416.44950446};
    double a = (((as[0] * m + as[1]) * m + as[2]) * m) + as[3];
    return (int)a;
}

static void applyDirtyPieceOneSide(Accumulator *next, const Accumulator *prev, const DirtyPiece *dp, int p,
                                   Position *pos, FinnyTable *finny)
{
    /* Copy parent's vectors for perspective p */
    memcpy(next->small_v[p], prev->small_v[p], sizeof(int16_t) * L1_SMALL);
    memcpy(next->big_v[p], prev->big_v[p], sizeof(int16_t) * L1_BIG);
    memcpy(next->big_threat_v[p], prev->big_threat_v[p], sizeof(int16_t) * L1_BIG);
    memcpy(next->small_psqtAccumulation[p], prev->small_psqtAccumulation[p], sizeof(int32_t) * 8);
    memcpy(next->big_psqtAccumulation[p], prev->big_psqtAccumulation[p], sizeof(int32_t) * 8);
    memcpy(next->big_threat_psqtAccumulation[p], prev->big_threat_psqtAccumulation[p], sizeof(int32_t) * 8);

    Square added_sq[2], removed_sq[3], ksq[2];
    Piece added_pc[2], removed_pc[3];
    int added_cnt = 0, removed_cnt = 0;

    ksq[0] = pos->king[0];
    ksq[1] = pos->king[1];

    removed_sq[removed_cnt] = dp->from;
    removed_pc[removed_cnt++] = dp->pc;

    if (dp->captured != NO_PIECE) {
        removed_sq[removed_cnt] = dp->to;
        removed_pc[removed_cnt++] = dp->captured;
    }

    if (dp->ep_sq != NO_SQUARE) {
        removed_sq[removed_cnt] = dp->ep_sq;
        removed_pc[removed_cnt++] = (Piece)(PAWN | opponent(pieceColor(dp->pc)));
    }

    added_sq[added_cnt] = dp->to;
    added_pc[added_cnt++] = (dp->promoted_to != NO_PIECE ? dp->promoted_to : dp->pc);

    if (dp->rook_from != NO_SQUARE) {
        Piece rook = (Piece)(ROOK | pieceColor(dp->pc));
        removed_sq[removed_cnt] = dp->rook_from;
        removed_pc[removed_cnt++] = rook;
        added_sq[added_cnt] = dp->rook_to;
        added_pc[added_cnt++] = rook;
    }

    updateAccumulatorOneSide(next, added_cnt, added_sq, added_pc, removed_cnt, removed_sq, removed_pc, ksq, pos,
                             finny, p);
}

void finalizeAccumulator(Variation *var, int p)
{
    int ply = var->ply;
    Accumulator *acc = &var->plyInfo[ply].accumulator;
    if (acc->computed[p])
        return;

    if (ply > 0) {
        const Accumulator *prev = &var->plyInfo[ply - 1].accumulator;
        if (prev->computed[p]) {
            const DirtyPiece *dp = &var->plyInfo[ply - 1].dirtyPiece;

            /* Incremental update is only safe for simple moves that buildThreatDirtyList handles correctly.
               EP, Castling, and Promotion require a full refresh for threats. */
            bool can_do_incremental = TRUE;
            if (pieceColor(dp->pc) == (Color)p && pieceType(dp->pc) == KING) {
                /* Our king moved: bucket might change. */
                can_do_incremental = FALSE;
            } else if (dp->rook_from != NO_SQUARE || dp->ep_sq != NO_SQUARE || dp->promoted_to != NO_PIECE) {
                /* Complex move. */
                can_do_incremental = FALSE;
            }

            if (can_do_incremental) {
                applyDirtyPieceOneSide(acc, prev, dp, p, &var->singlePosition, &var->finnyTable);
                return;
            }
        }
    }

    /* Fallback: full refresh (delta scan against Finny cache) */
    refreshAccumulatorOneSide(&var->singlePosition, acc, &var->finnyTable, p);
}

int evaluateNnueWithAccumulator(Position *pos, Accumulator *acc)
{
    assert(acc != NULL);
    /* Note: if using lazy evaluation, finalizeAccumulator should have been called before this.
       However, evaluateNnueWithAccumulator only takes pos and acc, not var.
       We'll ensure it's called in the search/eval path. */
    int p, v;
    evaluateNnueWithAccumulatorFull(pos, acc, &p, &v);
    int a = win_rate_scaling(pos);
    return (p + v) * 100 / a;
}

void evaluateNnueWithAccumulatorFull(Position *pos, Accumulator *acc, int *psqt_out, int *positional_out)
{
    assert(acc != NULL);
    if (!nnue_loaded) {
        if (psqt_out)
            *psqt_out = 0;
        if (positional_out)
            *positional_out = 0;
        return;
    }

    int num_pieces = pos->numberOfPieces[WHITE] + pos->numberOfPieces[BLACK];
    int bucket = min(7, (num_pieces - 1) / 4);

    Color side = pos->activeColor;

    // PSQT part: (psqtAccumulation[side] - psqtAccumulation[!side]) / 2
    int32_t psqt = (acc->small_psqtAccumulation[side][bucket] - acc->small_psqtAccumulation[!side][bucket]) / 2;

    uint8_t transformed[L1_SMALL] __attribute__((aligned(64)));
    transform_small(acc->small_v[side], acc->small_v[!side], transformed);

    int32_t fc0_out[L2_SMALL + 1];
    fc_u8s8(fc0_out, small_fc0_biases[bucket], transformed, small_fc0_weights[bucket], L1_SMALL, L2_SMALL + 1);

    // Pack activations to uint8 (values bounded to [0,127]); zero-fill padding to 32.
    uint8_t ac0_packed[32] __attribute__((aligned(64))) = {0};
    for (int i = 0; i < L2_SMALL; i++) {
        ac0_packed[i] = (uint8_t)sqr_clipped_relu(fc0_out[i]);
        ac0_packed[L2_SMALL + i] = (uint8_t)clipped_relu(fc0_out[i] >> 6);
    }

    int32_t fc1_out[L3_SMALL];
    fc_u8s8(fc1_out, small_fc1_biases[bucket], ac0_packed, small_fc1_weights[bucket], 32, L3_SMALL);

    uint8_t ac1_packed[32] __attribute__((aligned(64)));
    for (int i = 0; i < L3_SMALL; i++)
        ac1_packed[i] = (uint8_t)clipped_relu(fc1_out[i] >> 6);

    int32_t fc2_out = small_fc2_biases[bucket][0] + dot_u8s8(ac1_packed, small_fc2_weights[bucket], L3_SMALL);

    int32_t fwdOut = (int32_t)((int64_t)fc0_out[L2_SMALL] * (600 * 16) / (127 * 64));
    if (positional_out)
        *positional_out = (fc2_out + fwdOut) / 16;
    if (psqt_out)
        *psqt_out = psqt / 16;
}

int evaluateBigNnueWithAccumulator(Position *pos, Accumulator *acc)
{
    assert(acc != NULL);
    int p, v;
    evaluateBigNnueWithAccumulatorFull(pos, acc, &p, &v);
    int a = win_rate_scaling(pos);
    return (p + v) * 100 / a;
}

void evaluateBigNnueWithAccumulatorFull(Position *pos, Accumulator *acc, int *psqt_out, int *positional_out)
{
    assert(acc != NULL);
    if (!nnue_loaded) {
        if (psqt_out)
            *psqt_out = 0;
        if (positional_out)
            *positional_out = 0;
        return;
    }

    int num_pieces = pos->numberOfPieces[WHITE] + pos->numberOfPieces[BLACK];
    int bucket = min(7, (num_pieces - 1) / 4);

    Color side = pos->activeColor;

    // PSQT part
    int32_t psqt = (acc->big_psqtAccumulation[side][bucket] - acc->big_psqtAccumulation[!side][bucket]);
    psqt += (acc->big_threat_psqtAccumulation[side][bucket] - acc->big_threat_psqtAccumulation[!side][bucket]);
    psqt /= 2;

    uint8_t transformed[L1_BIG] __attribute__((aligned(64)));
    transform_big(acc->big_v[side], acc->big_threat_v[side], acc->big_v[!side], acc->big_threat_v[!side], transformed);

    int32_t fc0_out[L2_BIG + 1];
    fc_u8s8(fc0_out, big_fc0_biases[bucket], transformed, big_fc0_weights[bucket], L1_BIG, L2_BIG + 1);

    // Pack activations to uint8 (values bounded to [0,127]); zero-fill padding to 64.
    uint8_t ac0_packed[64] __attribute__((aligned(64))) = {0};
    for (int i = 0; i < L2_BIG; i++) {
        ac0_packed[i] = (uint8_t)sqr_clipped_relu(fc0_out[i]);
        ac0_packed[L2_BIG + i] = (uint8_t)clipped_relu(fc0_out[i] >> 6);
    }

    int32_t fc1_out[L3_BIG];
    fc_u8s8(fc1_out, big_fc1_biases[bucket], ac0_packed, big_fc1_weights[bucket], 64, L3_BIG);

    uint8_t ac1_packed[32] __attribute__((aligned(64)));
    for (int i = 0; i < L3_BIG; i++)
        ac1_packed[i] = (uint8_t)clipped_relu(fc1_out[i] >> 6);

    int32_t fc2_out = big_fc2_biases[bucket][0] + dot_u8s8(ac1_packed, big_fc2_weights[bucket], L3_BIG);

    int32_t fwdOut = (int32_t)((int64_t)fc0_out[L2_BIG] * (600 * 16) / (127 * 64));
    if (positional_out)
        *positional_out = (fc2_out + fwdOut) / 16;
    if (psqt_out)
        *psqt_out = psqt / 16;
}

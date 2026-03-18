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
#define LEB128_MAGIC_STRING_SIZE (sizeof("COMPRESSED_LEB128") - 1)

#define FT_INPUT_DIMENSIONS (64 * 11 * 64 / 2) // 22528
#define THREAT_INPUT_DIMENSIONS 60144

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

INCBIN(small_nnue_model, "nn-47fc8b7fff06.nnue")
INCBIN(big_nnue_model, "nn-9a0cc2a62c52.nnue")

// Small Feature transformer data
static int16_t small_ft_biases[L1_SMALL] __attribute__((aligned(64)));
static int16_t small_ft_weights[L1_SMALL * FT_INPUT_DIMENSIONS] __attribute__((aligned(64))); 
static int32_t small_ft_psqt_weights[FT_INPUT_DIMENSIONS * 8] __attribute__((aligned(64)));

// Small Network layers (8 stacks)
static int32_t small_fc0_biases[LAYER_STACKS][L2_SMALL + 1] __attribute__((aligned(64)));
static int8_t  small_fc0_weights[LAYER_STACKS][(L2_SMALL + 1) * L1_SMALL] __attribute__((aligned(64))); 

static int32_t small_fc1_biases[LAYER_STACKS][L3_SMALL] __attribute__((aligned(64)));
static int8_t  small_fc1_weights[LAYER_STACKS][L3_SMALL * 32] __attribute__((aligned(64))); 

static int32_t small_fc2_biases[LAYER_STACKS][1] __attribute__((aligned(64)));
static int8_t  small_fc2_weights[LAYER_STACKS][1 * L3_SMALL] __attribute__((aligned(64)));

// Big Feature transformer data
static int16_t big_ft_biases[L1_BIG] __attribute__((aligned(64)));
static int16_t big_ft_weights[L1_BIG * FT_INPUT_DIMENSIONS] __attribute__((aligned(64))); 
static int32_t big_ft_psqt_weights[FT_INPUT_DIMENSIONS * 8] __attribute__((aligned(64)));

static int8_t  big_ft_threat_weights[L1_BIG * THREAT_INPUT_DIMENSIONS] __attribute__((aligned(64)));
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
static int get_sf_piece(Piece pc) {
    if (pc == NO_PIECE) return 0;
    int pt = pieceType(pc);
    int color = pieceColor(pc);
    int sf_pt;
    if (pt == PAWN) sf_pt = 1;
    else if (pt == KNIGHT) sf_pt = 2;
    else if (pt == BISHOP) sf_pt = 3;
    else if (pt == ROOK) sf_pt = 4;
    else if (pt == QUEEN) sf_pt = 5;
    else if (pt == KING) sf_pt = 6;
    else return 0;
    return (color == WHITE) ? sf_pt : sf_pt + 8;
}

void initializeThreatLuts(void) {
    static const int numValidTargets[16] = {
        0, 6, 10, 8, 8, 10, 0, 0, 0, 6, 10, 8, 8, 10, 0, 0
    };

    static const int tmap[6][6] = {
        { 0,  1, -1,  2, -1, -1},
        { 0,  1,  2,  3,  4, -1},
        { 0,  1,  2,  3, -1, -1},
        { 0,  1,  2,  3, -1, -1},
        { 0,  1,  2,  3,  4, -1},
        {-1, -1, -1, -1, -1, -1}
    };

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
        switch(aType) {
            case 1: pAttackerType = PAWN; break;
            case 2: pAttackerType = KNIGHT; break;
            case 3: pAttackerType = BISHOP; break;
            case 4: pAttackerType = ROOK; break;
            case 5: pAttackerType = QUEEN; break;
            case 6: pAttackerType = KING; break;
            default: continue;
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
        switch(aType) {
            case 1: pAttackerType = PAWN; break;
            case 2: pAttackerType = KNIGHT; break;
            case 3: pAttackerType = BISHOP; break;
            case 4: pAttackerType = ROOK; break;
            case 5: pAttackerType = QUEEN; break;
            case 6: pAttackerType = KING; break;
            default: continue;
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
            
            uint32_t feature = (uint32_t)helper_offsets[attacker].cumulativeOffset
                             + (attackedColor * (numValidTargets[attacker] / 2) + m)
                               * helper_offsets[attacker].cumulativePieceOffset;

            bool excluded = (m < 0);
            index_lut1[attacker][attacked][0] = excluded ? THREAT_INPUT_DIMENSIONS : feature;
            index_lut1[attacker][attacked][1] = (excluded || semi_excluded) ? THREAT_INPUT_DIMENSIONS : feature;
        }
    }
}

// Big Network layers (8 stacks)
static int32_t big_fc0_biases[LAYER_STACKS][L2_BIG + 1] __attribute__((aligned(64)));
static int8_t  big_fc0_weights[LAYER_STACKS][(L2_BIG + 1) * L1_BIG] __attribute__((aligned(64))); 

static int32_t big_fc1_biases[LAYER_STACKS][L3_BIG] __attribute__((aligned(64)));
static int8_t  big_fc1_weights[LAYER_STACKS][L3_BIG * 64] __attribute__((aligned(64))); 

static int32_t big_fc2_biases[LAYER_STACKS][1] __attribute__((aligned(64)));
static int8_t  big_fc2_weights[LAYER_STACKS][1 * L3_BIG] __attribute__((aligned(64)));

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

static const Square OrientTBLThreats[64] = {
  A1, A1, A1, A1, H1, H1, H1, H1,
  A1, A1, A1, A1, H1, H1, H1, H1,
  A1, A1, A1, A1, H1, H1, H1, H1,
  A1, A1, A1, A1, H1, H1, H1, H1,
  A1, A1, A1, A1, H1, H1, H1, H1,
  A1, A1, A1, A1, H1, H1, H1, H1,
  A1, A1, A1, A1, H1, H1, H1, H1,
  A1, A1, A1, A1, H1, H1, H1, H1
};

#define B(v) (v * 704)
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

uint32_t make_threat_index(Color perspective, Piece attacker, Square from, Square to, Piece attacked, Square ksq) {
    int flip = 56 * perspective;
    int orientation = OrientTBLThreats[ksq] ^ flip;

    Square from_oriented = (Square)(from ^ orientation);
    Square to_oriented = (Square)(to ^ orientation);

    int swap = 8 * perspective;
    int sf_attacker = get_sf_piece(attacker) ^ swap;
    int sf_attacked = get_sf_piece(attacked) ^ swap;

    return index_lut1[sf_attacker][sf_attacked][from_oriented < to_oriented]
         + threat_offsets[sf_attacker][from_oriented]
         + index_lut2[sf_attacker][from_oriented][to_oriented];
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
    {PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE},
    {PS_NONE, PS_B_PAWN, PS_B_KNIGHT, PS_B_BISHOP, PS_B_ROOK, PS_B_QUEEN, PS_KING, PS_NONE,
     PS_NONE, PS_W_PAWN, PS_W_KNIGHT, PS_W_BISHOP, PS_W_ROOK, PS_W_QUEEN, PS_KING, PS_NONE}
};

static int get_feature_index(Square s, Piece pc, Square ksq, Color perspective) {
    const int flip = 56 * perspective;
    int sf_pc = get_sf_piece(pc);
    
    int orientation = OrientTBL[ksq] ^ flip;
    return (int)(s ^ orientation) + PieceSquareIndex[perspective][sf_pc] + KingBuckets[ksq ^ flip];
}

void refreshAccumulator(Position* pos, Accumulator* acc) {
    for (int p = 0; p < 2; p++) {
        for (int i = 0; i < L1_SMALL; i++) acc->small_v[p][i] = small_ft_biases[i];
        for (int i = 0; i < L1_BIG; i++) acc->big_v[p][i] = big_ft_biases[i];
        for (int i = 0; i < L1_BIG; i++) acc->big_threat_v[p][i] = 0;
        memset(acc->small_psqtAccumulation[p], 0, sizeof(int32_t) * 8);
        memset(acc->big_psqtAccumulation[p], 0, sizeof(int32_t) * 8);
        memset(acc->big_threat_psqtAccumulation[p], 0, sizeof(int32_t) * 8);
        
        Square ksq = pos->king[p];
        for (Square s = A1; s <= H8; s++) {
            Piece pc = pos->piece[s];
            if (pc != NO_PIECE) {
                int idx = get_feature_index(s, pc, ksq, p);
                if (idx < 0) continue;
                for (int i = 0; i < L1_SMALL; i++) acc->small_v[p][i] += small_ft_weights[idx * L1_SMALL + i];
                for (int i = 0; i < 8; i++) acc->small_psqtAccumulation[p][i] += small_ft_psqt_weights[idx * 8 + i];
                for (int i = 0; i < L1_BIG; i++) acc->big_v[p][i] += big_ft_weights[idx * L1_BIG + i];
                for (int i = 0; i < 8; i++) acc->big_psqtAccumulation[p][i] += big_ft_psqt_weights[idx * 8 + i];
            }
        }

        // Big net threats
        Bitboard occupied = pos->allPieces;
        static const PieceType attackerTypes[] = { PAWN, KNIGHT, BISHOP, ROOK, QUEEN };
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
                        attacks = (c == WHITE) ? ((shiftLeft(minValue[from]) | shiftRight(minValue[from])) << 8) : ((shiftLeft(minValue[from]) | shiftRight(minValue[from])) >> 8);
                    } else {
                        attacks = getMoves(from, attacker, occupied);
                    }
                    attacks &= occupied;

                    while (attacks) {
                        Square to = getLastSquare(&attacks);
                        Piece attacked = pos->piece[to];
                        uint32_t idx = make_threat_index(p, attacker, from, to, attacked, ksq);
                        if (idx < THREAT_INPUT_DIMENSIONS) {
                            for (int i = 0; i < L1_BIG; i++) acc->big_threat_v[p][i] += big_ft_threat_weights[idx * L1_BIG + i];
                            for (int i = 0; i < 8; i++) acc->big_threat_psqtAccumulation[p][i] += big_ft_threat_psqt_weights[idx * 8 + i];
                        }
                    }
                }
            }
        }
    }
}

void updateAccumulator(Accumulator* prev, Accumulator* next, int added_count, Square* added_sq, Piece* added_pc, int removed_count, Square* removed_sq, Piece* removed_pc, Square* ksq) {
    for (int p = 0; p < 2; p++) {
        memcpy(next->small_v[p], prev->small_v[p], sizeof(int16_t) * L1_SMALL);
        memcpy(next->big_v[p], prev->big_v[p], sizeof(int16_t) * L1_BIG);
        memcpy(next->big_threat_v[p], prev->big_threat_v[p], sizeof(int16_t) * L1_BIG);
        memcpy(next->small_psqtAccumulation[p], prev->small_psqtAccumulation[p], sizeof(int32_t) * 8);
        memcpy(next->big_psqtAccumulation[p], prev->big_psqtAccumulation[p], sizeof(int32_t) * 8);
        memcpy(next->big_threat_psqtAccumulation[p], prev->big_threat_psqtAccumulation[p], sizeof(int32_t) * 8);

        for (int j = 0; j < removed_count; j++) {
            int idx = get_feature_index(removed_sq[j], removed_pc[j], ksq[p], p);
            if (idx < 0) continue;
            for (int i = 0; i < L1_SMALL; i++) next->small_v[p][i] -= small_ft_weights[idx * L1_SMALL + i];
            for (int i = 0; i < 8; i++) next->small_psqtAccumulation[p][i] -= small_ft_psqt_weights[idx * 8 + i];
            for (int i = 0; i < L1_BIG; i++) next->big_v[p][i] -= big_ft_weights[idx * L1_BIG + i];
            for (int i = 0; i < 8; i++) next->big_psqtAccumulation[p][i] -= big_ft_psqt_weights[idx * 8 + i];
        }
        for (int j = 0; j < added_count; j++) {
            int idx = get_feature_index(added_sq[j], added_pc[j], ksq[p], p);
            if (idx < 0) continue;
            for (int i = 0; i < L1_SMALL; i++) next->small_v[p][i] += small_ft_weights[idx * L1_SMALL + i];
            for (int i = 0; i < 8; i++) next->small_psqtAccumulation[p][i] += small_ft_psqt_weights[idx * 8 + i];
            for (int i = 0; i < L1_BIG; i++) next->big_v[p][i] += big_ft_weights[idx * L1_BIG + i];
            for (int i = 0; i < 8; i++) next->big_psqtAccumulation[p][i] += big_ft_psqt_weights[idx * 8 + i];
        }
    }
}

static const uint8_t* read_data;
static const uint8_t* read_data_end;
static size_t read_pos;

static void mem_read(void* dest, size_t size) {
    if (read_pos + size > (size_t)(read_data_end - read_data)) {
        memset(dest, 0, size);
        read_pos = (size_t)(read_data_end - read_data);
        return;
    }
    memcpy(dest, &read_data[read_pos], size);
    read_pos += size;
}

static void mem_skip(size_t size) {
    read_pos += size;
    if (read_pos > (size_t)(read_data_end - read_data)) {
        read_pos = (size_t)(read_data_end - read_data);
    }
}

static void read_leb128_mem(void* out, size_t count, size_t element_size) {
    if (read_pos + 17 > (size_t)(read_data_end - read_data)) return;
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

/* Unused functions removed */

int initializeModuleNnue(void) {
    // Load small net
    read_data = small_nnue_model_data;
    read_data_end = small_nnue_model_end;
    read_pos = 0;
    
    if (read_data[0] == 0 && (size_t)(read_data_end - read_data) <= 1) return -1;

    uint32_t version, hash, desc_size;
    mem_read(&version, 4);
    if (version != NNUE_VERSION) return -2;
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
    
    if (read_data[0] == 0 && (size_t)(read_data_end - read_data) <= 1) return -1;

    mem_read(&version, 4);
    if (version != NNUE_VERSION) return -2;
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

static inline int32_t clipped_relu(int32_t x) {
    return (x < 0) ? 0 : (x > 127) ? 127 : x;
}

static inline int32_t sqr_clipped_relu(int32_t x) {
    long long v = (long long)x * x;
    v >>= 19;
    return (v > 127) ? 127 : (int32_t)v;
}

int win_rate_scaling(Position* pos) {
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

int evaluateNnueWithAccumulator(Position * pos, Accumulator * acc) {
    assert(acc != NULL);
    int p, v;
    evaluateNnueWithAccumulatorFull(pos, acc, &p, &v);
    int a = win_rate_scaling(pos);
    return (p + v) * 100 / a;
}

void evaluateNnueWithAccumulatorFull(Position * pos, Accumulator * acc, int * psqt_out, int * positional_out) {
    assert(acc != NULL);
    if (!nnue_loaded) {
        if (psqt_out) *psqt_out = 0;
        if (positional_out) *positional_out = 0;
        return;
    }

    int num_pieces = pos->numberOfPieces[WHITE] + pos->numberOfPieces[BLACK];
    int bucket = min(7, (num_pieces - 1) / 4);

    Color side = pos->activeColor;
    
    // PSQT part: (psqtAccumulation[side] - psqtAccumulation[!side]) / 2
    int32_t psqt = (acc->small_psqtAccumulation[side][bucket] - acc->small_psqtAccumulation[!side][bucket]) / 2;

    uint8_t transformed[L1_SMALL] __attribute__((aligned(64))); 
    for (int p = 0; p < 2; p++) {
        Color perspective = (p == 0 ? side : !side);
        for (int i = 0; i < L1_SMALL / 2; i++) {
            int16_t v0 = acc->small_v[perspective][i];
            int16_t v1 = acc->small_v[perspective][L1_SMALL / 2 + i];
            int32_t c0 = max(0, min(255, v0));
            int32_t c1 = max(0, min(255, v1));
            transformed[p * (L1_SMALL / 2) + i] = (uint8_t)((c0 * c1) / 512);
        }
    }

    int32_t fc0_out[L2_SMALL + 1];
    for (int i = 0; i <= L2_SMALL; i++) {
        fc0_out[i] = small_fc0_biases[bucket][i];
    }
    for (int j = 0; j < L1_SMALL; j++) {
        int32_t val = transformed[j];
        if (val == 0) continue;
        for (int i = 0; i <= L2_SMALL; i++) {
            fc0_out[i] += val * small_fc0_weights[bucket][i * L1_SMALL + j];
        }
    }

    int32_t ac0_out[32] = {0}; 
    for (int i = 0; i < L2_SMALL; i++) {
        ac0_out[i] = sqr_clipped_relu(fc0_out[i]);
        ac0_out[L2_SMALL + i] = clipped_relu(fc0_out[i] >> 6);
    }

    int32_t fc1_out[L3_SMALL];
    for (int i = 0; i < L3_SMALL; i++) {
        fc1_out[i] = small_fc1_biases[bucket][i];
        for (int j = 0; j < 30; j++) {
            fc1_out[i] += ac0_out[j] * small_fc1_weights[bucket][i * 32 + j];
        }
    }

    int32_t ac1_out[L3_SMALL];
    for (int i = 0; i < L3_SMALL; i++) {
        ac1_out[i] = clipped_relu(fc1_out[i] >> 6);
    }

    int32_t fc2_out = small_fc2_biases[bucket][0];
    for (int i = 0; i < L3_SMALL; i++) {
        fc2_out += ac1_out[i] * small_fc2_weights[bucket][i];
    }

    int32_t fwdOut = (int32_t)((int64_t)fc0_out[L2_SMALL] * (600 * 16) / (127 * 64));
    if (positional_out) *positional_out = (fc2_out + fwdOut) / 16;
    if (psqt_out) *psqt_out = psqt / 16;
}

int evaluateBigNnueWithAccumulator(Position * pos, Accumulator * acc) {
    assert(acc != NULL);
    int p, v;
    evaluateBigNnueWithAccumulatorFull(pos, acc, &p, &v);
    int a = win_rate_scaling(pos);
    return (p + v) * 100 / a;
}

void evaluateBigNnueWithAccumulatorFull(Position * pos, Accumulator * acc, int * psqt_out, int * positional_out) {
    assert(acc != NULL);
    if (!nnue_loaded) {
        if (psqt_out) *psqt_out = 0;
        if (positional_out) *positional_out = 0;
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
    for (int p = 0; p < 2; p++) {
        Color perspective = (p == 0 ? side : !side);
        for (int i = 0; i < L1_BIG / 2; i++) {
            int32_t v0 = acc->big_v[perspective][i] + acc->big_threat_v[perspective][i];
            int32_t v1 = acc->big_v[perspective][L1_BIG / 2 + i] + acc->big_threat_v[perspective][L1_BIG / 2 + i];
            int32_t c0 = max(0, min(255, v0));
            int32_t c1 = max(0, min(255, v1));
            transformed[p * (L1_BIG / 2) + i] = (uint8_t)((c0 * c1) / 512);
        }
    }

    int32_t fc0_out[L2_BIG + 1];
    for (int i = 0; i <= L2_BIG; i++) {
        fc0_out[i] = big_fc0_biases[bucket][i];
    }
    for (int j = 0; j < L1_BIG; j++) {
        int32_t val = transformed[j];
        if (val == 0) continue;
        for (int i = 0; i <= L2_BIG; i++) {
            fc0_out[i] += val * big_fc0_weights[bucket][i * L1_BIG + j];
        }
    }

    int32_t ac0_out[64] = {0}; 
    for (int i = 0; i < L2_BIG; i++) {
        ac0_out[i] = sqr_clipped_relu(fc0_out[i]);
        ac0_out[L2_BIG + i] = clipped_relu(fc0_out[i] >> 6);
    }

    int32_t fc1_out[L3_BIG];
    for (int i = 0; i < L3_BIG; i++) {
        fc1_out[i] = big_fc1_biases[bucket][i];
        for (int j = 0; j < 62; j++) {
            fc1_out[i] += (int32_t)ac0_out[j] * big_fc1_weights[bucket][i * 64 + j];
        }
    }

    int32_t ac1_out[L3_BIG];
    for (int i = 0; i < L3_BIG; i++) {
        ac1_out[i] = clipped_relu(fc1_out[i] >> 6);
    }

    int32_t fc2_out = big_fc2_biases[bucket][0];
    for (int i = 0; i < L3_BIG; i++) {
        fc2_out += ac1_out[i] * big_fc2_weights[bucket][i];
    }

    int32_t fwdOut = (int32_t)((int64_t)fc0_out[L2_BIG] * (600 * 16) / (127 * 64));
    if (positional_out) *positional_out = (fc2_out + fwdOut) / 16;
    if (psqt_out) *psqt_out = psqt / 16;
}

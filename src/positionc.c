#undef addBonus
#undef COLOR
#undef OPPCOLOR

#ifdef PERSPECTIVE_WHITE
#define COLOR WHITE
#define OPPCOLOR BLACK
#else
#define COLOR BLACK
#define OPPCOLOR WHITE
#endif

#ifdef PERSPECTIVE_WHITE
int makeWhiteMove(Variation *variation, const Move move)
#else
int makeBlackMove(Variation *variation, const Move move)
#endif
{
    Position *position = &variation->singlePosition;
    PlyInfo *plyInfo = &variation->plyInfo[variation->ply++];
    const Square from = getFromSquare(move);
    const Square to = getToSquare(move);
    const Piece newPiece = getNewPiece(move);
    const Piece movingPiece = position->piece[from];
    const Piece capturedPiece = position->piece[to];
    const Bitboard minTo = minValue[to];
    const Bitboard maxFrom = maxValue[from];
    int result = 0;

    assert(to == from || pieceType(capturedPiece) != KING);

    variation->positionHistory[POSITION_HISTORY_OFFSET - 1 + variation->ply] = plyInfo->hashKey = position->hashKey;
    plyInfo->currentMove = move;
    variation->plyInfo[variation->ply].staticValueAvailable = FALSE;
    variation->plyInfo[variation->ply].gainsUpdated = FALSE;
    position->hashKey = ~position->hashKey;

    if (position->enPassantSquare != NO_SQUARE) {
        position->hashKey ^= GENERATED_KEYTABLE[0][position->enPassantSquare];
    }

    position->enPassantSquare = NO_SQUARE;
    position->activeColor = OPPCOLOR;

    if (to == from) {
        variation->plyInfo[variation->ply].accumulator = plyInfo->accumulator;
        variation->plyInfo[variation->ply].staticValueAvailable = FALSE;
        variation->plyInfo[variation->ply].gainsUpdated = FALSE;

        assert(checkVariation(variation) == 0);

        return result; /* Nullmove */
    }

    plyInfo->captured = capturedPiece;
    position->piecesOfColor[COLOR] &= maxFrom;
    position->piecesOfColor[COLOR] |= minTo;
    position->piecesOfType[movingPiece] &= maxFrom;
    position->hashKey ^= GENERATED_KEYTABLE[movingPiece][from] ^ GENERATED_KEYTABLE[movingPiece][to];
    position->castlingRights &= remainingCastlings[to] & remainingCastlings[from];

    if (position->castlingRights != plyInfo->castlingRights) {
        position->hashKey ^=
            GENERATED_KEYTABLE[0][plyInfo->castlingRights] ^ GENERATED_KEYTABLE[0][position->castlingRights];
    }

    position->halfMoveClock++;
    position->piece[to] = movingPiece;
    position->piece[from] = NO_PIECE;

    if (capturedPiece != NO_PIECE) {
        position->halfMoveClock = 0;
        position->piecesOfColor[OPPCOLOR] &= ~minTo;
        position->piecesOfType[capturedPiece] &= ~minTo;
        position->numberOfPieces[OPPCOLOR]--;
        position->materialBalance -= sfBalanceValue[capturedPiece];
        position->materialCount -= sfMaterialValue[capturedPiece];

        if (pieceType(capturedPiece) == PAWN) {
            position->numberOfPawns[OPPCOLOR]--;
        }

        position->hashKey ^= GENERATED_KEYTABLE[capturedPiece][to];
    }

    if (pieceType(movingPiece) == PAWN) {
        position->halfMoveClock = 0;

        if (distance(from, to) == 2) {
            position->enPassantSquare = (Square)((from + to) >> 1);
            position->hashKey ^= GENERATED_KEYTABLE[0][position->enPassantSquare];
        } else if (to == plyInfo->enPassantSquare) {
            const Square captureSquare = (Square)(to + (rank(from) - rank(to)) * 8);
            const Piece capturedPawn = position->piece[captureSquare];

            clearSquare(position->piecesOfColor[OPPCOLOR], captureSquare);
            clearSquare(position->piecesOfType[capturedPawn], captureSquare);
            position->hashKey ^= GENERATED_KEYTABLE[capturedPawn][captureSquare];

            plyInfo->restoreSquare1 = captureSquare;
            plyInfo->restorePiece1 = capturedPawn;
            position->piece[captureSquare] = NO_PIECE;
            position->numberOfPieces[OPPCOLOR]--;
            position->numberOfPawns[OPPCOLOR]--;
            position->materialBalance -= sfBalanceValue[capturedPawn];
            position->materialCount -= sfMaterialValue[capturedPawn];
        } else if (newPiece != NO_PIECE) {
            const Piece effectiveNewPiece = (Piece)(newPiece | COLOR);

            plyInfo->restoreSquare1 = from;
            plyInfo->restorePiece1 = movingPiece;
            position->piece[to] = effectiveNewPiece;
            position->numberOfPawns[COLOR]--;
            position->materialBalance += sfBalanceValue[effectiveNewPiece] - sfBalanceValue[movingPiece];
            position->materialCount += sfMaterialValue[effectiveNewPiece] - sfMaterialValue[movingPiece];
            position->hashKey ^= GENERATED_KEYTABLE[movingPiece][to] ^ GENERATED_KEYTABLE[effectiveNewPiece][to];
            setSquare(position->piecesOfType[position->piece[to]], to);
        }
    } else if (pieceType(movingPiece) == KING) {
        position->king[COLOR] = to;

        if (distance(from, to) == 2) {
            const Square rookFrom = rookOrigin[to];
            const Square rookTo = (Square)((from + to) >> 1);
            const Piece movingRook = position->piece[rookFrom];

            plyInfo->restoreSquare1 = rookFrom;
            plyInfo->restorePiece1 = movingRook;
            plyInfo->restoreSquare2 = rookTo;
            plyInfo->restorePiece2 = position->piece[rookTo];
            position->piece[rookFrom] = NO_PIECE;
            position->piece[rookTo] = movingRook;
            position->halfMoveClock = 0;

            setSquare(position->piecesOfColor[COLOR], rookTo);
            clearSquare(position->piecesOfColor[COLOR], rookFrom);
            setSquare(position->piecesOfType[movingRook], rookTo);
            clearSquare(position->piecesOfType[movingRook], rookFrom);
            position->hashKey ^= GENERATED_KEYTABLE[movingRook][rookFrom] ^ GENERATED_KEYTABLE[movingRook][rookTo];

            if (getDirectAttackers(position, from, OPPCOLOR, position->allPieces) != EMPTY_BITBOARD ||
                getDirectAttackers(position, rookTo, OPPCOLOR, position->allPieces) != EMPTY_BITBOARD) {
                result = 1; /* castling move was not legal */
            }
        }
    }

    setSquare(position->piecesOfType[position->piece[to]], to);
    position->allPieces = position->piecesOfColor[WHITE] | position->piecesOfColor[BLACK];
    {
        bool needsRefreshFor[2] = { FALSE, FALSE };

        if (pieceType(movingPiece) == KING) {
            if (distance(from, to) == 2) {
                needsRefreshFor[0] = needsRefreshFor[1] = TRUE; /* castling: both sides refresh (rook also moves) */
            } else if (!kingStaysInSameBucket(from, to, COLOR)) {
                needsRefreshFor[COLOR] = TRUE; /* king moved to a different bucket/orientation */
            }
        } else if (to == plyInfo->enPassantSquare && pieceType(movingPiece) == PAWN) {
            needsRefreshFor[0] = needsRefreshFor[1] = TRUE; /* en-passant */
        } else if (newPiece != NO_PIECE) {
            needsRefreshFor[0] = needsRefreshFor[1] = TRUE; /* promotion */
        }

        Accumulator *nextAcc = &variation->plyInfo[variation->ply].accumulator;

        if (needsRefreshFor[0] && needsRefreshFor[1]) {
            refreshAccumulator(position, nextAcc, &variation->finnyTable);
        } else if (needsRefreshFor[COLOR]) {
            /* King moved to different bucket: Finny refresh for active side, incremental for opponent. */
            Square opp_added_sq[1], opp_removed_sq[2];
            Piece opp_added_pc[1], opp_removed_pc[2];
            int opp_added_cnt = 0, opp_removed_cnt = 0;

            opp_removed_sq[opp_removed_cnt] = from;
            opp_removed_pc[opp_removed_cnt++] = movingPiece;
            if (capturedPiece != NO_PIECE) {
                opp_removed_sq[opp_removed_cnt] = to;
                opp_removed_pc[opp_removed_cnt++] = capturedPiece;
            }
            opp_added_sq[opp_added_cnt] = to;
            opp_added_pc[opp_added_cnt++] = position->piece[to];

            memcpy(nextAcc, &plyInfo->accumulator, sizeof(Accumulator));
            refreshAccumulatorOneSide(position, nextAcc, &variation->finnyTable, COLOR);
            updateAccumulatorOneSide(nextAcc, opp_added_cnt, opp_added_sq, opp_added_pc,
                                     opp_removed_cnt, opp_removed_sq, opp_removed_pc,
                                     position->king, position, &variation->finnyTable, OPPCOLOR);
        } else {
            Square added_sq[2], removed_sq[3];
            Piece added_pc[2], removed_pc[3];
            int added_cnt = 0, removed_cnt = 0;

            removed_sq[removed_cnt] = from;
            removed_pc[removed_cnt++] = movingPiece;
            if (capturedPiece != NO_PIECE) {
                removed_sq[removed_cnt] = to;
                removed_pc[removed_cnt++] = capturedPiece;
            }
            added_sq[added_cnt] = to;
            added_pc[added_cnt++] = position->piece[to];
            updateAccumulator(&plyInfo->accumulator, nextAcc, added_cnt,
                              added_sq, added_pc, removed_cnt, removed_sq, removed_pc, position->king, position,
                              &variation->finnyTable);
        }
    }

    assert(checkVariation(variation) == 0);

    return result;
}

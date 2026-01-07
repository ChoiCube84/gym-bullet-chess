import numpy as np
import chess


def get_board_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a chess.Board to an (8, 8, 12) float32 tensor.
    Layers 0-5: White P, N, B, R, Q, K
    Layers 6-11: Black P, N, B, R, Q, K
    """
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # 0-based rank and file
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Determine layer index
            layer_offset = 0 if piece.color == chess.WHITE else 6
            # piece_type: P=1...K=6 -> index 0...5
            layer = layer_offset + (piece.piece_type - 1)

            tensor[rank, file, layer] = 1.0

    return tensor


def get_state_vector(
    board: chess.Board, white_time: float, black_time: float, max_time: float = 60.0
) -> np.ndarray:
    """
    Returns an 8-dim vector:
    [Turn(W=1,B=0), WK_Castle, WQ_Castle, BK_Castle, BQ_Castle, EnPassant, W_Time_Norm, B_Time_Norm]
    """
    state = np.zeros(8, dtype=np.float32)

    # 0. Turn
    state[0] = 1.0 if board.turn == chess.WHITE else 0.0

    # 1-4. Castling Rights
    state[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    state[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    state[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    state[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # 5. En Passant availability
    state[5] = 1.0 if board.ep_square is not None else 0.0

    # 6-7. Normalized Time
    state[6] = max(0.0, white_time / max_time)
    state[7] = max(0.0, black_time / max_time)

    return state


def decode_action_to_squares(action_idx: int) -> tuple[int, int]:
    """
    Decodes int(0..4095) to (from_square, to_square).
    """
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    return from_sq, to_sq


def int_to_move(action_idx: int, board: chess.Board) -> chess.Move:
    """
    Converts action index to chess.Move, handling auto-Queen promotion.
    """
    from_sq, to_sq = decode_action_to_squares(action_idx)

    promotion = None
    piece = board.piece_at(from_sq)

    # Check for promotion condition
    if piece and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(to_sq)
        # White promotes on rank 7 (8th rank), Black on rank 0 (1st rank)
        if (piece.color == chess.WHITE and rank == 7) or (
            piece.color == chess.BLACK and rank == 0
        ):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)

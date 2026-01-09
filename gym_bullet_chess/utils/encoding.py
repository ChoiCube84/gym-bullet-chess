import numpy as np
import chess


def get_board_tensor(board: chess.Board) -> np.ndarray:
    """
    Converts a chess.Board to an (8, 8, 12) float32 tensor suitable for CNNs.
    
    The board is viewed from White's perspective (rank 0 is White's 1st rank).
    
    Layers (Channel last):
        0: White Pawns
        1: White Knights
        2: White Bishops
        3: White Rooks
        4: White Queens
        5: White King
        6: Black Pawns
        7: Black Knights
        8: Black Bishops
        9: Black Rooks
        10: Black Queens
        11: Black King
        
    Args:
        board (chess.Board): The python-chess board object.
        
    Returns:
        np.ndarray: Shape (8, 8, 12) with values 0.0 or 1.0.
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
    Returns an 8-dim vector representing global game state and clocks.
    
    Indices:
        0: Turn (1.0 = White, 0.0 = Black)
        1: White King-side Castle (1.0 = Yes)
        2: White Queen-side Castle (1.0 = Yes)
        3: Black King-side Castle (1.0 = Yes)
        4: Black Queen-side Castle (1.0 = Yes)
        5: En Passant Available (1.0 = Yes)
        6: White Time (Normalized 0-1)
        7: Black Time (Normalized 0-1)
        
    Args:
        board (chess.Board): The current board.
        white_time (float): Remaining seconds for White.
        black_time (float): Remaining seconds for Black.
        max_time (float): The initial time control (e.g., 60.0).
        
    Returns:
        np.ndarray: Shape (8,) float32 vector.
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
    Decodes a discrete action index (0..4095) into (from_square, to_square).
    
    Args:
        action_idx (int): The integer action.
        
    Returns:
        tuple[int, int]: (from_square, to_square) indices (0-63).
    """
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    return from_sq, to_sq


def int_to_move(action_idx: int, board: chess.Board) -> chess.Move:
    """
    Converts an action index to a chess.Move object.
    
    Handles automatic promotion to Queen. If a pawn moves to the last rank,
    it is assumed to promote to a Queen. This simplifies the action space.
    
    Args:
        action_idx (int): The integer action.
        board (chess.Board): The board context (needed to check for pawn moves).
        
    Returns:
        chess.Move: The corresponding python-chess Move object.
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

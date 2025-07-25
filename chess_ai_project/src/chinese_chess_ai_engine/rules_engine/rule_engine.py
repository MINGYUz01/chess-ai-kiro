"""
象棋规则引擎

实现各棋子的走法生成、合法性验证和终局状态检测。
"""

from typing import List, Tuple, Optional, Set, Dict
import numpy as np
from .chess_board import ChessBoard
from .move import Move


class RuleEngine:
    """
    象棋规则引擎
    
    负责生成合法走法、验证走法合法性、检测终局状态等。
    """
    
    def __init__(self):
        """初始化规则引擎"""
        # 棋子移动方向定义
        self.king_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 帅/将：上下左右
        self.advisor_moves = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # 仕/士：斜向
        self.bishop_moves = [(2, 2), (2, -2), (-2, 2), (-2, -2)]  # 相/象：田字
        self.knight_moves = [  # 马：日字
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        self.knight_blocks = [  # 马腿位置
            (1, 0), (1, 0), (-1, 0), (-1, 0),
            (0, 1), (0, -1), (0, 1), (0, -1)
        ]
        
        # 九宫范围定义
        self.red_palace = [(7, 3), (7, 4), (7, 5), (8, 3), (8, 4), (8, 5), (9, 3), (9, 4), (9, 5)]
        self.black_palace = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
        
        # 相/象的塞象眼位置
        self.bishop_blocks = {
            (2, 2): (1, 1), (2, -2): (1, -1), (-2, 2): (-1, 1), (-2, -2): (-1, -1)
        }
    
    def generate_legal_moves(self, board: ChessBoard, player: Optional[int] = None) -> List[Move]:
        """
        生成指定玩家的所有合法走法
        
        Args:
            board: 当前棋盘状态
            player: 玩家（1: 红方, -1: 黑方），None表示当前玩家
            
        Returns:
            List[Move]: 合法走法列表
        """
        if player is None:
            player = board.current_player
        
        legal_moves = []
        
        # 遍历所有己方棋子
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece != 0 and (piece > 0) == (player > 0):
                    # 生成该棋子的所有可能走法
                    piece_moves = self.generate_piece_moves(board, (row, col))
                    legal_moves.extend(piece_moves)
        
        # 过滤掉会导致自己被将军的走法
        legal_moves = [move for move in legal_moves if not self._would_be_in_check(board, move, player)]
        
        return legal_moves
    
    def generate_piece_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """
        生成指定位置棋子的所有可能走法
        
        Args:
            board: 当前棋盘状态
            pos: 棋子位置
            
        Returns:
            List[Move]: 可能的走法列表
        """
        row, col = pos
        piece = board.board[row, col]
        
        if piece == 0:
            return []
        
        # 根据棋子类型生成走法
        piece_type = abs(piece)
        
        if piece_type == 1:  # 帅/将
            return self._generate_king_moves(board, pos)
        elif piece_type == 2:  # 仕/士
            return self._generate_advisor_moves(board, pos)
        elif piece_type == 3:  # 相/象
            return self._generate_bishop_moves(board, pos)
        elif piece_type == 4:  # 马
            return self._generate_knight_moves(board, pos)
        elif piece_type == 5:  # 车
            return self._generate_rook_moves(board, pos)
        elif piece_type == 6:  # 炮
            return self._generate_cannon_moves(board, pos)
        elif piece_type == 7:  # 兵/卒
            return self._generate_pawn_moves(board, pos)
        
        return []
    
    def _generate_king_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """生成帅/将的走法"""
        row, col = pos
        piece = board.board[row, col]
        moves = []
        
        # 确定九宫范围
        palace = self.red_palace if piece > 0 else self.black_palace
        
        for dr, dc in self.king_moves:
            new_row, new_col = row + dr, col + dc
            
            # 检查是否在九宫内
            if (new_row, new_col) in palace:
                target_piece = board.board[new_row, new_col]
                
                # 不能吃己方棋子
                if target_piece == 0 or (target_piece > 0) != (piece > 0):
                    move = Move(
                        from_pos=(row, col),
                        to_pos=(new_row, new_col),
                        piece=piece,
                        captured_piece=target_piece if target_piece != 0 else None
                    )
                    moves.append(move)
        
        return moves
    
    def _generate_advisor_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """生成仕/士的走法"""
        row, col = pos
        piece = board.board[row, col]
        moves = []
        
        # 确定九宫范围
        palace = self.red_palace if piece > 0 else self.black_palace
        
        for dr, dc in self.advisor_moves:
            new_row, new_col = row + dr, col + dc
            
            # 检查是否在九宫内
            if (new_row, new_col) in palace:
                target_piece = board.board[new_row, new_col]
                
                # 不能吃己方棋子
                if target_piece == 0 or (target_piece > 0) != (piece > 0):
                    move = Move(
                        from_pos=(row, col),
                        to_pos=(new_row, new_col),
                        piece=piece,
                        captured_piece=target_piece if target_piece != 0 else None
                    )
                    moves.append(move)
        
        return moves
    
    def _generate_bishop_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """生成相/象的走法"""
        row, col = pos
        piece = board.board[row, col]
        moves = []
        
        for dr, dc in self.bishop_moves:
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if not (0 <= new_row <= 9 and 0 <= new_col <= 8):
                continue
            
            # 检查是否过河
            if piece > 0 and new_row < 5:  # 红相不能过河
                continue
            if piece < 0 and new_row > 4:  # 黑象不能过河
                continue
            
            # 检查塞象眼
            block_dr, block_dc = self.bishop_blocks[(dr, dc)]
            block_row, block_col = row + block_dr, col + block_dc
            if board.board[block_row, block_col] != 0:  # 象眼被塞
                continue
            
            target_piece = board.board[new_row, new_col]
            
            # 不能吃己方棋子
            if target_piece == 0 or (target_piece > 0) != (piece > 0):
                move = Move(
                    from_pos=(row, col),
                    to_pos=(new_row, new_col),
                    piece=piece,
                    captured_piece=target_piece if target_piece != 0 else None
                )
                moves.append(move)
        
        return moves
    
    def _generate_knight_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """生成马的走法"""
        row, col = pos
        piece = board.board[row, col]
        moves = []
        
        for i, (dr, dc) in enumerate(self.knight_moves):
            new_row, new_col = row + dr, col + dc
            
            # 检查边界
            if not (0 <= new_row <= 9 and 0 <= new_col <= 8):
                continue
            
            # 检查马腿
            block_dr, block_dc = self.knight_blocks[i]
            block_row, block_col = row + block_dr, col + block_dc
            if board.board[block_row, block_col] != 0:  # 马腿被绊
                continue
            
            target_piece = board.board[new_row, new_col]
            
            # 不能吃己方棋子
            if target_piece == 0 or (target_piece > 0) != (piece > 0):
                move = Move(
                    from_pos=(row, col),
                    to_pos=(new_row, new_col),
                    piece=piece,
                    captured_piece=target_piece if target_piece != 0 else None
                )
                moves.append(move)
        
        return moves
    
    def _generate_rook_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """生成车的走法"""
        row, col = pos
        piece = board.board[row, col]
        moves = []
        
        # 四个方向：上下左右
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            for step in range(1, 10):  # 最多9步
                new_row, new_col = row + dr * step, col + dc * step
                
                # 检查边界
                if not (0 <= new_row <= 9 and 0 <= new_col <= 8):
                    break
                
                target_piece = board.board[new_row, new_col]
                
                if target_piece == 0:
                    # 空位，可以移动
                    move = Move(
                        from_pos=(row, col),
                        to_pos=(new_row, new_col),
                        piece=piece
                    )
                    moves.append(move)
                else:
                    # 有棋子，检查是否可以吃
                    if (target_piece > 0) != (piece > 0):  # 敌方棋子
                        move = Move(
                            from_pos=(row, col),
                            to_pos=(new_row, new_col),
                            piece=piece,
                            captured_piece=target_piece
                        )
                        moves.append(move)
                    break  # 无论如何都不能继续前进
        
        return moves
    
    def _generate_cannon_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """生成炮的走法"""
        row, col = pos
        piece = board.board[row, col]
        moves = []
        
        # 四个方向：上下左右
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            found_platform = False
            
            for step in range(1, 10):  # 最多9步
                new_row, new_col = row + dr * step, col + dc * step
                
                # 检查边界
                if not (0 <= new_row <= 9 and 0 <= new_col <= 8):
                    break
                
                target_piece = board.board[new_row, new_col]
                
                if not found_platform:
                    # 还没找到炮台
                    if target_piece == 0:
                        # 空位，可以移动
                        move = Move(
                            from_pos=(row, col),
                            to_pos=(new_row, new_col),
                            piece=piece
                        )
                        moves.append(move)
                    else:
                        # 找到炮台
                        found_platform = True
                else:
                    # 已经找到炮台
                    if target_piece != 0:
                        # 有棋子，检查是否可以吃
                        if (target_piece > 0) != (piece > 0):  # 敌方棋子
                            move = Move(
                                from_pos=(row, col),
                                to_pos=(new_row, new_col),
                                piece=piece,
                                captured_piece=target_piece
                            )
                            moves.append(move)
                        break  # 无论如何都不能继续
        
        return moves
    
    def _generate_pawn_moves(self, board: ChessBoard, pos: Tuple[int, int]) -> List[Move]:
        """生成兵/卒的走法"""
        row, col = pos
        piece = board.board[row, col]
        moves = []
        
        if piece > 0:  # 红兵
            # 向前（向上）
            if row > 0:
                new_row = row - 1
                target_piece = board.board[new_row, col]
                if target_piece == 0 or target_piece < 0:  # 空位或敌方棋子
                    move = Move(
                        from_pos=(row, col),
                        to_pos=(new_row, col),
                        piece=piece,
                        captured_piece=target_piece if target_piece != 0 else None
                    )
                    moves.append(move)
            
            # 过河后可以左右移动
            if row <= 4:  # 已过河
                for dc in [-1, 1]:  # 左右
                    new_col = col + dc
                    if 0 <= new_col <= 8:
                        target_piece = board.board[row, new_col]
                        if target_piece == 0 or target_piece < 0:  # 空位或敌方棋子
                            move = Move(
                                from_pos=(row, col),
                                to_pos=(row, new_col),
                                piece=piece,
                                captured_piece=target_piece if target_piece != 0 else None
                            )
                            moves.append(move)
        
        else:  # 黑卒
            # 向前（向下）
            if row < 9:
                new_row = row + 1
                target_piece = board.board[new_row, col]
                if target_piece == 0 or target_piece > 0:  # 空位或敌方棋子
                    move = Move(
                        from_pos=(row, col),
                        to_pos=(new_row, col),
                        piece=piece,
                        captured_piece=target_piece if target_piece != 0 else None
                    )
                    moves.append(move)
            
            # 过河后可以左右移动
            if row >= 5:  # 已过河
                for dc in [-1, 1]:  # 左右
                    new_col = col + dc
                    if 0 <= new_col <= 8:
                        target_piece = board.board[row, new_col]
                        if target_piece == 0 or target_piece > 0:  # 空位或敌方棋子
                            move = Move(
                                from_pos=(row, col),
                                to_pos=(row, new_col),
                                piece=piece,
                                captured_piece=target_piece if target_piece != 0 else None
                            )
                            moves.append(move)
        
        return moves
    
    def is_legal_move(self, board: ChessBoard, move: Move) -> bool:
        """
        验证走法是否合法
        
        Args:
            board: 当前棋盘状态
            move: 要验证的走法
            
        Returns:
            bool: 是否合法
        """
        # 基本格式检查
        if not self._is_valid_move_format(move):
            return False
        
        # 检查起始位置是否有对应的棋子
        from_row, from_col = move.from_pos
        actual_piece = board.board[from_row, from_col]
        if actual_piece != move.piece:
            return False
        
        # 检查是否轮到该玩家
        if (actual_piece > 0) != (board.current_player > 0):
            return False
        
        # 生成该棋子的所有合法走法
        legal_moves = self.generate_piece_moves(board, move.from_pos)
        
        # 检查该走法是否在合法走法中
        for legal_move in legal_moves:
            if (legal_move.from_pos == move.from_pos and 
                legal_move.to_pos == move.to_pos):
                # 检查是否会导致自己被将军
                if self._would_be_in_check(board, move, board.current_player):
                    return False
                return True
        
        return False
    
    def _would_be_in_check(self, board: ChessBoard, move: Move, player: int) -> bool:
        """
        检查执行走法后是否会导致自己被将军
        
        Args:
            board: 当前棋盘状态
            move: 要执行的走法
            player: 玩家
            
        Returns:
            bool: 是否会被将军
        """
        # 执行走法
        new_board = board.make_move(move)
        
        # 检查是否被将军
        return self.is_in_check(new_board, player)
    
    def is_in_check(self, board: ChessBoard, player: int) -> bool:
        """
        检查指定玩家是否被将军
        
        Args:
            board: 棋盘状态
            player: 玩家
            
        Returns:
            bool: 是否被将军
        """
        # 找到己方的王
        king_pos = board.find_king(player)
        if not king_pos:
            return False  # 没有王，不可能被将军
        
        # 检查敌方是否有棋子能攻击到王
        enemy_player = -player
        
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece != 0 and (piece > 0) == (enemy_player > 0):
                    # 敌方棋子，检查是否能攻击到王
                    if self._can_attack(board, (row, col), king_pos):
                        return True
        
        return False
    
    def _can_attack(self, board: ChessBoard, attacker_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> bool:
        """
        检查攻击者是否能攻击到目标位置
        
        Args:
            board: 棋盘状态
            attacker_pos: 攻击者位置
            target_pos: 目标位置
            
        Returns:
            bool: 是否能攻击到
        """
        # 生成攻击者的所有可能走法
        possible_moves = self.generate_piece_moves(board, attacker_pos)
        
        # 检查是否有走法能到达目标位置
        for move in possible_moves:
            if move.to_pos == target_pos:
                return True
        
        return False
    
    def is_checkmate(self, board: ChessBoard, player: int) -> bool:
        """
        检查指定玩家是否被将死
        
        Args:
            board: 棋盘状态
            player: 玩家
            
        Returns:
            bool: 是否被将死
        """
        # 首先必须被将军
        if not self.is_in_check(board, player):
            return False
        
        # 检查是否有任何合法走法可以解除将军
        legal_moves = self.generate_legal_moves(board, player)
        return len(legal_moves) == 0
    
    def is_stalemate(self, board: ChessBoard, player: int) -> bool:
        """
        检查指定玩家是否被困毙
        
        Args:
            board: 棋盘状态
            player: 玩家
            
        Returns:
            bool: 是否被困毙
        """
        # 困毙：没有被将军，但没有合法走法
        if self.is_in_check(board, player):
            return False
        
        legal_moves = self.generate_legal_moves(board, player)
        return len(legal_moves) == 0
    
    def get_game_status(self, board: ChessBoard) -> Dict[str, any]:
        """
        获取游戏状态
        
        Args:
            board: 棋盘状态
            
        Returns:
            Dict: 游戏状态信息
        """
        current_player = board.current_player
        opponent = -current_player
        
        # 检查当前玩家状态
        current_in_check = self.is_in_check(board, current_player)
        current_checkmate = self.is_checkmate(board, current_player)
        current_stalemate = self.is_stalemate(board, current_player)
        
        # 检查对手状态
        opponent_in_check = self.is_in_check(board, opponent)
        opponent_checkmate = self.is_checkmate(board, opponent)
        opponent_stalemate = self.is_stalemate(board, opponent)
        
        # 生成合法走法
        legal_moves = self.generate_legal_moves(board, current_player)
        
        status = {
            'current_player': current_player,
            'current_player_name': '红方' if current_player == 1 else '黑方',
            'in_check': current_in_check,
            'checkmate': current_checkmate,
            'stalemate': current_stalemate,
            'game_over': current_checkmate or current_stalemate or opponent_checkmate or opponent_stalemate,
            'winner': None,
            'legal_moves_count': len(legal_moves),
            'legal_moves': legal_moves[:10] if len(legal_moves) > 10 else legal_moves  # 只返回前10个走法
        }
        
        # 确定胜负
        if current_checkmate:
            status['winner'] = opponent
            status['winner_name'] = '黑方' if opponent == -1 else '红方'
            status['end_reason'] = '将死'
        elif opponent_checkmate:
            status['winner'] = current_player
            status['winner_name'] = '红方' if current_player == 1 else '黑方'
            status['end_reason'] = '将死'
        elif current_stalemate or opponent_stalemate:
            status['end_reason'] = '困毙和棋'
        
        return status
    
    def _is_valid_move_format(self, move: Move) -> bool:
        """
        检查走法格式是否有效
        
        Args:
            move: 要检查的走法
            
        Returns:
            bool: 格式是否有效
        """
        try:
            # 检查位置坐标
            from_row, from_col = move.from_pos
            to_row, to_col = move.to_pos
            
            # 坐标范围检查
            if not (0 <= from_row <= 9 and 0 <= from_col <= 8):
                return False
            if not (0 <= to_row <= 9 and 0 <= to_col <= 8):
                return False
            
            # 检查棋子类型
            if move.piece == 0:
                return False
            
            # 检查起始位置和目标位置不同
            if move.from_pos == move.to_pos:
                return False
            
            return True
            
        except (TypeError, ValueError):
            return False
    
    # ==================== 特殊规则处理 ====================
    
    def check_perpetual_check(self, board: ChessBoard, move_history: List[Move], max_checks: int = 3) -> bool:
        """
        检查长将（连续将军）
        
        Args:
            board: 当前棋盘状态
            move_history: 走法历史
            max_checks: 最大连续将军次数
            
        Returns:
            bool: 是否构成长将
        """
        if len(move_history) < max_checks * 2:
            return False
        
        # 检查最近的几步是否都是将军
        recent_moves = move_history[-max_checks * 2:]
        check_count = 0
        
        # 模拟执行走法，检查是否连续将军
        temp_board = board.copy()
        
        # 撤销最近的走法到检查起点
        for _ in range(len(recent_moves)):
            temp_board = temp_board.undo_move()
        
        # 重新执行走法，统计将军次数
        for i, move in enumerate(recent_moves):
            temp_board = temp_board.make_move(move)
            
            # 检查是否将军（只检查奇数步，即一方的走法）
            if i % 2 == 0:  # 假设检查第一个玩家的走法
                opponent = -move.piece // abs(move.piece)  # 对手
                if self.is_in_check(temp_board, opponent):
                    check_count += 1
        
        return check_count >= max_checks
    
    def check_perpetual_chase(self, board: ChessBoard, move_history: List[Move], max_chases: int = 3) -> bool:
        """
        检查长捉（连续捉子）
        
        Args:
            board: 当前棋盘状态
            move_history: 走法历史
            max_chases: 最大连续捉子次数
            
        Returns:
            bool: 是否构成长捉
        """
        if len(move_history) < max_chases * 2:
            return False
        
        # 检查最近的几步是否都在捉同一个子
        recent_moves = move_history[-max_chases * 2:]
        chase_targets = []
        
        # 模拟执行走法，检查捉子情况
        temp_board = board.copy()
        
        # 撤销最近的走法到检查起点
        for _ in range(len(recent_moves)):
            temp_board = temp_board.undo_move()
        
        # 重新执行走法，检查捉子
        for i, move in enumerate(recent_moves):
            if i % 2 == 0:  # 只检查一方的走法
                # 检查这步走法是否在捉子
                chased_pieces = self._get_chased_pieces(temp_board, move)
                chase_targets.append(chased_pieces)
            
            temp_board = temp_board.make_move(move)
        
        # 检查是否连续捉同一个子
        if len(chase_targets) >= max_chases:
            # 找到共同的被捉棋子
            common_targets = set(chase_targets[0])
            for targets in chase_targets[1:max_chases]:
                common_targets &= set(targets)
            
            return len(common_targets) > 0
        
        return False
    
    def _get_chased_pieces(self, board: ChessBoard, move: Move) -> List[Tuple[int, int]]:
        """
        获取走法执行后被捉的棋子位置
        
        Args:
            board: 棋盘状态
            move: 走法
            
        Returns:
            List[Tuple[int, int]]: 被捉棋子的位置列表
        """
        # 执行走法
        new_board = board.make_move(move)
        chased_pieces = []
        
        # 检查移动后的棋子能攻击到哪些敌方棋子
        piece_moves = self.generate_piece_moves(new_board, move.to_pos)
        
        for piece_move in piece_moves:
            target_pos = piece_move.to_pos
            target_piece = new_board.board[target_pos[0], target_pos[1]]
            
            # 如果能吃到敌方棋子，且该棋子没有足够保护，则算作被捉
            if target_piece != 0 and (target_piece > 0) != (move.piece > 0):
                if self._is_piece_undefended(new_board, target_pos):
                    chased_pieces.append(target_pos)
        
        return chased_pieces
    
    def _is_piece_undefended(self, board: ChessBoard, pos: Tuple[int, int]) -> bool:
        """
        检查棋子是否缺乏保护
        
        Args:
            board: 棋盘状态
            pos: 棋子位置
            
        Returns:
            bool: 是否缺乏保护
        """
        row, col = pos
        piece = board.board[row, col]
        
        if piece == 0:
            return True
        
        player = 1 if piece > 0 else -1
        
        # 统计攻击该位置的敌方棋子数量
        attackers = 0
        defenders = 0
        
        for r in range(10):
            for c in range(9):
                board_piece = board.board[r, c]
                if board_piece == 0:
                    continue
                
                if self._can_attack(board, (r, c), pos):
                    if (board_piece > 0) == (player > 0):
                        defenders += 1
                    else:
                        attackers += 1
        
        # 简单的保护判断：攻击者多于保护者
        return attackers > defenders
    
    def check_draw_conditions(self, board: ChessBoard) -> Tuple[bool, str]:
        """
        检查和棋条件
        
        Args:
            board: 棋盘状态
            
        Returns:
            Tuple[bool, str]: (是否和棋, 和棋原因)
        """
        # 1. 困毙和棋
        current_player = board.current_player
        if self.is_stalemate(board, current_player):
            return True, "困毙和棋"
        
        # 2. 重复局面和棋
        if board.is_repetition(max_repetitions=3):
            return True, "重复局面和棋"
        
        # 3. 无子可动和棋（双方都没有攻击性棋子）
        if self._is_insufficient_material(board):
            return True, "子力不足和棋"
        
        # 4. 长将和棋
        if len(board.move_history) >= 6:
            if self.check_perpetual_check(board, board.move_history):
                return True, "长将和棋"
        
        # 5. 长捉和棋
        if len(board.move_history) >= 6:
            if self.check_perpetual_chase(board, board.move_history):
                return True, "长捉和棋"
        
        # 6. 60回合无吃子和棋
        moves_since_capture = (board.metadata.get('round_count', 0) - 
                              board.metadata.get('last_capture_round', 0))
        if moves_since_capture >= 120:  # 60回合 = 120步
            return True, "60回合无吃子和棋"
        
        return False, ""
    
    def _is_insufficient_material(self, board: ChessBoard) -> bool:
        """
        检查是否子力不足
        
        Args:
            board: 棋盘状态
            
        Returns:
            bool: 是否子力不足
        """
        # 统计双方棋子
        red_pieces = []
        black_pieces = []
        
        for row in range(10):
            for col in range(9):
                piece = board.board[row, col]
                if piece > 0:
                    red_pieces.append(abs(piece))
                elif piece < 0:
                    black_pieces.append(abs(piece))
        
        # 检查是否只剩下帅/将和少量弱子
        def has_insufficient_material(pieces):
            # 移除帅/将
            pieces = [p for p in pieces if p != 1]
            
            # 如果没有其他棋子，肯定不足
            if not pieces:
                return True
            
            # 如果只有仕/士、相/象，也不足
            weak_pieces = [2, 3]  # 仕/士、相/象
            if all(p in weak_pieces for p in pieces):
                return True
            
            return False
        
        return has_insufficient_material(red_pieces) and has_insufficient_material(black_pieces)
    
    def get_move_threats(self, board: ChessBoard, move: Move) -> Dict[str, List[Tuple[int, int]]]:
        """
        分析走法的威胁情况
        
        Args:
            board: 棋盘状态
            move: 走法
            
        Returns:
            Dict: 威胁分析结果
        """
        # 执行走法
        new_board = board.make_move(move)
        
        threats = {
            'checks': [],      # 将军威胁
            'captures': [],    # 吃子威胁
            'defends': [],     # 保护的棋子
            'attacks': []      # 攻击的位置
        }
        
        # 检查是否将军
        opponent = -move.piece // abs(move.piece)
        if self.is_in_check(new_board, opponent):
            king_pos = new_board.find_king(opponent)
            if king_pos:
                threats['checks'].append(king_pos)
        
        # 分析移动后棋子的攻击范围
        piece_moves = self.generate_piece_moves(new_board, move.to_pos)
        
        for piece_move in piece_moves:
            target_pos = piece_move.to_pos
            target_piece = new_board.board[target_pos[0], target_pos[1]]
            
            if target_piece != 0:
                if (target_piece > 0) != (move.piece > 0):
                    # 攻击敌方棋子
                    threats['captures'].append(target_pos)
                else:
                    # 保护己方棋子
                    threats['defends'].append(target_pos)
            else:
                # 控制空位
                threats['attacks'].append(target_pos)
        
        return threats
    
    def evaluate_move_quality(self, board: ChessBoard, move: Move) -> Dict[str, any]:
        """
        评估走法质量
        
        Args:
            board: 棋盘状态
            move: 走法
            
        Returns:
            Dict: 走法评估结果
        """
        evaluation = {
            'move': move,
            'score': 0,
            'threats': self.get_move_threats(board, move),
            'safety': True,
            'tactical_value': 0,
            'positional_value': 0
        }
        
        # 基础分数：吃子价值
        if move.captured_piece:
            piece_values = {1: 1000, 2: 20, 3: 20, 4: 40, 5: 90, 6: 45, 7: 10}
            captured_value = piece_values.get(abs(move.captured_piece), 0)
            evaluation['score'] += captured_value
            evaluation['tactical_value'] += captured_value
        
        # 将军奖励
        if evaluation['threats']['checks']:
            evaluation['score'] += 50
            evaluation['tactical_value'] += 50
        
        # 检查走法后是否安全
        new_board = board.make_move(move)
        if self._is_piece_under_attack(new_board, move.to_pos):
            evaluation['safety'] = False
            evaluation['score'] -= 20
        
        # 位置价值（简单的中心控制）
        to_row, to_col = move.to_pos
        center_distance = abs(to_row - 4.5) + abs(to_col - 4)
        evaluation['positional_value'] = max(0, 10 - center_distance)
        evaluation['score'] += evaluation['positional_value']
        
        return evaluation
    
    def _is_piece_under_attack(self, board: ChessBoard, pos: Tuple[int, int]) -> bool:
        """
        检查指定位置的棋子是否受到攻击
        
        Args:
            board: 棋盘状态
            pos: 棋子位置
            
        Returns:
            bool: 是否受到攻击
        """
        row, col = pos
        piece = board.board[row, col]
        
        if piece == 0:
            return False
        
        # 检查敌方棋子是否能攻击到该位置
        enemy_player = -1 if piece > 0 else 1
        
        for r in range(10):
            for c in range(9):
                enemy_piece = board.board[r, c]
                if enemy_piece != 0 and (enemy_piece > 0) == (enemy_player > 0):
                    if self._can_attack(board, (r, c), pos):
                        return True
        
        return False
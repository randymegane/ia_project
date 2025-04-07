from player_divercite import PlayerDivercite
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from game_state_divercite import GameStateDivercite
from seahorse.game.game_layout.board import Piece
from seahorse.utils.custom_exceptions import MethodNotImplementedError

class MyPlayer(PlayerDivercite):
    """
    Player class for Divercite game that makes random moves.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "MyPlayer"):
        """
        Initialize the PlayerDivercite instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name)
        # Add any information you want to store about the player here
        # self.json_additional_info = {}

    def compute_action(self, current_state: GameStateDivercite, remaining_time: int = 1e9, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameState): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        if remaining_time > 10_000: 
            depth = 3
        elif remaining_time > 3_000:  
            depth = 2
        else:
            depth = 1  

        possible_actions = current_state.generate_possible_heavy_actions()

        best_value = float("-inf")
        best_action = None

        for action in possible_actions:
            next_state = action.get_next_game_state()
            value, _ = self.min_value(next_state, depth - 1, remaining_time, float("-inf"), float("inf"))

            if value > best_value:
                best_value = value
                best_action = action

        return best_action
    
    
    def max_value(self, state: GameState, depth: int, remaining_time: int, alpha: float, beta: float) -> float:
            if depth == 0 or state.is_done():
                return (self.utility(state, remaining_time), None)

            value_star = float('-inf')
            best_action = None

            actions = sorted(state.generate_possible_heavy_actions(), key=lambda a: self.utility(a.get_next_game_state(), remaining_time))

            for action in actions:
                next_state = action.get_next_game_state()
                (value,_) = self.min_value(next_state, depth - 1, remaining_time, alpha, beta)

                if value > value_star:
                    value_star = value
                    best_action = action

                if value_star >= beta:
                    break 
                alpha = max(alpha, value_star)
            
            return (value_star, best_action)


    def min_value(self, state: GameState, depth: int, remaining_time, alpha: float, beta: float) -> float:
            if depth == 0 or state.is_done():
                return (self.utility(state, remaining_time), None)

            value_star = float('inf')
            best_action = None

            actions = sorted(state.generate_possible_heavy_actions(), key=lambda a: self.utility(a.get_next_game_state(), remaining_time))
            
            for action in actions:
                next_state = action.get_next_game_state()
                (value,_) = self.max_value(next_state, depth - 1,remaining_time, alpha, beta)

                if value < value_star:
                    value_star = value
                    best_action = action
                
                if value_star <= alpha:
                    break  # 💥 Couper la branche
                beta = min(beta, value_star)
            
            return (value_star, best_action)
        
        
    def utility(self, state: GameStateDivercite, remaining_time: int) -> float:

        # calcule le score de l'agent par rapport à son adversaire
        player_id = self.get_id()
        player_score = state.scores[player_id]
        opponent_id = [pid for pid in state.scores if pid != self.get_id()][0]
        opponent_score = state.scores[opponent_id] 
        score = player_score - opponent_score

        # facteur de temps
        time_factor = (remaining_time / 10) * 0.05
        score += time_factor


        return score
    

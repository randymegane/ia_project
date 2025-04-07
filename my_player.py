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

        best_score = float("-inf")
        best_action = None

        for action in current_state.generate_possible_heavy_actions():
            next_state = action.get_next_game_state()
            score = self.max_value(next_state, remaining_time, depth - 1, float("-inf"), float("inf"))
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    
    def minimax(self,state: GameState, depth: int, remaining_time: int, alpha: float, beta: float, maximize: bool) -> float:
            if depth == 0 or state.is_done():
                return (self.utility(state, remaining_time))
            
            if maximize:
                return self.max_value(state, remaining_time, depth, alpha, beta)
            else:
                return self.min_value(state, remaining_time, depth, alpha, beta)
    
    
    def max_value(self, state: GameState, depth: int, remaining_time: int, alpha: float, beta: float) -> float:
            if depth == 0 or state.is_done():
                return (self.utility(state, remaining_time))

            value_star = float('-inf')

            actions = sorted(state.generate_possible_heavy_actions(), key=lambda a: self.utility(a.get_next_game_state(), remaining_time))
            
            for action in actions:
                next_state = action.get_next_game_state()
                value = self.minimax(next_state, remaining_time, depth-1, alpha, beta, False )
                
                value_star = max(value_star, value)
                alpha = max(alpha, value)

                if beta <= alpha:
                     break
 
            return value_star


    def min_value(self, state: GameState, depth: int, remaining_time, alpha: float, beta: float) -> float:
            if depth == 0 or state.is_done():
                return (self.utility(state, remaining_time))

            value_star = float('inf')

            actions = sorted(state.generate_possible_heavy_actions(), key=lambda a: self.utility(a.get_next_game_state(), remaining_time))
            
            for action in actions:
                next_state = action.get_next_game_state()
                value = self.minimax(next_state, remaining_time, depth - 1,remaining_time, alpha, beta, True)
                
                value_star = min(value_star, value)
                beta = min(beta, value)

                if beta <= alpha:
                    break
    
            return value_star
        
        
    def utility(self, state: GameStateDivercite, remaining_time: int) -> float:

        # calcule le score de l'agent par rapport à son adversaire
        player_id = self.get_id()

        player_score = state.scores[player_id]
        opponent_score = sum(state.scores.values()) - player_score
        score = player_score - opponent_score

        # facteur de temps
        time_factor = (remaining_time / 10) * 0.05
        score += time_factor

        return score
    

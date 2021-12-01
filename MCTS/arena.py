import numpy as np

from mcts import MCTS

class Arena:
  def __init__(self, player1, player2, game):
    self.player1 = player1
    self.player2 = player2
    self.game = game

  def play_game(self, nb_simulations):
    state = self.game.get_init_state()
    current_player = np.random.choice([self.player1, self.player2])

    while True:
      mcts = MCTS(self.game, state, current_player, nb_simulations)
      root = mcts.run()
      _, action = root.select_action(0, self.game.nb_actions)
      state = self.game.get_new_state(state, action)
      current_player = self.player2 if current_player == self.player1 else self.player1

      r = self.game.get_reward(state)
      if r is not None:
        if current_player == self.player1: return r
        else: return -r
      
  
  def play_games(self, nb_games, nb_simulations):
    w, d, l = 0,0,0

    for _ in range(nb_games):
      result = self.play_game(nb_simulations)
      if result == 1: w += 1
      elif result == 0: d += 1
      elif result == -1: l += 1
    
    return w, l
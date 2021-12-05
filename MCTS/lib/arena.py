import numpy as np

from lib.mcts import MCTS

class Arena:
  def __init__(self, player1, player2, game):
    self.player1 = player1
    self.player2 = player2
    self.game = game

  def play_game(self, p1, p2, nb_simulations):
    state = self.game.get_init_state()
    current_player = p1

    while True:
      mcts = MCTS(self.game, state, current_player, nb_simulations)
      root = mcts.run()
      _, action = root.select_action(0, self.game.nb_actions)
      state = self.game.get_new_state(state, action)
      current_player = p2 if current_player == p1 else p1

      r = self.game.get_reward(state)
      if r is not None:
        if current_player == self.player1: return r
        else: return -r
      
  
  def play_games(self, nb_games, nb_simulations):
    w, d, l = 0,0,0
    p1, p2 = self.player1, self.player2

    for _ in range(nb_games):
      result = self.play_game(p1, p2, nb_simulations)
      if result == 1: w += 1
      elif result == 0: d += 1
      elif result == -1: l += 1
      p1, p2 = p2, p1
    
    return w, l
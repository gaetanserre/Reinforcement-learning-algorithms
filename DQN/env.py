import gym

class Env:
  def __init__(self):
    self.env = gym.make("CartPole-v1")
    self.old_obs = self.env.reset()
    self.reward_range = self.env.reward_range

    self.nb_nothing = 0
  
  def reset(self):
    self.nb_nothing = 0
    return self.env.reset()
  
  def close(self):
    return self.env.close()
  
  def render(self, **kargs):
    return self.env.render(**kargs)
  
  def step(self, action):
    res = self.env.step(action)
    self.old_obs = res[0]
    return res
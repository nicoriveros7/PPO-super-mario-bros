import gym
import numpy as np
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """
    def __init__(self, env, max_steps=10000):
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        obs, reward, done, info = self.env.step(action)
        if self.current_step >= self.max_steps:
            done = True
            info['time_limit_reached'] = True
        info['Current_Step'] = self.current_step
        return obs, reward, done, info

def main():
    steps = 0
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = TimeLimitWrapper(env, max_steps=2000)

    obs = env.reset()
    print(f"Obs shape: {obs.shape}")

    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        steps += 1
        if steps % 1000 == 0:
            print(f"Total Steps: {steps}")
            print(info)

    print("Final Info:")
    print(info)
    env.close()

if __name__ == "__main__":
    main()
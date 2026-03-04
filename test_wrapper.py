import gym
import gym_super_mario_bros
from stable_baselines3.common.atari_wrappers import WarpFrame

# Crear entorno base
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = WarpFrame(env)

# Probar reset
obs = env.reset()
print("Observación después de WarpFrame:", obs)
print("Tipo de observación:", type(obs))
print("Forma de observación:", obs.shape if obs is not None else None)

env.close()
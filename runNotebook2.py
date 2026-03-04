import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gym.wrappers import GrayScaleObservation
import cv2
import numpy as np
import torch
import time

# Configuración del dispositivo
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Wrapper para reescalar observaciones
class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        super().__init__(env)
        (oldh, oldw, oldc) = env.observation_space.shape
        newshape = (size, size, oldc)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=newshape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        return frame

# Wrapper para saltar frames (ejecutar acción N veces)
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# Wrapper para la recompensa personalizada y manejo de finalización
class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info.get("flag_get", False):
            reward += 500
            done = True
            print("GOAL")
        if info.get("life", 2) < 2:
            reward -= 500
            done = True
        self.current_score = info.get("score", 0)
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info.get("x_pos", 0)
        return state, reward / 10., done, info

# Función para crear el entorno con todos los wrappers necesarios
def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomRewardAndDoneEnv(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeEnv(env, size=84)
    return env

# Crear entorno vectorizado con apilamiento de frames igual que en entrenamiento
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4, channels_order='last')

# Cargar modelo entrenado
model = PPO.load("model/ppo_mario_final.zip", device=device)

# Evaluar el modelo durante 10 episodios
max_steps_per_episode = 5000
for episode in range(10):
    obs = env.reset()
    done = [False]
    total_reward = 0
    step_count = 0
    
    while not done[0] and step_count < max_steps_per_episode:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        step_count += 1
        env.render()
        time.sleep(0.02)

    print(f"Episodio {episode + 1} finalizado con recompensa total: {total_reward:.2f}, "
          f"pasos: {step_count}, x_pos: {info[0].get('x_pos', 'N/A')}, "
          f"flag_get: {info[0].get('flag_get', False)}")
    if info[0].get('flag_get', False):
        print(f"🎉 Nivel completado en el episodio {episode + 1}!")
    elif info[0].get('life', 2) < 2:
        print("El agente perdió todas las vidas.")
    else:
        print("El agente se detuvo antes de completar el nivel.")

env.close()

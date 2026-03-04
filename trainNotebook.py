import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.wrappers import GrayScaleObservation
import torch as th
from torch import nn
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Configuración del dispositivo
device = 'mps' if th.backends.mps.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Parámetros del modelo
CHECK_FREQ_NUMB = 200000
TOTAL_TIMESTEP_NUMB = 5000000
LEARNING_RATE = 0.0001
GAE = 1.0
ENT_COEF = 0.01
N_STEPS = 512
GAMMA = 0.9
BATCH_SIZE = 64
N_EPOCHS = 10

# Parámetros de prueba
EPISODE_NUMBERS = 20
MAX_TIMESTEP_TEST = 1000

# Directorio para guardar modelos y logs
save_dir = Path('./model')
save_dir.mkdir(parents=True, exist_ok=True)
reward_log_path = save_dir / 'reward_log.csv'

# Inicializar archivo de log
with open(reward_log_path, 'a') as f:
    print('timesteps,reward,best_reward', file=f)

# Wrapper para saltar frames
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

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

# Wrapper para recompensa personalizada
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
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

# Arquitectura CNN personalizada
class MarioNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(MarioNet, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcular la dimensión de salida de la CNN
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# Configuración de la política con MarioNet
policy_kwargs = dict(
    features_extractor_class=MarioNet,
    features_extractor_kwargs=dict(features_dim=512),
)

# Callback para guardar modelos y registrar métricas
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = (self.save_path / f'best_model_{self.n_calls}')
            self.model.save(model_path)

            total_reward = [0] * EPISODE_NUMBERS
            total_time = [0] * EPISODE_NUMBERS
            best_reward = 0

            for i in range(EPISODE_NUMBERS):
                state = self.model.env.reset()
                done = False
                total_reward[i] = 0
                total_time[i] = 0
                while not done and total_time[i] < MAX_TIMESTEP_TEST:
                    action, _ = self.model.predict(state)
                    state, reward, done, info = self.model.env.step(action)
                    total_reward[i] += reward[0]
                    total_time[i] += 1

                if total_reward[i] > best_reward:
                    best_reward = total_reward[i]

            print(f'time steps: {self.n_calls}/{TOTAL_TIMESTEP_NUMB}')
            print(f'average reward: {sum(total_reward) / EPISODE_NUMBERS}, '
                  f'average time: {sum(total_time) / EPISODE_NUMBERS}, '
                  f'best_reward: {best_reward}')

            with open(reward_log_path, 'a') as f:
                print(f'{self.n_calls},{sum(total_reward) / EPISODE_NUMBERS},{best_reward}', file=f)

        return True

# Configuración del entorno
STAGE_NAME = 'SuperMarioBros-1-1-v0'
env = gym_super_mario_bros.make(STAGE_NAME)
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = CustomRewardAndDoneEnv(env)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=True)
env = ResizeEnv(env, size=84)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4, channels_order='last')

# Configuración del modelo PPO
model = PPO('CnnPolicy', env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=save_dir,
            learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
            gamma=GAMMA, gae_lambda=GAE, ent_coef=ENT_COEF, device=device)

# Entrenamiento con barra de progreso
callback = TrainAndLoggingCallback(check_freq=CHECK_FREQ_NUMB, save_path=save_dir)
print("------------- Iniciando entrenamiento -------------")
step_block = 100000  # Bloque de timesteps para la barra de progreso
for _ in tqdm(range(TOTAL_TIMESTEP_NUMB // step_block), desc="Entrenando PPO Mario", total=TOTAL_TIMESTEP_NUMB // step_block):
    model.learn(total_timesteps=step_block, callback=callback, reset_num_timesteps=False)
print("------------- Entrenamiento finalizado -------------")

# Guardar el modelo final
model.save(save_dir / "ppo_mario_final")
print("\nModelos guardados en", save_dir)

# Cerrar el entorno
env.close()













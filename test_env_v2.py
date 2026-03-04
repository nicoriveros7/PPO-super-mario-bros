import gym
import gym_super_mario_bros

# Crear entorno básico
STAGE_NAME = 'SuperMarioBros-1-1-v0'  # Versión estándar
env = gym_super_mario_bros.make(STAGE_NAME)
obs = env.reset()

# Verificar observación
print("Observación inicial:", obs)
print("Tipo de observación:", type(obs))
print("Forma de observación:", obs.shape if obs is not None else None)

env.close()
import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo limpio
data = pd.read_csv("model/reward_log.csv")

# Extraer columnas
timesteps = data["timesteps"]
avg_reward = data["reward"]
best_reward = data["best_reward"]

# Crear la gráfica
plt.figure(figsize=(12, 6))
plt.plot(timesteps, avg_reward, label="Recompensa promedio (entrenamiento)", color="blue", linewidth=2)
plt.plot(timesteps, best_reward, label="Mejor recompensa (entrenamiento)", color="orange", linewidth=2)

# Añadir un punto para la recompensa promedio de la evaluación final (nueva iteración)
plt.scatter([5_000_000], [986.90], color="green", s=100, label="Recompensa promedio (evaluación final)", zorder=5)
plt.annotate("Evaluación final: 986.90", (5_000_000, 860.58), textcoords="offset points", xytext=(-50, 20), ha="center", fontsize=10, color="green")

# Configurar etiquetas y título
plt.xlabel("Timesteps (millones)", fontsize=12)
plt.ylabel("Recompensa", fontsize=12)
plt.title("Progreso del entrenamiento de PPO en Super Mario Bros. 1-1", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)

# Ajustar el eje X para mostrar timesteps en millones
plt.xticks(ticks=range(0, 5_500_000, 500_000), labels=[f"{x/1_000_000:.1f}" for x in range(0, 5_500_000, 500_000)], rotation=45)

# Ajustar el eje Y para el rango de las recompensas
plt.ylim(0, 1200)  # Máximo de 1200 para incluir todas las recompensas con margen

# Ajustar el diseño
plt.tight_layout()

# Guardar la gráfica
plt.savefig("model/training_progress_final2.png", dpi=300)
print("Gráfica guardada como 'model/training_progress_final2.png'")

# Mostrar la gráfica
plt.show()
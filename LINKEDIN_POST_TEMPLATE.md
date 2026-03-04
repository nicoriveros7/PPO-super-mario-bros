# Plantilla de post para LinkedIn

## Versión corta (rápida)
Hoy terminé un proyecto de **Reinforcement Learning** donde entrené un agente con **PPO** para jugar **Super Mario Bros 1-1** 🎮

### Lo más importante:
- Política CNN + frame stacking (84x84, 4 frames)
- Reward shaping basado en progreso (`x_pos`) y objetivo (`flag_get`)
- Entrenamiento en Python con `gym-super-mario-bros` y `stable-baselines3`

### Qué aprendí:
- Diseñar bien la recompensa cambia completamente el comportamiento del agente
- Medir progreso por métricas claras acelera iteraciones
- Balancear exploración/explotación requiere bastante tuning

Repositorio: [pon aquí tu link de GitHub]

#ReinforcementLearning #MachineLearning #DeepLearning #Python #PPO #AI #OpenToWork

---

## Versión completa (recomendada)
Después de varias iteraciones, terminé un proyecto de **aprendizaje por refuerzo** para entrenar un agente que juegue **Super Mario Bros 1-1** usando **PPO**.

### Problema
Quería construir un experimento end-to-end: desde preprocesamiento del entorno, diseño de recompensa y entrenamiento, hasta visualización de métricas y evaluación del agente.

### Enfoque técnico
- Algoritmo: PPO (`stable-baselines3`)
- Estado: escala de grises + resize 84x84 + stack de 4 frames
- Entorno: `gym-super-mario-bros`
- Reward shaping:
  - + progreso horizontal (`x_pos`)
  - + bonus por completar nivel (`flag_get`)
  - - penalización por muerte

### Resultado
- Agente funcional, capaz de avanzar de forma consistente en el nivel 1-1
- Pipeline completo: entrenamiento, evaluación y gráficas de progreso
- Código publicado con estructura reproducible en GitHub

### Aprendizajes clave
1. La función de recompensa define gran parte del comportamiento emergente.
2. La observación temporal (stack de frames) mejora decisiones en entornos dinámicos.
3. La instrumentación de métricas es tan importante como el modelo.

Si te interesa RL aplicado en videojuegos o control secuencial, feliz de conectar.

Repositorio: [pon aquí tu link de GitHub]
Demo (video/GIF): [pon aquí tu link]

#ReinforcementLearning #PPO #MachineLearning #ArtificialIntelligence #Python #DeepLearning #DataScience #OpenToWork

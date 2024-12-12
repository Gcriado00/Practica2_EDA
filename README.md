# Practica2_EDA
Guillermo Criado Cabreros: Correo CUNEF: guillermo.criado@cunef.edu GitHub: https://github.com/Gcriado00/Practica1_EDA

Sara Pérez Sánchez: Correo CUNEF: s.perez@cunef.edu GitHub: https://github.com/saraperezsanchez/Practica1_EDA

Introducción:

En el ámbito de la analítica de datos aplicada a problemas reales, la identificación de patrones y la 
predicción de comportamientos son elementos clave para la toma de decisiones informadas. En esta práctica,
el objetivo es abordar un problema de predicción en un contexto financiero, utilizando diversas técnicas 
avanzadas de aprendizaje automático. Específicamente, estamos trabajando con un conjunto de datos desbalanceado 
que presenta una proporción desigual entre las clases objetivo, lo que plantea retos significativos en el
desarrollo de modelos precisos y equitativos.

Para lograr resultados robustos, implementamos y comparamos múltiples algoritmos de clasificación, 
incluyendo métodos basados en árboles de decisión, técnicas de boosting y regularización. A lo largo 
del análisis, se evaluan métricas como precisión, recall y G-Mean, con un enfoque especial en garantizar
que las predicciones sean útiles para identificar instancias de la clase minoritaria, crítica para el 
contexto del problema.

Objetivos:

Desarrollar modelos de clasificación efectivos, implementando y comparando diferentes algoritmos de 
aprendizaje automático, como Random Forest, Gradient Boosting, AdaBoost, XGBoost y LGBM, para abordar el 
problema de predicción.

Manejar datos desbalanceados explorando técnicas que permitan mejorar el desempeño en la clase minoritaria, 
asegurando que los modelos no favorezcan únicamente a la clase mayoritaria.

Optimizar el desempeño de los modelos realizando ajustes de hiperparámetros en los algoritmos seleccionados
para maximizar métricas clave como el G-Mean y el recall, con un balance adecuado entre precisión y sensibilidad.

Interpretar resultados y seleccionar el modelo óptimo, evaluando así la importancia de las variables y 
analizando las predicciones para determinar el modelo que mejor se adapta al problema, priorizando tanto
la precisión global como la detección de la clase minoritaria.

Aplicar técnicas de regularización usando herramientas como Ridge para prevenir el overfitting y seleccionar 
variables relevantes, mejorando la interpretabilidad y la generalización de los modelos.

Conclusión:

Tras analizar los objetivos del proyecto, que era principalmente predecir los clientes que podrían incurrir 
en impagos, vamos a establecer una serie de conclusiones. En primer lugar, la regularización ridge fue una
herramienta importante a la hora de abordar problemas de multicolinealidad y seleccionar características 
relevantes en un conjunto de datos con múltiples variables correlacionadas. Esta técnica permitió reducir 
la complejidad del modelo manteniendo la mayoría de las variables con pesos más bajos, lo que asegura un 
balance adecuado entre simplificación y preservación de información. La selección de variables con Ridge 
nos proporcionó un conjunto de características más manejable y relevante, lo cual sirvió como base para 
entrenar y evaluar diversos modelos de clasificación.

Tras analizar seis modelos diferentes, vimos que cada uno presentaba sus fortalezas y debilidades, aunque 
todos tenían un problema a la hora de detectar la clase minoritaria (riesgo de impago), ya que el recall 
no era el más óptimo en los distintos clasificadores. Modelos como XGBoost o Random Forest, por ejemplo, 
presentaron números más altos de accuracy, aunque muy bajos de recall. GradientBoosting tuvo un mejor 
resultado, con un recall del 26.59% para la clase minoritaria, siendo un posible candidato. Sin embargo, 
AdaBoost, aunque en términos de accuracy no tenía un gran porcentaje, presentó el mejor recall con 41%, 
lo que lo convirtió en el principal candidato, ya que el accuracy no fue tan importante al tener los datos 
desbalanceados (aun habiendo aplicado la técnica SMOTE previamente).

El modelo AdaBoostClassifier fue seleccionado como el mejor candidato tras un riguroso proceso de ajuste 
de hiperparámetros (n_estimators=200, learning_rate=0.2) y evaluación de desempeño. Aunque sacrifica parte 
del rendimiento en la clase mayoritaria, AdaBoost logró un recall significativamente superior para la clase 
minoritaria (46%), lo cual es crítico en un problema de detección de fraudes.Con un G-Mean de 0.694 tras 
optimización, el modelo muestra un mejor equilibrio entre las tasas de verdadero positivo y negativo, lo que 
refuerza su efectividad en escenarios desbalanceados.

Las curvas ROC-AUC, Precision-Recall, Lift, y Cumulative Gains mostraron que AdaBoost es el modelo que mejor 
equilibra la detección de la clase minoritaria sin comprometer en exceso el rendimiento general. La curva 
Precision-Recall destacó la mejora en la sensibilidad hacia la clase 1, mientras que la curva Lift demostró 
que el modelo es capaz de identificar instancias positivas con mayor probabilidad al principio del análisis. 
La curva de Ganancias Acumuladas reflejó que el modelo proporciona un beneficio tangible en la detección de 
la clase 1, aunque sigue existiendo margen para mejorar.

Debido a la incompatibilidad de SHAP con AdaBoost, se utilizó el atributo feature_importances_ para evaluar 
la relevancia de las características. El gráfico de importancia de variables mostró que EXT_SOURCE_3, 
EXT_SOURCE_2, y OBS_30_CNT_SOCIAL_CIRCLE son algunas de las características más influyentes en las decisiones 
del modelo. Aunque esta técnica no es tan detallada como SHAP, proporciona una visión útil y directa de cómo 
el modelo pondera las características en sus predicciones.
# Miguel David Moreno Montañez
# Ejercicio 2: Predicción de la satisfacción del cliente
# Machine Learning
# 02/04/2025
# Observación: entregado a tiempo funciona pero incompleto
# Explicacion: Me apolle en la IA para la generacion del los datos de prueba
# y para la generacion de los graficos y verificacion de posibles errores

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

df = pd.DataFrame({
    "Edad": [25, 40, 30, 35, 50, 28, 45, 33, 38, 42],
    "Genero": ["M", "F", "F", "M", "F", "M", "F", "M", "F", "M"],
    "Historial_Compras": [5, 20, 10, 7, 25, 12, 30, 8, 15, 22],
    "Tiempo_En_Sitio": [10.5, 5.2, 8.3, 7.0, 4.8, 9.1, 6.5, 12.3, 11.0, 5.6],
    "Clicks_Anuncios": [3, 1, 2, 5, 0, 4, 3, 6, 2, 1],
    "Gasto_Promedio": [200, 500, 300, 250, 700, 400, 450, 550, 600, 350],
    "Compra": [1, 1, 0, 0, 1, 1, 0, 1, 1, 0]
})

df = pd.get_dummies(df, drop_first=True)

X_lin = df.drop(columns=['Gasto_Promedio', 'Compra']) 
y_lin = df['Gasto_Promedio']

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.3, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train_lin, y_train_lin)

y_pred_lin = lin_reg.predict(X_test_lin)

mse = mean_squared_error(y_test_lin, y_pred_lin)
print(f"Error cuadrático medio (MSE) del modelo de Regresión Lineal: {mse:.2f}")



X_log = df.drop(columns=['Compra'])
y_log = df['Compra']

scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log_scaled, y_log, test_size=0.3, random_state=42)

log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train_log, y_train_log)

y_pred_log = log_reg.predict(X_test_log)

accuracy = accuracy_score(y_test_log, y_pred_log)
print(f"Precisión del modelo de Regresión Logística: {accuracy:.2f}")

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
sns.histplot(y_pred_lin, bins=10, kde=True)
plt.title("Distribución de Predicciones de Gasto Promedio")

plt.subplot(1,2,2)
sns.heatmap(pd.crosstab(y_test_log, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicción")
plt.ylabel("Realidad")
plt.title("Matriz de Confusión para la Regresión Logística")

plt.show()

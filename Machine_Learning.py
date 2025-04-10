# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)
n = 200
data = pd.DataFrame({
    'Edad': np.random.randint(18, 65, n),
    'Genero': np.random.randint(0, 2, n),
    'HistorialCompras': np.random.randint(0, 20, n),
    'TiempoSitioWeb': np.random.normal(10, 2, n).round(2),
    'Compra': np.random.binomial(1, 0.5, n)
})

data['GastoPromedio'] = data['Compra'] * np.random.uniform(5, 200, n).round(2)

X = data[['Edad', 'Genero', 'HistorialCompras', 'TiempoSitioWeb']]
y_class = data['Compra']
y_reg = data['GastoPromedio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X_scaled, y_class, y_reg, test_size=0.3, random_state=42)


log_model = LogisticRegression()
log_model.fit(X_train, y_class_train)

lin_model = LinearRegression()
lin_model.fit(X_train[y_class_train == 1], y_reg_train[y_class_train == 1])

prob_compra = log_model.predict_proba(X_test)[:, 1]
gasto_estimado = lin_model.predict(X_test)
valor_cliente = prob_compra * gasto_estimado

resultados = pd.DataFrame({
    'Prob_Compra': prob_compra.round(2),
    'Gasto_Pred': gasto_estimado.round(2),
    'Valor_Cliente': valor_cliente.round(2),
    'Compra_Real': y_class_test.values,
    'Gasto_Real': y_reg_test.values
})
print("Primeros resultados:")
print(resultados.head())


print("\n--- Predicción para un nuevo cliente ---")

nuevo_cliente = pd.DataFrame([{
    'Edad': 30,
    'Genero': 1,
    'HistorialCompras': 5,
    'TiempoSitioWeb': 12.5
}])

nuevo_cliente_scaled = scaler.transform(nuevo_cliente)

prob_nueva_compra = log_model.predict_proba(nuevo_cliente_scaled)[0, 1]
gasto_nuevo_estimado = lin_model.predict(nuevo_cliente_scaled)[0]
valor_nuevo_cliente = prob_nueva_compra * gasto_nuevo_estimado

print("\nEvaluación del modelo de Regresión Logística:")
y_class_pred = log_model.predict(X_test)
print("Accuracy:", accuracy_score(y_class_test, y_class_pred))
print("Matriz de confusión:\n", confusion_matrix(y_class_test, y_class_pred))
print(classification_report(y_class_test, y_class_pred))


print(f"Probabilidad de compra: {prob_nueva_compra*100:.2f}%")
print(f"Gasto estimado si compra: ${gasto_nuevo_estimado:.0f}")
print(f"Valor esperado del cliente: ${valor_nuevo_cliente:.2f}")


plt.figure(figsize=(10, 5))
sns.histplot(valor_cliente, bins=20, kde=True)
plt.title("Distribución del Valor Estimado del Cliente")
plt.xlabel("Valor Estimado")
plt.ylabel("Frecuencia")
plt.show()
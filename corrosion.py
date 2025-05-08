# Importamos las bibliotecas necesarias para realizar el análisis
import pandas as pd  # Para manejar datos en forma de tablas (como hojas de cálculo)
import numpy as np  # Para realizar cálculos matemáticos y operaciones numéricas
import matplotlib.pyplot as plt  # Para crear gráficos y visualizar los resultados
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.linear_model import LinearRegression  # Para crear un modelo de regresión lineal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Para evaluar el rendimiento del modelo

# Cargamos los datos desde un archivo llamado 'corrosion-rate.csv'
# Este archivo contiene información sobre cómo el peso de un material cambia con el tiempo debido a la corrosión
data = pd.read_csv('corrosion-rate.csv')  # Asegúrate de que el archivo esté en el mismo directorio que este script

# Mostramos las primeras filas del conjunto de datos para entender su estructura
print("Primeras filas del conjunto de datos:")
print(data.head())  # Esto nos da una idea de cómo están organizados los datos

# Verificamos si hay valores faltantes en el conjunto de datos
# Los valores faltantes pueden causar problemas en el análisis, por lo que debemos identificarlos
print("\n¿Hay valores faltantes?")
print(data.isnull().sum())  # Muestra cuántos valores faltan en cada columna

# Eliminamos las filas que tienen valores faltantes
# Esto asegura que solo trabajemos con datos completos
data = data.dropna()

# Creamos una nueva columna llamada 'Weight Loss (g)' (pérdida de peso en gramos)
# Calculamos esta pérdida como la diferencia entre el peso inicial y el peso final del material
data['Weight Loss (g)'] = data['Weight of Bare Sepcimen (g)'] - data['Weight of Coated Specimen (g)']

# Definimos las variables para el análisis:
# - 'Days' (días) será la variable independiente (X), es decir, el tiempo que el material estuvo expuesto
# - 'Weight Loss (g)' será la variable dependiente (y), es decir, cuánto peso perdió el material
X = data[['Days']]  # Variable independiente (entrada del modelo)
y = data['Weight Loss (g)']  # Variable dependiente (salida del modelo)

# Dividimos los datos en dos partes:
# - Datos de entrenamiento (80%): para que el modelo aprenda
# - Datos de prueba (20%): para evaluar qué tan bien funciona el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos un modelo de regresión lineal
# Este modelo intenta encontrar una línea recta que relacione el tiempo (días) con la pérdida de peso
model = LinearRegression()

# Entrenamos el modelo usando los datos de entrenamiento
# Aquí, el modelo aprende la relación entre el tiempo y la pérdida de peso
model.fit(X_train, y_train)

# Usamos el modelo entrenado para hacer predicciones sobre los datos de prueba
# Esto nos dice qué tan bien el modelo puede predecir la pérdida de peso en datos que no ha visto antes
y_pred = model.predict(X_test)

# Calculamos métricas para evaluar el rendimiento del modelo:
# - MSE (Error Cuadrático Medio): mide el promedio de los errores al cuadrado (valores predichos vs. reales)
# - MAE (Error Absoluto Medio): mide el promedio de las diferencias absolutas entre valores predichos y reales
# - R² (Coeficiente de Determinación): mide qué tan bien el modelo explica la variabilidad de los datos
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostramos las métricas en la consola para entender el rendimiento del modelo
print("\nMétricas de evaluación del modelo:")
print(f"Error cuadrático medio (MSE): {mse:.2f}")
print(f"Error absoluto medio (MAE): {mae:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")

# Creamos un gráfico para comparar los valores reales (observados) con los valores predichos por el modelo
plt.figure(figsize=(10, 6))  # Configuramos el tamaño del gráfico
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)  # Puntos que representan las predicciones
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # Línea ideal (predicción perfecta)
plt.xlabel('Valores reales (Pérdida de peso)')  # Etiqueta del eje X
plt.ylabel('Valores predichos (Pérdida de peso)')  # Etiqueta del eje Y
plt.title('Comparación de valores reales vs. predichos')  # Título del gráfico
plt.grid(True)  # Agregamos una cuadrícula para facilitar la lectura
plt.show()  # Mostramos el gráfico

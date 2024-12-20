# ============================
# Análisis de Idealista Madrid
# ============================

# Importamos las bibliotecas necesarias

# Bibliotecas estándar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funciones y modelos de Scikit-learn
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y prueba
from sklearn.linear_model import LinearRegression  # Para el modelo de regresión lineal
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas de evaluación

# Estilo de gráficos
plt.style.use('default')
sns.set_theme(style="whitegrid")

# ---------------------------
# CARGA Y EXPLORACIÓN INICIAL
# ---------------------------
print("Cargando y explorando el dataset...")
df = pd.read_csv("idealista_madrid.csv")

# Mostramos columnas y resumen del dataset
print("Columnas disponibles:")
print(df.columns)
print("\nInformación general del dataset:")
print(df.info())

# Resumen manual de las columnas más relevantes
print("""
Columnas: 
- 'url', 'listingUrl', 'title', 'id', 'price', 'baths', 'rooms', 'sqft',
- 'description' (1 valor nulo), 'address', 'typology', 'advertiserProfessionalName',
- 'advertiserName'.
""")

# ---------------------------
# LIMPIEZA DE DATOS
# ---------------------------
print("\nIniciando limpieza de datos...")

# Convertimos columnas clave a tipos numéricos
cols_numeric = ['price', 'baths', 'rooms', 'sqft']
for col in cols_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Identificamos valores nulos
print("\nValores nulos por columna:")
print(df.isnull().sum())

# ---------------------------
# DETECCIÓN DE OUTLIERS
# ---------------------------
print("\nDetectando valores atípicos...")

# Función para calcular outliers
def calcular_outliers(columna):
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[columna] < lower_bound) | (df[columna] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Identificamos outliers en variables clave y almacenamos límites
outliers_price, lower_bound_price, upper_bound_price = calcular_outliers('price')
outliers_sqft, lower_bound_sqft, upper_bound_sqft = calcular_outliers('sqft')
outliers_rooms, lower_bound_rooms, upper_bound_rooms = calcular_outliers('rooms')
outliers_baths, lower_bound_baths, upper_bound_baths = calcular_outliers('baths')

# Imprimimos la información sobre los outliers
for col, lower, upper in zip(['price', 'sqft', 'rooms', 'baths'], 
                              [lower_bound_price, lower_bound_sqft, lower_bound_rooms, lower_bound_baths], 
                              [upper_bound_price, upper_bound_sqft, upper_bound_rooms, upper_bound_baths]):
    print(f"\nOutliers en '{col}':")
    print(f"- Límite inferior: {lower:.2f}, Límite superior: {upper:.2f}")

# ---------------------------
# FILTRADO SIN OUTLIERS
# ---------------------------
print("\nFiltrando dataset para eliminar outliers...")

df_cleaned = df[
    (df['price'] >= lower_bound_price) & (df['price'] <= upper_bound_price) &
    (df['sqft'] >= lower_bound_sqft) & (df['sqft'] <= upper_bound_sqft) &
    (df['rooms'] >= lower_bound_rooms) & (df['rooms'] <= upper_bound_rooms) &
    (df['baths'] >= lower_bound_baths) & (df['baths'] <= upper_bound_baths)
]

# Guardamos el dataset limpio
df_cleaned.to_csv('idealista_madrid_clean.csv', index=False)
print("\nDataset limpio guardado como 'idealista_madrid_clean.csv'.")

# ---------------------------
# DISTRIBUCIONES DE VARIABLES
# ---------------------------
print("\nVisualizando distribuciones de variables clave...")
variables = ['price', 'sqft', 'rooms', 'baths']

for var in variables:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[var], kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribución de {var}', fontsize=14)
    plt.xlabel(var, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.tight_layout()
    plt.show()

# ---------------------------
# CORRELACIÓN ENTRE VARIABLES
# ---------------------------
print("\nCalculando y visualizando la matriz de correlación...")
correlation_matrix = df[['price', 'sqft', 'rooms', 'baths']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Mapa de calor de correlación', fontsize=14)
plt.tight_layout()
plt.show()

# ---------------------------
# RELACIONES CLAVE ENTRE VARIABLES
# ---------------------------

# Precio vs. Tamaño (sqft)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sqft', y='price', data=df, color='blue', alpha=0.6)
plt.title('Relación entre Precio y Tamaño', fontsize=14)
plt.xlabel('Tamaño (sqft)', fontsize=12)
plt.ylabel('Precio (€)', fontsize=12)
plt.tight_layout()
plt.show()

# Precio vs. Número de Habitaciones
plt.figure(figsize=(8, 6))
sns.boxplot(x='rooms', y='price', data=df, palette='Blues')
plt.title('Distribución de Precios por Número de Habitaciones', fontsize=14)
plt.xlabel('Número de Habitaciones', fontsize=12)
plt.ylabel('Precio (€)', fontsize=12)
plt.tight_layout()
plt.show()

# ---------------------------
# CONCLUSIONES
# ---------------------------
print("""
Conclusiones:
1. Relación Precio-Tamaño:
   - Hay una clara correlación positiva entre el tamaño (sqft) y el precio.
   - La relación es aproximadamente lineal, pero con cierta dispersión.
   - En propiedades más grandes, la dispersión en precios aumenta, lo que sugiere la influencia de otros factores.

2. Relación Precio-Habitaciones:
   - Las propiedades con más habitaciones tienen precios promedio más altos.
   - El mayor salto en precios ocurre al pasar de 2 a 3 habitaciones.
   - Las viviendas de 2 habitaciones muestran un equilibrio entre precio y accesibilidad.

3. Observaciones Generales:
   - Las viviendas pequeñas (1 habitación o estudios) tienen precios más predecibles.
   - Las propiedades con 3 o más habitaciones presentan mayor variabilidad en precios.
   - Para inversión, las propiedades de 2 habitaciones parecen ser las más rentables por su equilibrio entre precio y tamaño.
""")

# ==============================
# PASAMOS AL MODELADO PREDICTIVO CON SCIKIT-LEARN
# ==============================

# 1. DIVISIÓN DEL CONJUNTO DE DATOS EN ENTRENAMIENTO Y PRUEBA.

# Variables independientes (X) y dependientes (y)
X = df[['sqft', 'rooms', 'baths']]  # Características
y = df['price']  # Objetivo: predecir el precio  

# Dividimos el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random_state , es una semilla para que los datos puedan ser reproducibles

print("\nConjunto de datos dividido en entrenamiento y prueba.")

# Creamos el modelo de regresion lineal
model = LinearRegression() 

# Entrenar el modelo
model.fit(X_train, y_train)  # El modelo aprende de la relación entre X e Y

# Realizar predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test) 

# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)  # Error cuadrático medio
rmse = np.sqrt(mse)  # Raíz del error cuadrático
r2 = r2_score(y_test, y_pred)  # R2, que mide la calidad de la predicción

# Resultados de la evaluación
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualización de las predicciones vs. valores reales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # Línea de 1:1
plt.title('Predicción vs Realidad', fontsize=14)
plt.xlabel('Valor Real del Precio', fontsize=12)
plt.ylabel('Valor Predicho del Precio', fontsize=12)
plt.tight_layout()
plt.show()

# Visualización de los coeficientes del modelo
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coeficiente'])
print("\nCoeficientes del modelo:")
print(coefficients)

# ============================
# Resumen de Análisis y Oportunidades
# ============================

# Resumen de las conclusiones del análisis
print("""
Resumen de Análisis y Oportunidades:

1. Relación Precio-Tamaño:
   - Existe una fuerte correlación positiva entre el tamaño (sqft) y el precio. Las propiedades más grandes tienen precios más altos, aunque con mayor dispersión. Esto sugiere que el tamaño es un factor clave en la determinación del precio, pero también existen otros factores que afectan la variabilidad en el precio, como la ubicación o la tipología de la propiedad.

2. Relación Precio-Habitaciones:
   - Las propiedades con más habitaciones tienden a ser más caras. El salto más significativo en el precio ocurre entre 2 y 3 habitaciones, lo que indica que las propiedades de 3 habitaciones y más presentan un precio significativamente más alto. Las propiedades de 2 habitaciones ofrecen un equilibrio entre precio y espacio, lo que podría hacerlas atractivas para compradores con presupuesto medio.

3. Observaciones Generales:
   - Las propiedades pequeñas (1 habitación o estudios) tienen precios más predecibles, lo que sugiere que el mercado para estas propiedades es más estable y menos volátil.
   - Las propiedades con 3 o más habitaciones tienen una mayor variabilidad en los precios, lo que indica que factores adicionales, como la ubicación, el tipo de propiedad y otros atributos, juegan un papel importante en la determinación de su precio.
   - Para los inversores, las propiedades de 2 habitaciones parecen ser las más rentables por su relación calidad-precio equilibrada, mientras que las propiedades más grandes podrían ofrecer mayores rendimientos, pero con una mayor incertidumbre en el precio.

4. Predicción de Precios:
   - El modelo de regresión lineal muestra un rendimiento decente, con un R2 positivo, lo que indica que el modelo es capaz de predecir de manera razonable los precios en función de las características como el tamaño, el número de habitaciones y el número de baños. Sin embargo, el modelo podría beneficiarse de la inclusión de otras variables adicionales, como la ubicación exacta o la tipología de la propiedad, para mejorar la precisión de las predicciones.

5. Oportunidades de Mercado:
   - Las propiedades de 2 habitaciones parecen ofrecer la mejor relación calidad-precio, ya que combinan un precio relativamente asequible con un tamaño suficiente para la mayoría de los compradores.
   - El mercado de propiedades de lujo (aquellas que superan los 1.5 millones de euros) muestra una mayor volatilidad en los precios, lo que podría ofrecer oportunidades de negociación.
   - Las propiedades más pequeñas tienen precios más estables y predecibles, lo que las hace más atractivas para aquellos que buscan una inversión segura y menos susceptible a fluctuaciones en el mercado.

6. Recomendaciones para Inversores:
   - Los inversores que buscan propiedades con alta rentabilidad podrían considerar las de 2 habitaciones, ya que presentan un buen balance entre precio y tamaño, además de tener una demanda relativamente estable.
   - Para aquellos dispuestos a asumir más riesgos, las propiedades grandes (3 o más habitaciones) podrían ofrecer una mayor rentabilidad, pero con una mayor incertidumbre en el precio.
   - Las propiedades pequeñas son ideales para aquellos que buscan estabilidad y precios más previsibles, aunque con un rendimiento de inversión más moderado.

En resumen, el análisis muestra que el mercado inmobiliario en Madrid está determinado principalmente por el tamaño y el número de habitaciones, con propiedades de 2 habitaciones como una opción estratégica para compradores e inversores debido a su equilibrio entre precio y accesibilidad.
""")

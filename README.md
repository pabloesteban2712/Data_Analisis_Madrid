# Análisis de Datos Inmobiliarios en Madrid

Este proyecto realiza un análisis completo de las propiedades inmobiliarias en Madrid utilizando datos extraídos de la plataforma Idealista. El análisis se enfoca en entender cómo las características de las propiedades (tamaño, número de habitaciones, número de baños) se correlacionan con el precio. Además, se utiliza un modelo predictivo basado en regresión lineal para estimar el precio de las propiedades en función de estas características.

## Tabla de Contenidos

1. [Introducción](#introducción)
2. [Objetivos del Proyecto](#objetivos-del-proyecto)
3. [Metodología](#metodología)
4. [Resultados](#resultados)
5. [Conclusiones](#conclusiones)
6. [Recomendaciones](#recomendaciones)
7. [Oportunidades de Mercado](#oportunidades-de-mercado)

---

## Introducción

El mercado inmobiliario en Madrid ha experimentado un crecimiento significativo en los últimos años. Comprender los factores que influyen en los precios de las propiedades es crucial para compradores, vendedores e inversores. Este proyecto tiene como objetivo analizar un conjunto de datos de Idealista que contiene información sobre propiedades inmobiliarias en Madrid, con el fin de identificar patrones en los precios y prever futuros precios basados en características clave.

## Objetivos del Proyecto

1. Analizar la relación entre el precio de las propiedades y sus características (tamaño, número de habitaciones, número de baños).
2. Realizar una limpieza de datos para asegurar que el análisis no se vea afectado por valores atípicos o nulos.
3. Construir un modelo predictivo utilizando regresión lineal para estimar el precio de las propiedades en función de las características seleccionadas.
4. Identificar oportunidades de inversión en el mercado inmobiliario de Madrid.

## Metodología

### 1. Carga y Exploración de Datos

El conjunto de datos fue cargado y explorado para comprender sus características. Se identificaron las columnas principales, como `price` (precio), `sqft` (tamaño en metros cuadrados), `rooms` (número de habitaciones), y `baths` (número de baños).

### 2. Limpieza de Datos

Durante la limpieza de datos, se convirtieron las columnas a los tipos adecuados (numéricos), y se gestionaron los valores nulos. Se también eliminaron los valores atípicos mediante un análisis de cuartiles.

### 3. Análisis Exploratorio

Se realizaron visualizaciones para entender mejor la distribución de las variables clave, como la relación entre el precio y el tamaño, y la correlación entre las variables más relevantes.

### 4. Modelado Predictivo

Se construyó un modelo de regresión lineal utilizando las variables `sqft`, `rooms` y `baths` para predecir el precio. El modelo fue evaluado con métricas como el Error Cuadrático Medio (MSE), la Raíz del Error Cuadrático Medio (RMSE) y el R2 Score.

## Resultados

A través del análisis de los datos, se encontraron varias relaciones importantes:

1. **Relación entre Precio y Tamaño:** Existe una correlación positiva clara entre el precio y el tamaño de la propiedad (en metros cuadrados). Las propiedades más grandes tienen precios más altos, aunque con cierta dispersión, lo que indica que otros factores también influyen en los precios.

2. **Relación entre Precio y Número de Habitaciones:** Las propiedades con más habitaciones tienden a ser más caras, con un salto significativo en el precio entre 2 y 3 habitaciones. Esto sugiere que las propiedades de 3 habitaciones o más tienen un precio mucho mayor que las de 2 habitaciones.

3. **Estabilidad en Propiedades Pequeñas:** Las propiedades pequeñas (como estudios o de 1 habitación) tienden a tener precios más estables y predecibles. En cambio, las propiedades más grandes presentan mayor variabilidad en el precio.

## Conclusiones

1. **Precio y Tamaño (sqft):** Existe una correlación positiva significativa entre el tamaño de la propiedad y su precio. Las propiedades más grandes tienden a tener precios más altos. Sin embargo, el modelo muestra que no solo el tamaño afecta al precio, sino también otros factores como la ubicación o la tipología de la propiedad.

2. **Precio y Número de Habitaciones:** La cantidad de habitaciones tiene un impacto directo en el precio. Las propiedades de 2 habitaciones ofrecen un buen equilibrio entre precio y tamaño, lo que las convierte en una opción popular y asequible para muchos compradores. Las propiedades con 3 o más habitaciones tienen precios más altos y una mayor dispersión, lo que indica que otros factores como la ubicación y la tipología juegan un papel importante.

3. **Propiedades Pequeñas vs. Grandes:** Las propiedades de 1 habitación o estudios son más previsibles en términos de precio, lo que puede hacerlas atractivas para compradores que buscan estabilidad. Las propiedades más grandes (3 o más habitaciones) tienen una mayor variabilidad en los precios, lo que puede ofrecer oportunidades interesantes para los inversores que buscan propiedades con mayores rendimientos, aunque con mayor riesgo.

## Recomendaciones

1. **Propiedades de 2 Habitaciones:** Dado que estas propiedades ofrecen un buen equilibrio entre precio y espacio, se recomienda que tanto compradores como inversores consideren propiedades de 2 habitaciones, especialmente en zonas bien ubicadas.

2. **Inversión en Propiedades Grandes (3+ Habitaciones):** Las propiedades más grandes tienen un mayor rendimiento de inversión potencial debido a su mayor precio de venta. Sin embargo, la variabilidad en los precios puede aumentar el riesgo, por lo que los inversores deben tener en cuenta otros factores como la ubicación y el tipo de propiedad.

3. **Propiedades Pequeñas:** Las propiedades pequeñas son atractivas para aquellos que buscan estabilidad en sus inversiones y precios más predecibles. Estas propiedades pueden ser una buena opción para aquellos que buscan un mercado inmobiliario más estable y menos volátil.

## Oportunidades de Mercado

1. **Mercado de Propiedades de 2 Habitaciones:** Las propiedades de 2 habitaciones representan una excelente oportunidad para los inversores, ya que suelen tener una alta demanda y ofrecen un buen balance entre precio y tamaño. Además, son atractivas para una amplia gama de compradores, desde individuos hasta pequeñas familias.

2. **Propiedades de Lujo (Más de 1.5 Millones de Euros):** Las propiedades en el rango de precios más altos muestran una mayor volatilidad en el mercado. Aunque estas propiedades pueden ser más difíciles de vender, pueden ofrecer una alta rentabilidad en mercados de lujo o en zonas de alto poder adquisitivo.

3. **Estabilidad en el Mercado de Propiedades Pequeñas:** Las propiedades más pequeñas, como estudios y apartamentos de 1 habitación, tienen una mayor estabilidad en términos de precios y pueden ser una buena opción para los compradores que buscan una inversión segura y menos sujeta a fluctuaciones en el mercado.

---

## Cómo Ejecutar el Proyecto

1. Clona este repositorio:

git clone https://github.com/pabloesteban2712/Data_Analisis_Madrid

2. Instala las dependencias necesarias:

pip install -r requirements.txt

3. Ejecuta! 

python main.py


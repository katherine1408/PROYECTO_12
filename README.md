# 🛡️ Sprint 12 – Álgebra Lineal Aplicada a Machine Learning y Seguridad de Datos (Sure Tomorrow)

## 📌 Descripción del Proyecto

La compañía de seguros **Sure Tomorrow** desea implementar soluciones de machine learning para marketing, predicción de beneficios y protección de datos personales.

Este proyecto combina conceptos de **álgebra lineal, clasificación, regresión y ofuscación de datos**, aplicados a un caso realista. El dataset contiene características de clientes asegurados y su historial de beneficios.

## 🎯 Objetivos del Proyecto

1. 🧬 **Encontrar clientes similares** a uno específico para campañas de marketing personalizado.
2. 🤖 **Predecir** si un cliente nuevo recibirá un beneficio de seguro (clasificación binaria).
3. 📈 **Estimar** la cantidad de beneficios que recibirá un nuevo cliente (regresión lineal).
4. 🔐 **Ofuscar datos personales** sin afectar la calidad del modelo de regresión (protección de privacidad).

## 📁 Dataset utilizado

- `insurance_us.csv`

Columnas:

- `sex`: género del asegurado
- `age`: edad
- `salary`: salario anual estimado
- `children`: número de hijos
- `insurance_benefits`: número de beneficios de seguro recibidos (objetivo)

## 🧰 Funcionalidades del Proyecto

### 🔍 Preprocesamiento
- Revisión de datos nulos, atípicos y tipos de variables
- Estandarización de características numéricas

### 📌 Tarea 1 – Búsqueda de vecinos similares
- Aplicación de distancia euclidiana para encontrar clientes más cercanos
- Visualización y análisis del perfil del cliente objetivo vs. similares

### 📌 Tarea 2 – Clasificación binaria
- Implementación de modelos entrenados vs modelo dummy
- Comparación de desempeño con métricas (`F1`, `accuracy`)
- Explicación conceptual sobre cuándo el modelo puede ser peor que el dummy

### 📌 Tarea 3 – Regresión lineal
- Entrenamiento del modelo
- Evaluación con métricas: RMSE, R²

### 📌 Tarea 4 – Protección de datos (ofuscación)
- Enmascaramiento con matriz ortogonal aleatoria
- Evaluación del impacto del enmascaramiento en la calidad del modelo

## 🛠️ Herramientas utilizadas

- Python  
- pandas / numpy  
- scikit-learn (`LinearRegression`, `DummyClassifier`, `StandardScaler`)  
- matplotlib / seaborn  

---

📌 Proyecto desarrollado como parte del Sprint 12 del programa de Ciencia de Datos en **TripleTen**.

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- INICIO: Corrección de rutas para importación y archivos ---
# Obtener la ruta del directorio del script (notebooks)
NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
# Subir dos niveles para llegar a la raíz del proyecto
PROJECT_ROOT = os.path.abspath(os.path.join(NOTEBOOKS_DIR, os.pardir, os.pardir))
# Añadir la ruta raíz al path de Python para encontrar el módulo 'src'
sys.path.append(PROJECT_ROOT)

# Ahora la importación de funciones personalizadas funcionará
from src.utils.functions import cargar_y_preprocesar_datos, escalar_y_codificar
# --- FIN: Corrección de rutas ---


# --- Carga y preprocesamiento de datos ---
# Definir rutas de archivos de forma robusta
ruta_train_csv = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'customer_churn_dataset-training-master.csv')
ruta_test_csv = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'customer_churn_dataset-testing-master.csv')

# Cargar y preprocesar datos usando la función actualizada
df_concatenado, ids_clientes = cargar_y_preprocesar_datos(ruta_train_csv, ruta_test_csv)

print("--- Información General del DataFrame ---")
df_concatenado.info()
print("\n--- Valores Únicos por Columna ---")
print(df_concatenado.nunique())
print("\n--- Primeras 5 Filas ---")
print(df_concatenado.head(5))


# --- Análisis Exploratorio de Datos (EDA) ---
# Definir variables numéricas y categóricas
variables_numericas = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
variables_categoricas = ['Gender', 'Subscription Type', 'Contract Length']

# ANÁLISIS UNIVARIADO
print("\n--- Análisis Univariado ---")
# Estadísticas descriptivas para variables numéricas
estadisticas_numericas = df_concatenado.drop(columns="Churn").describe().T
print(estadisticas_numericas)

# Visualización de distribuciones (opcional, puede generar muchos gráficos)
# for variable in variables_numericas:
#     plt.figure(figsize=(8, 4))
#     sns.histplot(data=df_concatenado, x=variable)
#     plt.title(f'Distribución de {variable}')
#     plt.xlabel(variable)
#     plt.ylabel('Frecuencia')
#     plt.show()

# ANÁLISIS BIVARIADO
print("\n--- Análisis Bivariado (Relación con Churn) ---")
# Relación entre variables numéricas y el objetivo
# for variable in variables_numericas:
#     plt.figure(figsize=(8, 5))
#     sns.boxplot(data=df_concatenado, x='Churn', y=variable)
#     plt.title(f'Relación entre Churn y {variable}')
#     plt.xlabel('Churn')
#     plt.ylabel(variable)
#     plt.show()

# --- Manejo de Outliers ---
print("\n--- Manejando Outliers ---")
# Calcular IQR para 'Support Calls' y 'Total Spend' en la categoría Churn 0
Q1_SC = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Support Calls'].quantile(0.25)
Q3_SC = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Support Calls'].quantile(0.75)
IQR_SC = Q3_SC - Q1_SC

Q1_TS = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Total Spend'].quantile(0.25)
Q3_TS = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Total Spend'].quantile(0.75)
IQR_TS = Q3_TS - Q1_TS

# Calcular límites superior e inferior
limite_superior_SC = Q3_SC + 1.5 * IQR_SC
limite_inferior_TS = Q1_TS - 1.5 * IQR_TS

# Eliminar filas con outliers
df_concatenado = df_concatenado[~((df_concatenado['Churn'] == 0) & ((df_concatenado['Support Calls'] > limite_superior_SC) | (df_concatenado['Total Spend'] < limite_inferior_TS)))]
print("Outliers eliminados.")

# --- Matriz de Correlación ---
print("\n--- Generando Matriz de Correlación ---")
matriz_correlacion = df_concatenado[['Churn'] + variables_numericas].corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm')
# plt.title('Matriz de Correlación')
# plt.show()
print(matriz_correlacion)


# --- Codificar y Escalar Variables ---
print("\n--- Codificando y Escalando Variables ---")
df_limpio = escalar_y_codificar(df_concatenado)
print("Variables codificadas y escaladas.")
print(df_limpio.head())


# --- Guardar el DataFrame Limpio ---
ruta_csv_limpio = os.path.join(PROJECT_ROOT, 'src', 'data', 'processed', 'df_clean.csv')
df_limpio.to_csv(ruta_csv_limpio, index=False)
print(f"\nDataFrame limpio guardado exitosamente en: {ruta_csv_limpio}")
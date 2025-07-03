import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler

# --- INICIO: Corrección de rutas para importación y archivos ---
# Obtener la ruta absoluta del directorio del script (src)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Obtener la ruta raíz del proyecto (un nivel arriba de src)
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, os.pardir))
# Añadir la ruta raíz al path de Python para encontrar módulos como 'src'
sys.path.append(PROJECT_ROOT)

# Ahora la importación funcionará correctamente
from src.utils.functions import escalar_y_codificar, cargar_y_predecir_modelo, cargar_y_preprocesar_datos
# --- FIN: Corrección de rutas ---


# Menú de la barra lateral
st.sidebar.title('Menú')
seccion = st.sidebar.radio('Ir a la sección:', ('Introducción', 'Análisis EDA', 'Entrenamiento del Modelo', 'Predicción', 'Conclusiones y Recomendaciones'))

# Contenido de la sección
if seccion == 'Introducción':
    st.title('Introducción')
    
    # Información del proyecto
    st.write("- TelecomConnect es un proveedor de telecomunicaciones que enfrenta el desafío de retener a sus clientes en un mercado altamente competitivo. La retención de clientes es esencial para el éxito a largo plazo de la empresa.")
    st.write("- La retención de clientes se refiere a la capacidad de una empresa para evitar que sus clientes actuales abandonen o cancelen sus servicios. Esta métrica es crítica en un mercado donde adquirir nuevos clientes puede ser costoso, y retener a los existentes puede ser más rentable.")
    st.write("- En este proyecto, TelecomConnect nos ha encargado realizar un análisis profundo de sus datos y desarrollar un modelo de predicción de abandono (churn). Este modelo nos permitirá anticipar la fuga de clientes y tomar medidas proactivas para retenerlos.")
    st.write("- Nuestro objetivo es ayudar a TelecomConnect a comprender mejor a sus clientes, identificar patrones de comportamiento y predecir quiénes son más propensos a abandonar. Al hacerlo, la empresa puede implementar estrategias de retención más efectivas y mejorar la satisfacción del cliente.")
    st.write("- A lo largo de este proyecto, utilizaremos una base de datos proporcionada por TelecomConnect que contiene diversas variables relacionadas con los clientes y sus interacciones con los servicios de telecomunicaciones.")

elif seccion == 'Análisis EDA':
    st.title('Análisis Exploratorio de Datos (EDA)')

    # Construir rutas relativas usando la variable PROJECT_ROOT
    archivo_train_csv = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'customer_churn_dataset-training-master.csv')
    archivo_test_csv = os.path.join(PROJECT_ROOT, 'src', 'data', 'raw', 'customer_churn_dataset-testing-master.csv')
    
    try:
        df_concatenado, ids_clientes = cargar_y_preprocesar_datos(archivo_train_csv, archivo_test_csv)
        st.success("Datos cargados y preprocesados correctamente.")
        
        st.write("## Información del DataFrame de Clientes")
        st.write("El DataFrame contiene información detallada sobre los clientes y sus interacciones con la empresa.")
        
        # Variables numéricas y categóricas
        variables_numericas = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
        variables_categoricas = ['Gender', 'Subscription Type', 'Contract Length']

        st.subheader('Estadísticas de Variables Numéricas:')
        st.write(df_concatenado[variables_numericas].describe())

        st.subheader('Distribución de Variables Categóricas:')
        for variable in variables_categoricas:
            fig, ax = plt.subplots(figsize=(6, 4))
            conteos = df_concatenado[variable].value_counts()
            ax.pie(conteos, labels=conteos.index, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Distribución de {variable}')
            st.pyplot(fig)

        st.subheader('Relación entre Variables y Abandono (Churn)')
        for variable in variables_numericas:
            st.write(f"**Análisis de: {variable}**")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            sns.histplot(data=df_concatenado, x=variable, hue='Churn', kde=True, ax=ax[0])
            ax[0].set_title(f'Distribución por Abandono')
            sns.boxplot(data=df_concatenado, x='Churn', y=variable, ax=ax[1])
            ax[1].set_title(f'Boxplot por Abandono')
            st.pyplot(fig)
        
        st.subheader('Matriz de Correlación:')
        matriz_correlacion = df_concatenado[['Churn'] + variables_numericas].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    except FileNotFoundError as e:
        st.error(f"Error al cargar los datos: No se pudo encontrar el archivo. Verifica la ruta.")
        st.error(e)
    except Exception as e:
        st.error(f"Ocurrió un error inesperado durante el análisis EDA: {e}")


elif seccion == 'Entrenamiento del Modelo':
    st.title('Entrenamiento del Modelo')

    ruta_resultados = os.path.join(PROJECT_ROOT, 'src', 'data', 'processed', 'df_resultados.csv')
    try:
        df_resultados = pd.read_csv(ruta_resultados)
        st.subheader('Resultados Comparativos de Modelos')
        st.dataframe(df_resultados)
    except FileNotFoundError:
        st.error(f"No se pudo encontrar el archivo de resultados en la ruta: {ruta_resultados}")

    st.info('Nota: No se utilizó la división train-test ya que el conjunto de datos de prueba fue proporcionado por la empresa.')
    
    st.subheader('Selección del Modelo: Regresión Logística')
    st.write('Se eligió el modelo de **Regresión Logística** por las siguientes razones:')
    st.write('- **Rendimiento Confiable (ROC-AUC: 0.76)**: Ofrece un buen poder predictivo.')
    st.write('- **Equilibrio entre Sensibilidad y Especificidad**: Logra una exactitud del 72%, un balance efectivo para detectar tanto a los clientes que abandonarán como a los que no, lo cual es crucial en un problema de churn.')

elif seccion == 'Predicción':
    st.title('Predicción de Abandono de Clientes')

    archivo_cargado = st.file_uploader("Cargar archivo CSV con datos de clientes para predecir", type=["csv"])

    if archivo_cargado is not None:
        df_a_predecir = pd.read_csv(archivo_cargado)
        df_original = df_a_predecir.copy()
        st.write("Vista previa de los datos cargados:")
        st.dataframe(df_original.head())

        if st.button("Realizar Predicciones"):
            with st.spinner('Procesando datos y generando predicciones...'):
                try:
                    variables_caracteristicas = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Male', 'Contract Length_cod']
                    
                    df_procesado = escalar_y_codificar(df_a_predecir)
                    X_test = df_procesado[variables_caracteristicas]
                    
                    resultados_prediccion, resultados_probabilidad, resultados_importancia = cargar_y_predecir_modelo(X_test)

                    df_predicciones = pd.DataFrame({
                        'Prediccion_Abandono': resultados_prediccion, 
                        'Probabilidad_No_Abandono': resultados_probabilidad[:, 0], 
                        'Probabilidad_Abandono': resultados_probabilidad[:, 1]
                    })

                    df_resultado_final = pd.concat([df_original, df_predicciones], axis=1)

                    st.success('¡Predicciones generadas con éxito!')
                    st.write("Resultados de la Predicción:")
                    st.dataframe(df_resultado_final)

                    st.subheader('Distribución de Probabilidades de Abandono')
                    fig, ax = plt.subplots()
                    ax.hist(resultados_probabilidad[:, 1], bins=20, color='blue', alpha=0.7)
                    ax.set_xlabel('Probabilidad de Abandono (Churn)')
                    ax.set_ylabel('Frecuencia')
                    st.pyplot(fig)

                    st.subheader('Importancia de las Características del Modelo')
                    importancia_caracteristicas = pd.DataFrame({'Característica': variables_caracteristicas, 'Importancia': resultados_importancia})
                    importancia_caracteristicas = importancia_caracteristicas.sort_values(by='Importancia', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importancia', y='Característica', data=importancia_caracteristicas, ax=ax)
                    ax.set_title('Importancia de las Características para la Predicción')
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Ocurrió un error durante la predicción: {e}")

elif seccion == 'Conclusiones y Recomendaciones':
    st.title('Conclusiones y Recomendaciones')

    st.write('**Conclusiones Clave:**')
    st.write('1. El modelo de Regresión Logística alcanza una **exactitud del 72%**, lo que proporciona una base sólida para identificar a los clientes en riesgo.')
    st.write('2. Las **llamadas de soporte**, la **duración del contrato** y el **retraso en los pagos** son los factores más influyentes en la decisión de un cliente de abandonar la empresa.')
    
    st.write('**Recomendaciones Accionables:**')
    st.write("- **Mejorar la Calidad del Soporte:** Implementar un sistema para monitorear y mejorar la calidad del servicio de soporte, ya que un alto número de llamadas es un fuerte predictor de abandono.")
    st.write("- **Fomentar Contratos a Largo Plazo:** Diseñar ofertas y descuentos atractivos para incentivar a los clientes con contratos mensuales a migrar a planes anuales o trimestrales, que tienen una tasa de retención mucho mayor.")
    st.write("- **Campañas de Retención Proactivas:** Utilizar las predicciones del modelo para identificar a los clientes con alta probabilidad de abandono y dirigirse a ellos con campañas de retención personalizadas antes de que cancelen el servicio.")
    st.write("- **Mejora Continua del Modelo:** Para aumentar la precisión del modelo, se recomienda recopilar más datos y variables, como la satisfacción del cliente (NPS), el uso de servicios específicos o datos de la competencia.")


import joblib
import streamlit as st
import pandas as pd
import os
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from src.utils.functions import escalar_y_codificar, cargar_y_predecir_modelo

import streamlit as st

# Título de la aplicación
st.title('Presentación en Streamlit')

# Menú lateral
st.sidebar.title('Menú')
seccion = st.sidebar.radio('Ir a sección:', ('Introducción', 'Análisis EDA', 'Entrenamiento Modelo', 'Predicción'))

# Contenido de la sección
if seccion == 'Introducción':
    st.write('Bienvenido a la sección de Introducción.')
    st.write('Bienvenidos a la plataforma de análisis y predicción de Churn de la empresa TelecomConnect. '
         'TelecomConnect es un proveedor líder de servicios de telecomunicaciones que se enfrenta al desafío de retener a sus clientes en un mercado competitivo. '
         'La retención de clientes es esencial para el éxito a largo plazo de TelecomConnect. Para abordar este desafío que nos solicitaron, utilizaremos datos para analizar patrones y '
         'desarrollar un modelo de predicción de Churn, lo que nos permitirá anticiparnos a las bajas de clientes y tomar medidas proactivas. '
         'Los datos y el análisis desempeñan un papel fundamental en la retención de clientes y en la mejora de la satisfacción del cliente. ')

elif seccion == 'Análisis EDA':
    st.write('Bienvenido a la sección de Análisis EDA.')
    # Agrega aquí el contenido de la sección de Análisis EDA

elif seccion == 'Entrenamiento Modelo':
    st.write('Bienvenido a la sección de Entrenamiento del Modelo.')
    # Agrega aquí el contenido de la sección de Entrenamiento del Modelo

elif seccion == 'Predicción':
    st.write('Bienvenido a la sección de Predicción.')
    # Agrega aquí el contenido de la sección de Predicción

    # Subir archivo CSV
    subir_archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])

    variables_features = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Male', 'Contract Length_cod']
    variable_target = "Churn"

    if subir_archivo is not None:
        # Leer el archivo CSV
        df = pd.read_csv(subir_archivo)
        df_1 = df.copy()

        # cargar el modelo y realizar predicciones:
        if st.button("Cargar Modelo y Realizar Predicciones"):
            # Ejecutar la función para escalar y codificar los datos
            df = escalar_y_codificar(df)

            # Obtener las variables de entrada para el modelo
            X_test = df[variables_features]

            # Realizar predicciones y obtener probabilidades
            resultados_predicción, probabilidades_predicción, importancia_predicciones = cargar_y_predecir_modelo(X_test)

            # Crear un nuevo DataFrame con las predicciones y las probabilidades
            df_predicciones = pd.DataFrame({'Predicciones': resultados_predicción, 'Probabilidad 0': probabilidades_predicción[:, 0], 'Probabilidad 1': probabilidades_predicción[:, 1]})

            # Concatenar el DataFrame original con el DataFrame de predicciones
            df_resultado = pd.concat([df_1, df_predicciones], axis=1)

            # Mostrar los resultados
            st.write("Resultados de Predicción:")
            st.write(df_resultado)

            # Crear un gráfico para mostrar las probabilidades
            import matplotlib.pyplot as plt

            # Histograma de las probabilidades
            st.subheader('Histograma de Probabilidades')
            plt.hist(probabilidades_predicción[:, 1], bins=20, color='blue', alpha=0.7)
            plt.xlabel('Probabilidad de Churn')
            plt.ylabel('Frecuencia')
            st.pyplot(plt)

            # Crear un DataFrame con las características y sus importancias
            feature_importancias = pd.DataFrame({'Feature': variables_features, 'Importancia': importancia_predicciones})
            feature_importancias = feature_importancias.sort_values(by='Importancia', ascending=False)

            # Graficar las importancias
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importancia', y='Feature', data=feature_importancias)
            plt.title('Importancia de Características')
            plt.xlabel('Importancia')
            plt.ylabel('Característica')

            # Mostrar la gráfica de importancias
            st.subheader('Importancia de Características')
            st.pyplot(plt)

            # Agregar espacio entre elementos
            st.markdown('<br>', unsafe_allow_html=True)

            # Estilizar botones
            st.write(
                '<style>div.Widget.row-widget.stButton > div{background-color: #3498db; color: white; text-align: center}</style>',
                unsafe_allow_html=True
            )

            # Agregar espacio entre elementos
            st.markdown('<br>', unsafe_allow_html=True)





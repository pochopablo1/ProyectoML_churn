import joblib
import streamlit as st
import pandas as pd
import os
import sys
import seaborn as sns

sys.path.append("C:/Users/Hp/Desktop/proyecto")

from src.utils.functions import escalar_y_codificar, cargar_y_predecir_modelo

# Título de la aplicación
st.title('Predicción de Churn')
st.write('Esta aplicación realiza predicciones de Churn usando un modelo de regresión logística.')

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
        resultados_prediccion, probabilidades_prediccion, importancia_predicciones = cargar_y_predecir_modelo(X_test)

        # Crear un nuevo DataFrame con las predicciones y las probabilidades
        df_predicciones = pd.DataFrame({'Predicciones': resultados_prediccion, 'Probabilidad 0': probabilidades_prediccion[:, 0], 'Probabilidad 1': probabilidades_prediccion[:, 1]})

        # Concatenar el DataFrame original con el DataFrame de predicciones
        df_resultado = pd.concat([df_1, df_predicciones], axis=1)

        # Mostrar los resultados
        st.write("Resultados de Predicción:")
        st.write(df_resultado)

        # Crear un gráfico para mostrar las probabilidades
        import matplotlib.pyplot as plt

        # Histograma de las probabilidades
        st.subheader('Histograma de Probabilidades')
        plt.hist(probabilidades_prediccion[:, 1], bins=20, color='blue', alpha=0.7)
        plt.xlabel('Probabilidad de Churn')
        plt.ylabel('Frecuencia')
        st.pyplot(plt)

        

        # Crear un DataFrame con las características y sus importancias
        feature_importances = pd.DataFrame({'Feature': variables_features, 'Importance': importancia_predicciones})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

        # Graficar las importancias
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances)
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

        # Agregar un pie de página
        st.markdown('Hecho por Pablo Santilli')
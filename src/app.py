
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import pandas as pd
import os
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


st.set_option('deprecation.showPyplotGlobalUse', False)

sys.path.append("C:/Users/Hp/Desktop/ProyectoML_churn")

from utils.functions import escalar_y_codificar, cargar_y_predecir_modelo, cargar_y_preprocesar_datos, preparar_datos




# menu lateral
st.sidebar.title('Menú')
seccion = st.sidebar.radio('Ir a sección:', ('Introducción', 'Análisis EDA', 'Entrenamiento Modelo', 'Predicción','Conclusiones y Recomendaciones'))

# Contenido de la sección
if seccion == 'Introducción':

    st.title('Introduccion')
    
    # Información sobre el proyecto

    st.write("-TelecomConnect es un proveedor de telecomunicaciones que se enfrenta al desafío de retener a sus clientes en un mercado altamente competitivo. La retención de clientes es esencial para el éxito a largo plazo de la empresa.")

    st.write("-La retención de clientes se refiere a la capacidad de una empresa para mantener a sus clientes actuales, evitando que se den de baja o 'den churn'. Esta métrica es crítica en un mercado donde adquirir nuevos clientes puede ser costoso y mantener a los clientes existentes puede ser más rentable.")

    st.write("-En este proyecto, hemos sido encargados por TelecomConnect para llevar a cabo un análisis en profundidad de sus datos y desarrollar un modelo de predicción de Churn. Este modelo nos permitirá anticiparnos a las bajas de clientes y tomar medidas proactivas para retenerlos.")

    st.write("-Nuestro objetivo es ayudar a TelecomConnect a entender mejor a sus clientes, identificar patrones de comportamiento y predecir quiénes son más propensos a darse de baja. Al hacerlo, la empresa podrá implementar estrategias de retención más efectivas y mejorar la satisfacción del cliente.")

    st.write("-A lo largo de este proyecto, utilizaremos una base de datos proporcionada por TelecomConnect que contiene una variedad de variables relacionadas con los clientes y su interacción con los servicios de telecomunicaciones.")



elif seccion == 'Análisis EDA':
    st.write('Bienvenido a la sección de Análisis EDA.')

    
    sys.path.append("C:/Users/Hp/Desktop/ProyectoML_churn")
    train_csv = 'src/data/raw/customer_churn_dataset-training-master.csv'
    test_csv = 'src/data/raw/customer_churn_dataset-testing-master.csv'
    df_concatenado, customer_ids = cargar_y_preprocesar_datos(train_csv, test_csv)

    # Título de la sección
    st.title('Análisis Exploratorio de Datos (EDA)')


    # Título
    st.write("## Información del DataFrame de Clientes")

    # Descripción de las variables
    st.write("El DataFrame contiene información detallada sobre clientes y sus interacciones con una empresa. A continuación, se describen las variables presentes en este conjunto de datos:")

    st.write("- **CustomerID**: Identificador único para cada cliente. Tipo de dato: Número de punto flotante.")
    st.write("- **Age**: Edad de los clientes. Tipo de dato: Número de punto flotante.")
    st.write("- **Gender**: Género de los clientes. Tipo de dato: Cadena de texto (objeto).")
    st.write("- **Tenure**: Antigüedad de la relación del cliente con la empresa. Tipo de dato: Número de punto flotante.")
    st.write("- **Usage Frequency**: Frecuencia de uso de los servicios. Tipo de dato: Número de punto flotante.")
    st.write("- **Support Calls**: Cantidad de llamadas de soporte. Tipo de dato: Número de punto flotante.")
    st.write("- **Payment Delay**: Retraso en los pagos. Tipo de dato: Número de punto flotante.")
    st.write("- **Subscription Type**: Tipo de suscripción del cliente. Tipo de dato: Cadena de texto (objeto).")
    st.write("- **Contract Length**: Duración del contrato. Tipo de dato: Cadena de texto (objeto).")
    st.write("- **Total Spend**: Gasto total del cliente. Tipo de dato: Número de punto flotante.")
    st.write("- **Last Interaction**: Fecha de la última interacción del cliente. Tipo de dato: Número de punto flotante.")
    st.write("- **Churn**: Indicador de cancelación del cliente. Tipo de dato: Número de punto flotante.")
    st.write("- **Dataset**: Etiqueta del conjunto de datos. Tipo de dato: Cadena de texto (objeto).")

    st.write("Estas variables proporcionan información valiosa sobre la base de clientes de la empresa, incluyendo detalles demográficos, comportamiento de uso y métricas relacionadas con la retención de clientes. Puedes utilizar esta información para realizar análisis y visualizaciones en tu aplicación de Streamlit.")



    # Variables numéricas y categóricas
    variables_numericas = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    variables_categoricas = ['Gender', 'Subscription Type', 'Contract Length']

    st.subheader('Estadísticas de Variables Numéricas:')
    st.write(df_concatenado[variables_numericas].describe())

    # Visualización de Distribuciones para Variables Numéricas
    st.subheader('Visualización de Distribuciones para Variables Numéricas:')
    for variable in variables_numericas:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df_concatenado, x=variable)
        plt.title(f'Distribución de {variable}')
        plt.xlabel(variable)
        plt.ylabel('Frecuencia')
        st.pyplot(plt) 
        st.write(f'Distribución de {variable}')

    # Visualización de Diagramas de Caja para Variables Numéricas
    st.subheader('Visualización de Diagramas de Caja para Variables Numéricas:')
    for variable in variables_numericas:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_concatenado, y=variable)
        plt.title(f'Gráfico de Bigote de {variable}')
        plt.ylabel(variable)
        st.pyplot()  
        st.write(f'Gráfico de Bigote de {variable}')

    # Visualización de la Distribución de Variables Categóricas
    st.subheader('Visualización de la Distribución de Variables Categóricas:')
    for variable in variables_categoricas:
        plt.figure(figsize=(6, 4))
        counts = df_concatenado[variable].value_counts()
        labels = counts.index
        plt.pie(counts, labels=labels,autopct='%1.1f%%')
        plt.title(f'Distribución de {variable}')
        st.pyplot()
        st.write(f'Distribución de {variable}')

    st.write('A continuación se presentan algunas estadísticas clave de las variables:')
    st.write('- Age: La edad promedio es de aproximadamente 39 años, con una distribución que va desde 18 hasta 65 años.')
    st.write('- Tenure: La tenencia promedio es de aproximadamente 31 meses, con valores que van desde 1 hasta 60 meses.')
    st.write('- Usage Frequency: La frecuencia de uso promedio es de aproximadamente 15,7, con valores que van desde 1 hasta 30.')
    st.write('- Support Calls: El número promedio de llamadas de soporte es de aproximadamente 3,8, con valores que van desde 0 hasta 10.')
    st.write('- Payment Delay: El retraso promedio en el pago es de aproximadamente 13,5, con valores que van desde 0 hasta 30.')
    st.write('- Total Spend: El gasto total promedio es de aproximadamente 620, con valores que van desde 100 hasta 1,000.')
    st.write('- Last Interaction: El tiempo promedio desde la última interacción es de aproximadamente 14,6 unidades de tiempo, con valores que van desde 1 hasta 30.')

    # Relación entre variables numéricas y target
    for variable in variables_numericas:
        st.subheader(f'Relación entre {variable} y Churn:')

        # Histograma
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df_concatenado, x=variable, hue='Churn', kde=True)
        plt.title(f'Distribución de {variable} por Churn')
        plt.xlabel(variable)
        plt.ylabel('Frecuencia')
        st.pyplot()
        plt.close()

        # Diagrama de Caja
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_concatenado, x='Churn', y=variable)
        plt.title(f'Relación entre Churn y {variable}')
        plt.xlabel('Churn')
        plt.ylabel(variable)
        st.pyplot()
        plt.close()
        
    st.write('La variable Age tiene incidencia en el target. El promedio de edad de los que dejan la empresa es mayor a los que no.')

    st.write('Las variables Tenure y Usage Frequency parecen no tener influencia en la variable target.')

    st.write('Support Calls tiene mucha diferencia entre los que se quedan en la empresa (bajo promedio de llamadas) y los que se fueron (alto promedio de llamadas). Encontramos valores atípicos en los clientes que se quedan en la empresa (luego vamos a analizar).')

    st.write('Los clientes que se van de la empresa tienen un promedio más alto de días en demoras de pagos.')

    st.write('La variable Total Spend también parece tener influencia en nuestros target. Los clientes que más gastan son los que deciden quedarse en la empresa. Encontramos valores atípicos en los clientes que se quedan en la empresa (luego vamos a analizar).')

    st.write('Por último, observamos que los clientes que se van de la empresa en promedio, hace más tiempo que no realizan transacciones.')


    # Relación entre variables categóricas y target (churn)
    for variable in variables_categoricas:
        st.subheader(f'Relación entre {variable} y Churn:')
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_concatenado, x=variable, hue='Churn')
        plt.title(f'Relación entre Churn y {variable}')
        plt.xlabel(variable)
        plt.ylabel('Frecuencia')
        plt.legend(title='Churn', loc='upper right', labels=['No Churn', 'Churn'])
        st.pyplot()
        plt.close()

    st.write('Las mujeres parecen ser más propensas a irse de la empresa (variable Gender).')

    st.write('No se observan diferencias entre los diferentes tipos de suscripciones.')

    st.write('Los contratos anuales y cuatrimestrales tienen un comportamiento similar con el target, pero los clientes con contratos mensuales se van de la empresa.')


    # Correlaciones entre variables numéricas
    correlacion_matrix = df_concatenado[['Churn', 'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Total Spend', 'Last Interaction', 'Payment Delay']].corr()

    st.subheader('Matriz de Correlación:')
    st.write('La matriz de correlación muestra las relaciones entre las variables numéricas y el target (Churn).')

    # Mostrar la matriz de correlación
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlacion_matrix, annot=True, cmap='coolwarm')
    st.pyplot()
    plt.close()

    # Comentarios sobre la matriz de correlación
    st.write('Comentarios sobre la matriz de correlación:')
    st.write('Churn está altamente correlacionado con Support Calls, Total Spend y Payment Delay.')
    st.write('Age y Tenure tienen correlaciones bajas con Churn, lo que sugiere una influencia limitada en la retención.')
    st.write('Support Calls y Payment Delay también tienen una correlación significativa con Total Spend.')



        # Proceso de Escalado y Codificación de Datos
    st.subheader('Proceso de Escalado y Codificación de Datos:')
    st.write('En este bloque de código, se realizaron una serie de transformaciones en los datos para prepararlos para el análisis y modelado. A continuación, se resumen las principales acciones realizadas:')


    st.write('1. Codificación de Variables Categóricas: Se aplicó la codificación one-hot (get_dummies)'
              'a las variables categóricas "Gender" y "Subscription Type". Esto se hizo para convertir' 
              'las variables categóricas en variables numéricas binarias (0 o 1), lo que es esencial para'
              'que los algoritmos de machine learning las utilicen.')

    st.write('2. Codificación de la Variable "Contract Length": Se creó una nueva variable llamada'
              '"Contract Length_cod", que se establece en 1 si el valor original en "Contract Length" es'
              '"Annual" o "Quarterly", y 0 en caso de "Monthly". Esto permite representar la duración del'
              'contrato de manera binaria.'
              'Esta acción se llevó a cabo porque la variable sin modificaciones perjudicaba al modelo y al'
              'observar que la no habia casi diferencias entre los contratos anuales y cuatrimestrales se los'
              'agrupó en 1 y mensual en otro grupo')

    st.write('3. Estandarización de Variables Numéricas: Se seleccionaron las columnas numéricas "Age", '
             '"Support Calls", "Payment Delay", "Total Spend" y "Last Interaction". Luego, se utilizó el '
             'escalador MinMaxScaler para llevar a cabo la estandarización de estas variables, lo que'
             'asegura que todas tengan un rango de valores similar (entre 0 y 1) para un mejor rendimiento'
             'en los modelos de machine learning.')







elif seccion == 'Entrenamiento Modelo':
    st.write('Bienvenido a la sección de Entrenamiento del Modelo.')

    # Cargar los resultados del entrenamiento del modelo
    ruta = 'C:/Users/Hp/Desktop/ProyectoML_churn/src/data/processed/df_resultados.csv'
    resultados_df = pd.read_csv(ruta)

    # Información sobre el entrenamiento del modelo
    st.write('No utilizamos la división de tren y prueba ya que el conjunto de prueba fue proporcionado por la empresa.')

    
    # Modelos utilizados
    st.subheader('Modelos utilizados')
    st.write('Para entrenar el modelo, utilizamos los siguientes clasificadores:')
    st.write('- Regresión Logística')
    st.write('- Árbol de Decisión')
    st.write('- Random Forest')
    st.write('- Gradient Boosting')
    st.write('- K-Nearest Neighbor')
    st.write('- Gaussian Naive Bayes')

    # Mostrar los resultados del entrenamiento
    st.subheader('Resultados del entrenamiento del modelo')
    st.dataframe(resultados_df)

    st.subheader('Resultados de los Hiperparametros')
    st.title("Resultados del Modelo")

    #diccionarios
    resultados = {
        "Random Forest": {
            "Precisión": 0.6311,
            "Informe de clasificación": """
                precision    recall  f1-score   support
            0.0       0.98      0.10      0.18     21097
            1.0       0.62      1.00      0.76     30493
        accuracy                           0.63     51590
    macro avg       0.80      0.55      0.47     51590
    weighted avg       0.76      0.63      0.52     51590
            """
        },
        "Decision Tree": {
            "Precisión": 0.6381,
            "Informe de clasificación": """
                precision    recall  f1-score   support
            0.0       0.98      0.12      0.21     21097
            1.0       0.62      1.00      0.77     30493
        accuracy                           0.64     51590
    macro avg       0.80      0.56      0.49     51590
    weighted avg       0.77      0.64      0.54     51590
            """
        },
        "Regresión Logística": {
            "Precisión": 0.7198,
            "Informe de clasificación": """
                precision    recall  f1-score   support
            0.0       0.94      0.33      0.49     21097
            1.0       0.68      0.99      0.81     30493
        accuracy                           0.72     51590
    macro avg       0.81      0.66      0.65     51590
    weighted avg       0.79      0.72      0.68     51590
            """
        },
        "K-Nearest Neighbors": {
            "Precisión": 0.6423,
            "Informe de clasificación": """
                precision    recall  f1-score   support
            0.0       0.96      0.13      0.23     21097
            1.0       0.62      1.00      0.77     30493
        accuracy                           0.64     51590
    macro avg       0.79      0.56      0.50     51590
    weighted avg       0.76      0.64      0.55     51590
            """
        },
        "Gaussian Naive Bayes": {
            "Precisión": 0.6581,
            "Informe de clasificación": """
                precision    recall  f1-score   support
            0.0       0.98      0.17      0.29     21097
            1.0       0.63      1.00      0.78     30493
        accuracy                           0.66     51590
    macro avg       0.81      0.58      0.53     51590
    weighted avg       0.78      0.66      0.58     51590
            """
        }
    }

    for modelo, datos in resultados.items():
        st.subheader(modelo)
        st.markdown(f"Precisión: {datos['Precisión']:.4f}")
        st.write(f"Informe de clasificación:\n{datos['Informe de clasificación']}")
elif seccion == 'Predicción':
    st.write('Bienvenido a la sección de Predicción.')

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

            st.write(
                '<style>div.Widget.row-widget.stButton > div{background-color: #3498db; color: white; text-align: center}</style>',
                unsafe_allow_html=True
            )

            # Agregar espacio entre elementos
            st.markdown('<br>', unsafe_allow_html=True)

elif seccion == 'Conclusiones y Recomendaciones':
    st.title('Conclusiones y Recomendaciones')

    # conclusiones y recomendaciones

    st.write('El modelo creado tiene una efectividad del 70%, por lo que se solicita a la empresa más información'

        '(variables y datos adicionales) para intentar mejorar el modelo y sus predicciones.')

    st.write('Las variables que más afectan a la pérdida de clientes son el soporte de llamadas,' 

        'lo que sugiere prestar atención a la calidad del servicio proporcionado por la empresa' 

        'para evitar la fuga de clientes. Aquellos clientes que realizan más llamadas probablemente' 

        'tengan problemas no resueltos.')

    st.write('Además, se observó que los contratos mensuales tienen un alto promedio de deserción de la empresa.'

        'Se recomienda explorar estrategias para retener a estos clientes, como la oferta de contratos a más'

        'largo plazo.')

    st.write('Se sugiere la generación de una campaña de retención utilizando el análisis predictivo para' 

        'evaluar su eficacia. Mientras tanto, se espera obtener nuevos datos para mejorar el modelo y,' 

        'por ende, las predicciones.')
    

    



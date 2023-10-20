
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import pandas as pd
import os
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write(os.getcwd())

from utils.functions import escalar_y_codificar, cargar_y_predecir_modelo

# Título de la aplicación
st.title('Presentación caso de negocio')

# menu lateral
st.sidebar.title('Menú')
seccion = st.sidebar.radio('Ir a sección:', ('Introducción', 'Análisis EDA', 'Entrenamiento Modelo', 'Predicción'))

# Contenido de la sección
if seccion == 'Introducción':
    st.write('Bienvenidos a la plataforma de análisis y predicción de Churn de la empresa TelecomConnect. '
         'TelecomConnect es un proveedor líder de servicios de telecomunicaciones que se enfrenta al desafío de retener a sus clientes en un mercado competitivo. '
         'La retención de clientes es esencial para el éxito a largo plazo de TelecomConnect. Para abordar este desafío que nos solicitaron, utilizaremos datos para analizar patrones y '
         'desarrollar un modelo de predicción de Churn, lo que nos permitirá anticiparnos a las bajas de clientes y tomar medidas proactivas. '
         'Los datos y el análisis desempeñan un papel fundamental en la retención de clientes y en la mejora de la satisfacción del cliente. ')

elif seccion == 'Análisis EDA':
    st.write('Bienvenido a la sección de Análisis EDA.')

    
    # Agrega la ruta al directorio y carga los datos
    sys.path.append("C:/Users/Hp/Desktop/ProyectoML_churn")
    train_csv = 'src/data/raw/customer_churn_dataset-training-master.csv'
    test_csv = 'src/data/raw/customer_churn_dataset-testing-master.csv'
    df_concatenado, customer_ids = cargar_y_preprocesar_datos(train_csv, test_csv)

    # Título de la sección
    st.title('Análisis Exploratorio de Datos (EDA)')

    # Información general del DataFrame
    st.subheader('Información general del DataFrame:')
    st.write(df_concatenado.info())

    # Número de valores únicos en cada columna
    st.subheader('Número de valores únicos en cada columna:')
    st.write(df_concatenado.nunique())

    # Muestra las primeras 5 filas del DataFrame
    st.subheader('Primeras 5 filas del DataFrame:')
    st.write(df_concatenado.head(5))

    # Variables numéricas y categóricas
    variables_numericas = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    variables_categoricas = ['Gender', 'Subscription Type', 'Contract Length']

    # Análisis Univariable de Variables Numéricas
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
        st.pyplot(plt)  # Muestra el gráfico con plt en lugar de sns.histplot
        st.write(f'Distribución de {variable}')

    # Visualización de Diagramas de Caja para Variables Numéricas
    st.subheader('Visualización de Diagramas de Caja para Variables Numéricas:')
    for variable in variables_numericas:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_concatenado, y=variable)
        plt.title(f'Gráfico de Bigote de {variable}')
        plt.ylabel(variable)
        st.pyplot()  # Usar st.pyplot() para mostrar el gráfico
        st.write(f'Gráfico de Bigote de {variable}')

    # Visualización de la Distribución de Variables Categóricas
    st.subheader('Visualización de la Distribución de Variables Categóricas:')
    for variable in variables_categoricas:
        plt.figure(figsize=(6, 4))
        counts = df_concatenado[variable].value_counts()
        labels = counts.index
        plt.pie(counts, labels=labels,autopct='%1.1f%%')
        plt.title(f'Distribución de {variable}')
        st.pyplot()  # Usar st.pyplot() para mostrar el gráfico
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

    # Mostrar la matriz de correlación como un gráfico de calor
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

    # Añade el resumen del proceso de escalamiento y codificación aquí
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





import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import joblib




def cargar_y_preprocesar_datos(train_csv, test_csv):
    # Cargar los datos desde los archivos CSV
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Eliminar filas con valores nulos en todos los campos
    df_train = df_train.dropna(how='all')

    # Agregar una columna 'dataset' para distinguir entre conjuntos de entrenamiento y prueba
    df_train['dataset'] = 'train'
    df_test['dataset'] = 'test'

    # Concatenar los conjuntos de entrenamiento y prueba
    df_concatenado = pd.concat([df_train, df_test], ignore_index=True)

    # Guardar la columna "CustomerID" en una variable y eliminarla del DataFrame
    customer_ids = df_concatenado["CustomerID"]
    df_concatenado = df_concatenado.drop("CustomerID", axis=1)

    return df_concatenado, customer_ids


def escalar_y_codificar(df):
    # Aplica get_dummies a las variables categóricas
    dummies = pd.get_dummies(df[["Gender", 'Subscription Type']], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(["Gender", 'Subscription Type'], axis=1)

    # Codifica la variable Contract Length
    df['Contract Length_cod'] = df['Contract Length'].apply(lambda x: 1 if x in ('Annual', 'Quarterly') else 0)

    # Selecciona las columnas numéricas para escalar
    variables_scaler = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

    # Crea un objeto MinMaxScaler
    scaler = MinMaxScaler()

    # Aplica la estandarización a las columnas numéricas seleccionadas
    df[variables_scaler] = scaler.fit_transform(df[variables_scaler])

    return df


def preparar_datos(df):
    # Filtrar el DataFrame para obtener los conjuntos de entrenamiento y prueba
    train_data = df[df["dataset"] == "train"]
    test_data = df[df["dataset"] == "test"]

    variables_features = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Male', 'Contract Length_cod']
    variable_target = "Churn"

    # Seleccionar las variables de características y el objetivo para entrenamiento y prueba
    X_train = train_data[variables_features]
    y_train = train_data[variable_target]
    X_test = test_data[variables_features]
    y_test = test_data[variable_target]

    return X_train, y_train, X_test, y_test




def cargar_y_predecir_modelo(nuevos_datos):
    
    modelo_pkl = "src/models/modelo_lr_mejor.pkl" 

    # modelo previamente entrenado desde el archivo .pkl
    modelo = joblib.load(modelo_pkl)

    # predicciones en los nuevos datos
    predicciones = modelo.predict(nuevos_datos)

    # probabilidades de las predicciones (0 y 1)
    probabilidades = modelo.predict_proba(nuevos_datos)

    importancias = modelo.coef_[0]

    # predicciones + probabilidades
    return predicciones, probabilidades , importancias
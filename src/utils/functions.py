
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

# --- INICIO: Lógica de Rutas para cargar el modelo ---
# Obtener la ruta del directorio 'utils' donde se encuentra este archivo
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
# Subir dos niveles para llegar a la raíz del proyecto (desde 'utils' a 'src', y de 'src' a la raíz)
PROJECT_ROOT = os.path.abspath(os.path.join(UTILS_DIR, os.pardir, os.pardir))
# --- FIN: Lógica de Rutas ---

def cargar_y_preprocesar_datos(ruta_train_csv, ruta_test_csv):
    """
    Carga y preprocesa los datos de churn de clientes desde archivos CSV.

    Args:
        ruta_train_csv (str): Ruta al archivo del conjunto de datos de entrenamiento.
        ruta_test_csv (str): Ruta al archivo del conjunto de datos de prueba.

    Returns:
        df_concatenado (pd.DataFrame): Datos concatenados y preprocesados.
        ids_clientes (pd.Series): IDs de los clientes.
    """
    df_train = pd.read_csv(ruta_train_csv)
    df_test = pd.read_csv(ruta_test_csv)

    df_train = df_train.dropna(how='all')

    df_train['dataset'] = 'train'
    df_test['dataset'] = 'test'

    df_concatenado = pd.concat([df_train, df_test], ignore_index=True)

    ids_clientes = df_concatenado["CustomerID"]
    df_concatenado = df_concatenado.drop("CustomerID", axis=1)

    return df_concatenado, ids_clientes

def escalar_y_codificar(df):
    """
    Escala y codifica variables categóricas en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        df (pd.DataFrame): DataFrame con variables escaladas y codificadas.
    """
    # Crear una copia para evitar modificar el DataFrame original que se muestra en la app
    df_procesado = df.copy()

    # Codificar Gender y Subscription Type como columnas binarias
    dummies = pd.get_dummies(df_procesado[['Gender', 'Subscription Type']], drop_first=True)
    dummies = dummies.astype(int)
    df_procesado = pd.concat([df_procesado, dummies], axis=1)
    df_procesado = df_procesado.drop(['Gender', 'Subscription Type'], axis=1)

    # Codificar la variable Contract Length
    df_procesado['Contract Length_cod'] = df_procesado['Contract Length'].apply(lambda x: 1 if x in ('Annual', 'Quarterly') else 0)

    # Seleccionar columnas numéricas para escalar
    variables_a_escalar = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

    # Crear un objeto MinMaxScaler
    scaler = MinMaxScaler()

    # Aplicar escalado a las columnas numéricas seleccionadas
    df_procesado[variables_a_escalar] = scaler.fit_transform(df_procesado[variables_a_escalar])

    return df_procesado

def cargar_y_predecir_modelo(nuevos_datos):
    """
    Carga un modelo pre-entrenado y realiza predicciones sobre nuevos datos.

    Args:
        nuevos_datos (pd.DataFrame): Nuevos datos para la predicción.

    Returns:
        predicciones (array): Clases predichas (0 o 1).
        probabilidades (array): Probabilidades predichas para cada clase.
        importancias (array): Coeficientes del modelo.
    """
    # Construir la ruta al modelo de forma robusta
    ruta_modelo_pkl = os.path.join(PROJECT_ROOT, 'src', 'models', 'modelo_lr_mejor.pkl')

    # Cargar el modelo pre-entrenado desde el archivo .pkl
    modelo = joblib.load(ruta_modelo_pkl)

    # Realizar predicciones sobre los nuevos datos
    predicciones = modelo.predict(nuevos_datos)

    # Obtener las probabilidades predichas (0 y 1)
    probabilidades = modelo.predict_proba(nuevos_datos)

    # Obtener la importancia de las características (coeficientes)
    importancias = modelo.coef_[0]

    # Devolver predicciones, probabilidades e importancias
    return predicciones, probabilidades, importancias


# --- Funciones de Ayuda para Entrenamiento y Evaluación (usadas en el notebook) ---

def preparar_datos(df):
    """
    Prepara los datos para el entrenamiento y la prueba.

    Args:
        df (pd.DataFrame): DataFrame preprocesado.

    Returns:
        X_train, y_train, X_test, y_test: Divisiones de datos de entrenamiento y prueba.
    """
    train_data = df[df["dataset"] == "train"]
    test_data = df[df["dataset"] == "test"]

    variables_features = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Male', 'Contract Length_cod']
    variable_target = "Churn"

    X_train = train_data[variables_features]
    y_train = train_data[variable_target]
    X_test = test_data[variables_features]
    y_test = test_data[variable_target]

    return X_train, y_train, X_test, y_test

def evaluar_clasificador(modelo, X_train, y_train, X_test, y_test):
    """Evalúa un clasificador y devuelve sus métricas."""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

def graficar_matriz_confusion(y_test, y_pred, title="Matriz de Confusión"):
    """Grafica la matriz de confusión."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')
    plt.title(title)
    st.pyplot(plt) # Usar st.pyplot para mostrar en Streamlit si es necesario

def graficar_curva_roc(y_test, y_probs, nombre_modelo):
    """Grafica la curva ROC."""
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC para {nombre_modelo}')
    plt.legend(loc='lower right')
    st.pyplot(plt) # Usar st.pyplot para mostrar en Streamlit si es necesario

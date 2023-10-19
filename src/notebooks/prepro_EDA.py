import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.append("C:/Users/Hp/Desktop/ProyectoML_churn")


from src.utils.functions import cargar_y_preprocesar_datos, escalar_y_codificar


train_csv = 'src/data/raw/customer_churn_dataset-training-master.csv'
test_csv = 'src/data/raw/customer_churn_dataset-testing-master.csv'


df_concatenado, customer_ids = cargar_y_preprocesar_datos(train_csv, test_csv)

#información general del DataFrame
df_concatenado.info()
df_concatenado.nunique()
df_concatenado.head(5)

# Variables numéricas
variables_numericas = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

# Variables categóricas
variables_categoricas = ['Gender', 'Subscription Type', 'Contract Length']

# ANALISIS UNIVARIANTE

# estadístico de variables numéricas
estadisticos_num = df_concatenado.drop(columns="Churn").describe().T
estadisticos_num

# Visualización de Distribuciones
for variable in variables_numericas:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_concatenado, x=variable)
    plt.title(f'Distribución de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.show()

# Visualización de Diagramas de Caja
for variable in variables_numericas:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_concatenado, y=variable)
    plt.title(f'Gráfico de Bigote de {variable}')
    plt.ylabel(variable)
    plt.show()

# Visualización de la Distribución de Variables Categóricas
for variable in variables_categoricas:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_concatenado, x=variable)
    plt.title(f'Distribución de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.show()


#Age: La edad promedio es de aproximadamente 39 años, con una distribución que va desde 18 hasta 65 años. 

#Tenure: La tenencia promedio es de aproximadamente 31 meses, con valores que van desde 1 hasta 60 meses.

#Usage Frequency: La frecuencia de uso promedio es de aproximadamente 15,7, con valores que van desde 1 hasta 30.

#Support Calls: El número promedio de llamadas de soporte es de aproximadamente 3,8, con valores que van desde 0 hasta 10.

#Payment Delay: El retraso promedio en el pago es de aproximadamente 13,5, con valores que van desde 0 hasta 30.

#Total Spend: El gasto total promedio es de aproximadamente 620, con valores que van desde 100 hasta 1,000.

#Last Interaction: El tiempo promedio desde la última interacción es de aproximadamente 14,6 unidades de tiempo, con valores que van desde 1 hasta 30.

#Las variables siguen una distribucion normal y no poseen valores atipicos

#Tenemos mas mujeres que hombres

#El contrato mensual es el que presenta menos frecuencia


# ANALISIS BIVARIANTE


# Relacion entre variables numericas y target
for variable in variables_numericas:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_concatenado, x=variable, hue='Churn', kde=True)
    plt.title(f'Distribución de {variable} por Churn')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.show()

for variable in variables_numericas:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_concatenado, x='Churn', y=variable)
    plt.title(f'Relación entre Churn y {variable}')
    plt.xlabel('Churn')
    plt.ylabel(variable)
    plt.show()


# La variable Age tiene indicencia en el target. El promedio de edad de los que dejan la empresa es mayor a los que no

# Las variables Tenure y Usage Frequency parece no tener injerencia en la variable target

# Support Calls tiene mucha diferencia entre los que se quedan en la empresa (bajo promedio de llamadas) y los que se fueron (alto promedio de llamadas).
# Encontramos valores atipicos en los clientes que se quedan en la empresa. ( luego vamos a analizar)

# Los clientes que se van de la empresa tienen un promedio mas alto de dias en demoras de pagos.

# La variable Total Spend tambien parece tener injerencia en nuestros target. Los clientes que mas gastan son los que deciden quedarse en la empresa.
# Encontramos valores atipicos en los clientes que se quedan en la empresa. ( luego vamos a analizar)

# Por ultimo observamos que los clientes que se van de la empresa en promedio, hace mas tiempo que no realizan transacciones.

##################

# Relacion entre variables categóticas y target (churn)
for variable in variables_categoricas:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_concatenado, x=variable, hue='Churn')
    plt.title(f'Relación entre Churn y {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.legend(title='Churn', loc='upper right', labels=['No Churn', 'Churn'])
    plt.show()


# tabla variables categóricas y Churn
for variable in variables_categoricas:
    tabla_frecuencia = pd.crosstab(index=df_concatenado[variable], columns=df_concatenado['Churn'], normalize='index')
    print(tabla_frecuencia)


# Las mujeres parecen ser mas propensas a irse de la empresa. (variable Gender)

# No se observan diferencias entre los diferentes tipos de suscripciones

# Los contratos anuales y cuatrimestrales tienen un comportamiento similar con el target, pero los clientes con contratos mensuales se van de la empresa.

#####################


 # ANALISIS MULTIVARIANTE


# Grafico boxplot de variables cat y num dividos por churn
for variable_cat in variables_categoricas:
    for variable_num in variables_numericas:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_concatenado, x=variable_cat, y=variable_num, hue='Churn')
        plt.title(f'Relación entre {variable_cat} y {variable_num} por Churn')
        plt.xlabel(variable_cat)
        plt.ylabel(variable_num)
        plt.legend(title='Churn', loc='upper right', labels=['No Churn', 'Churn'])
        plt.xticks(rotation=45)
        plt.show()

df_churn_0 = df_concatenado[df_concatenado['Churn'] == 0]
df_churn_1 = df_concatenado[df_concatenado['Churn'] == 1]

# Calculamos estadísticas descriptivas para cada churn
churn_0_estadisticos = df_churn_0.describe().T
churn_1_estadisticos = df_churn_1.describe().T

print(churn_0_estadisticos)
print(churn_1_estadisticos)

#############################

# TRATAMIENTO DE OUTLIERS

# creamos una copia

df_concatenado_copy = df_concatenado.copy()

# RIQ para Support Calls y Total Spend en Churn 0
Q1_SC = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Support Calls'].quantile(0.25)
Q3_SC = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Support Calls'].quantile(0.75)
IQR_SC = Q3_SC - Q1_SC

Q1_TS = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Total Spend'].quantile(0.25)
Q3_TS = df_concatenado.loc[df_concatenado['Churn'] == 0, 'Total Spend'].quantile(0.75)
IQR_TS = Q3_TS - Q1_TS

# Lim sup e inf para Support Calls y Total Spend
limite_superior_SC = Q3_SC + 1.5 * IQR_SC
limite_inferior_TS = Q1_TS - 1.5 * IQR_TS

# eliminar filas con valores atípicos en Support Calls y Total Spend
df_concatenado = df_concatenado[~((df_concatenado['Churn'] == 0) & ((df_concatenado['Support Calls'] > limite_superior_SC) | (df_concatenado['Total Spend'] < limite_inferior_TS)))]


# Correlaciones entre variables numéricas
correlacion_matrix = df_concatenado[['Churn','Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Total Spend','Last Interaction','Payment Delay']].corr()
sns.heatmap(correlacion_matrix, annot=True, cmap='coolwarm')
plt.show()


# CODIFICAMOS Y ESCALAMOS VARIABLES

df_concatenado = escalar_y_codificar(df_concatenado)


df_clean = df_concatenado.copy()

csv_ruta = "src/data/processed/df_clean.csv"

df_clean.to_csv(csv_ruta, index=False)
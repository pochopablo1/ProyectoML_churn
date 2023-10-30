import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.getcwd()))


# Import custom functions from your project
from src.utils.functions import load_and_preprocess_data, scale_and_encode

# Define file paths
train_csv = 'src/data/raw/customer_churn_dataset-training-master.csv'
test_csv = 'src/data/raw/customer_churn_dataset-testing-master.csv'

# Load and preprocess data
df_concatenated, customer_ids = load_and_preprocess_data(train_csv, test_csv)

# General information about the DataFrame
df_concatenated.info()
df_concatenated.nunique()
df_concatenated.head(5)

# Numeric variables
numeric_variables = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

# Categorical variables
categorical_variables = ['Gender', 'Subscription Type', 'Contract Length']

# UNIVARIATE ANALYSIS

# Descriptive statistics for numeric variables
numeric_statistics = df_concatenated.drop(columns="Churn").describe().T
numeric_statistics

# Visualization of Distributions
for variable in numeric_variables:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_concatenated, x=variable)
    plt.title(f'Distribution of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.show()

# Box Plots
for variable in numeric_variables:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df_concatenated, y=variable)
    plt.title(f'Box Plot of {variable}')
    plt.ylabel(variable)
    plt.show()

for variable in categorical_variables:
    plt.figure(figsize=(6, 4))
    counts = df_concatenated[variable].value_counts()
    labels = counts.index
    plt.pie(counts, labels=labels)
    plt.title(f'Distribution of {variable}')
    plt.show()

# BIVARIATE ANALYSIS

# Relationship between numeric variables and target
for variable in numeric_variables:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_concatenated, x=variable, hue='Churn', kde=True)
    plt.title(f'Distribution of {variable} by Churn')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    plt.show()

for variable in numeric_variables:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_concatenated, x='Churn', y=variable)
    plt.title(f'Relationship between Churn and {variable}')
    plt.xlabel('Churn')
    plt.ylabel(variable)
    plt.show()

# Multivariate Analysis

# Boxplots of categorical and numeric variables by Churn
for categorical_variable in categorical_variables:
    for numeric_variable in numeric_variables:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_concatenated, x=categorical_variable, y=numeric_variable, hue='Churn')
        plt.title(f'Relationship between {categorical_variable} and {numeric_variable} by Churn')
        plt.xlabel(categorical_variable)
        plt.ylabel(numeric_variable)
        plt.legend(title='Churn', loc='upper right', labels=['No Churn', 'Churn'])
        plt.xticks(rotation=45)
        plt.show()

df_churn_0 = df_concatenated[df_concatenated['Churn'] == 0]
df_churn_1 = df_concatenated[df_concatenated['Churn'] == 1]

# Calculate descriptive statistics for each Churn category
churn_0_statistics = df_churn_0.describe().T
churn_1_statistics = df_churn_1.describe().T

print(churn_0_statistics)
print(churn_1_statistics)

# OUTLIER HANDLING

# Create a copy of the DataFrame
df_concatenated_copy = df_concatenated.copy()

# Calculate IQR for Support Calls and Total Spend in Churn 0
Q1_SC = df_concatenated.loc[df_concatenated['Churn'] == 0, 'Support Calls'].quantile(0.25)
Q3_SC = df_concatenated.loc[df_concatenated['Churn'] == 0, 'Support Calls'].quantile(0.75)
IQR_SC = Q3_SC - Q1_SC

Q1_TS = df_concatenated.loc[df_concatenated['Churn'] == 0, 'Total Spend'].quantile(0.25)
Q3_TS = df_concatenated.loc[df_concatenated['Churn'] == 0, 'Total Spend'].quantile(0.75)
IQR_TS = Q3_TS - Q1_TS

# Calculate upper and lower limits for Support Calls and Total Spend
upper_limit_SC = Q3_SC + 1.5 * IQR_SC
lower_limit_TS = Q1_TS - 1.5 * IQR_TS

# Remove rows with outliers in Support Calls and Total Spend
df_concatenated = df_concatenated[~((df_concatenated['Churn'] == 0) & ((df_concatenated['Support Calls'] > upper_limit_SC) | (df_concatenated['Total Spend'] < lower_limit_TS)))]


# Correlations between numeric variables
correlation_matrix = df_concatenated[['Churn', 'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Total Spend', 'Last Interaction', 'Payment Delay']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# ENCODE AND SCALE VARIABLES

df_concatenated = scale_and_encode(df_concatenated)

df_clean = df_concatenated.copy()
print(df_clean)

# Save the cleaned DataFrame to a CSV file
csv_path = "src/data/processed/df_clean.csv"
df_clean.to_csv(csv_path, index=False)
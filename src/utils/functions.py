

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import MinMaxScaler
import joblib




def load_and_preprocess_data(train_csv, test_csv):
    """
    Load and preprocess customer churn data from CSV files.

    Args:
        train_csv (str): File path to the training dataset.
        test_csv (str): File path to the testing dataset.

    Returns:
        df_concatenated (pd.DataFrame): Concatenated and preprocessed data.
        customer_ids (pd.Series): Customer IDs.
    """
    # Load data from CSV files
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Remove rows with all missing values
    df_train = df_train.dropna(how='all')

    # Add a 'dataset' column to distinguish between training and testing sets
    df_train['dataset'] = 'train'
    df_test['dataset'] = 'test'

    # Concatenate training and testing sets
    df_concatenated = pd.concat([df_train, df_test], ignore_index=True)

    # Save the "CustomerID" column in a variable and remove it from the DataFrame
    customer_ids = df_concatenated["CustomerID"]
    df_concatenated = df_concatenated.drop("CustomerID", axis=1)

    return df_concatenated, customer_ids

def scale_and_encode(df):
    """
    Scale and encode categorical variables in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        df (pd.DataFrame): DataFrame with scaled and encoded variables.
    """
    # Encode Gender and Subscription Type as binary columns
    dummies = pd.get_dummies(df[['Gender', 'Subscription Type']], drop_first=True)
    dummies = dummies.astype(int)  # Convert binary columns to integers (0 and 1)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(['Gender', 'Subscription Type'], axis=1)

    # Encode the Contract Length variable
    df['Contract Length_cod'] = df['Contract Length'].apply(lambda x: 1 if x in ('Annual', 'Quarterly') else 0)

    # Select numeric columns for scaling
    variables_to_scale = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']

    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Apply scaling to the selected numeric columns
    df[variables_to_scale] = scaler.fit_transform(df[variables_to_scale])

    return df




def prepare_data(df):
    """
    Prepare data for training and testing.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        X_train, y_train, X_test, y_test: Training and testing data splits.
    """
    # Filter the DataFrame to obtain training and testing sets
    train_data = df[df["dataset"] == "train"]
    test_data = df[df["dataset"] == "test"]

    # Define feature variables and target variable
    variables_features = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Male', 'Contract Length_cod']
    variable_target = "Churn"

    # Select feature and target variables for training and testing
    X_train = train_data[variables_features]
    y_train = train_data[variable_target]
    X_test = test_data[variables_features]
    y_test = test_data[variable_target]

    return X_train, y_train, X_test, y_test

def load_and_predict_model(new_data):
    """
    Load a pre-trained model and make predictions on new data.

    Args:
        new_data (pd.DataFrame): New data for prediction.

    Returns:
        predictions (array): Predicted classes (0 or 1).
        probabilities (array): Predicted probabilities for each class.
        importances (array): Coefficients of the model.
    """
    model_pkl = "src/models/modelo_lr_mejor.pkl"

    # Load the pre-trained model from the .pkl file
    model = joblib.load(model_pkl)

    # Make predictions on the new data
    predictions = model.predict(new_data)

    # Get predicted probabilities (0 and 1)
    probabilities = model.predict_proba(new_data)

    # Get feature importances (coefficients)
    importances = model.coef_[0]

    # Return predictions, probabilities, and importances
    return predictions, probabilities, importances




def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


def evaluate_classifier(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_test, y_probs, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    plt.show()
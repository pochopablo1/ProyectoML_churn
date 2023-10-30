import matplotlib.pyplot as plt
import joblib
import streamlit as st
import pandas as pd
import os
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from utils.functions import scale_and_encode, load_and_predict_model, load_and_preprocess_data, prepare_data

# Sidebar menu
st.sidebar.title('Menu')
section = st.sidebar.radio('Go to section:', ('Introduction', 'EDA Analysis', 'Model Training', 'Prediction', 'Conclusions and Recommendations'))

# Section content
if section == 'Introduction':
    st.title('Introduction')
    
    # Project information
    st.write("- TelecomConnect is a telecommunications provider facing the challenge of retaining its customers in a highly competitive market. Customer retention is essential for the long-term success of the company.")
    st.write("- Customer retention refers to a company's ability to keep its current customers from churning or canceling their services. This metric is critical in a market where acquiring new customers can be costly, and retaining existing customers can be more profitable.")
    st.write("- In this project, we have been commissioned by TelecomConnect to conduct an in-depth analysis of their data and develop a churn prediction model. This model will allow us to anticipate customer churn and take proactive measures to retain them.")
    st.write("- Our goal is to help TelecomConnect better understand their customers, identify behavioral patterns, and predict who is more likely to churn. By doing so, the company can implement more effective retention strategies and improve customer satisfaction.")
    st.write("- Throughout this project, we will use a database provided by TelecomConnect containing various customer-related variables and their interactions with telecommunications services.")

elif section == 'EDA Analysis':
    st.write('Welcome to the EDA Analysis section.')

    sys.path.append("C:/Users/Hp/Desktop/ProyectoML_churn")
    train_csv = 'src/data/raw/customer_churn_dataset-training-master.csv'
    test_csv = 'src/data/raw/customer_churn_dataset-testing-master.csv'
    df_concatenated, customer_ids = load_and_preprocess_data(train_csv, test_csv)

    # Section title
    st.title('Exploratory Data Analysis (EDA)')

    # Section title
    st.write("## Customer DataFrame Information")

    # Variable descriptions
    st.write("The DataFrame contains detailed information about customers and their interactions with a company. The following variables are present in this dataset:")
    st.write("- **CustomerID**: Unique identifier for each customer. Data type: float.")
    st.write("- **Age**: Age of the customers. Data type: float.")
    st.write("- **Gender**: Gender of the customers. Data type: object.")
    st.write("- **Tenure**: Customer's tenure with the company. Data type: float.")
    st.write("- **Usage Frequency**: Frequency of service usage. Data type: float.")
    st.write("- **Support Calls**: Number of support calls. Data type: float.")
    st.write("- **Payment Delay**: Payment delay. Data type: float.")
    st.write("- **Subscription Type**: Customer's subscription type. Data type: string.")
    st.write("- **Contract Length**: Contract duration. Data type: string.")
    st.write("- **Total Spend**: Customer's total spending. Data type: float.")
    st.write("- **Last Interaction**: Date of the last customer interaction. Data type: float.")
    st.write("- **Churn**: Customer churn indicator. Data type: float.")
    st.write("- **Dataset**: Dataset label. Data type: string.")
    st.write("These variables provide valuable information about the company's customer base, including demographic details, usage behavior, and customer retention metrics. You can use this information for analysis and visualizations in your Streamlit application.")

    # Numeric and categorical variables
    numeric_variables = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    categorical_variables = ['Gender', 'Subscription Type', 'Contract Length']

    st.subheader('Statistics for Numeric Variables:')
    st.write(df_concatenated[numeric_variables].describe())

    # Visualization of Categorical Variables Distribution
    st.subheader('Visualization of Categorical Variables Distribution:')
    for variable in categorical_variables:
        plt.figure(figsize=(6, 4))
        counts = df_concatenated[variable].value_counts()
        labels = counts.index
        plt.pie(counts, labels=labels, autopct='%1.1f%%')
        plt.title(f'Distribution of {variable}')
        st.pyplot()
        st.write(f'Distribution of {variable}')

    st.write('Here are some key statistics for the variables:')
    st.write('- Age: The average age is approximately 39 years, with a range from 18 to 65 years.')
    st.write('- Tenure: The average tenure is approximately 31 months, with values ranging from 1 to 60 months.')
    st.write('- Usage Frequency: The average usage frequency is approximately 15.7, with values ranging from 1 to 30.')
    st.write('- Support Calls: The average number of support calls is approximately 3.8, with values ranging from 0 to 10.')
    st.write('- Payment Delay: The average payment delay is approximately 13.5, with values ranging from 0 to 30.')
    st.write('- Total Spend: The average total spending is approximately 620, with values ranging from 100 to 1,000.')
    st.write('- Last Interaction: The average time since the last interaction is approximately 14.6 time units, with values ranging from 1 to 30.')

    # Relationship between numeric variables and target
    for variable in numeric_variables:
        st.subheader(f'Relationship between {variable} and Churn:')
        
        # Histogram
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df_concatenated, x=variable, hue='Churn', kde=True)
        plt.title(f'Distribution of {variable} by Churn')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        st.pyplot()
        plt.close()
        
        # Box Plot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df_concatenated, x='Churn', y=variable)
        plt.title(f'Relationship between Churn and {variable}')
        plt.xlabel('Churn')
        plt.ylabel(variable)
        st.pyplot()
        plt.close()
    
    st.write('The Age variable has an impact on the target. The average age of those who leave the company is higher than those who stay.')
    st.write('The Tenure and Usage Frequency variables seem to have no influence on the target.')
    st.write('Support Calls show a significant difference between those who stay with the company (low average calls) and those who left (high average calls). We find outliers in customers who stay with the company (we will analyze later).')
    st.write('Customers who leave the company have a higher average of payment delay days.')
    st.write('The Total Spend variable also appears to influence our target. Customers who spend more are the ones who decide to stay with the company. We find outliers in customers who stay with the company (we will analyze later).')
    st.write('Lastly, we observe that customers who leave the company, on average, have not made transactions for a longer time.')

    # Relationship between categorical variables and target (churn)
    for variable in categorical_variables:
        st.subheader(f'Relationship between {variable} and Churn:')
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_concatenated, x=variable, hue='Churn')
        plt.title(f'Relationship between Churn and {variable}')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.legend(title='Churn', loc='upper right', labels=['No Churn', 'Churn'])
        st.pyplot()
        plt.close()

    st.write('Females seem to be more likely to leave the company (Gender variable).')
    st.write('No differences are observed between different types of subscriptions.')
    st.write('Annual and quarterly contracts have a similar behavior with the target, but customers with monthly contracts are leaving the company.')

    # Correlations between numeric variables
    correlation_matrix = df_concatenated[['Churn', 'Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Total Spend', 'Last Interaction', 'Payment Delay']].corr()

    st.subheader('Correlation Matrix:')
    st.write('The correlation matrix shows the relationships between numeric variables and the target (Churn).')

    # Show the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    st.pyplot()
    plt.close()

    # Comments on the correlation matrix
    st.write('Comments on the correlation matrix:')
    st.write('Churn is highly correlated with Support Calls, Total Spend, and Payment Delay.')
    st.write('Age and Tenure have low correlations with Churn, suggesting limited influence on retention.')
    st.write('Support Calls and Payment Delay also have significant correlations with Total Spend.')

    # Data Scaling and Encoding Process
    st.subheader('Data Scaling and Encoding Process:')
    st.write('In this code block, a series of data transformations were performed to prepare them for analysis and modeling. The main actions taken are summarized below:')
    st.write('1. Encoding Categorical Variables: One-hot encoding (get_dummies) was applied to the categorical variable "Gender." This was done to convert categorical variables into binary numeric variables (0 or 1), which is essential for machine learning algorithms to use.')
    st.write('2. Encoding of the "Contract Length" Variable: A new variable called "Contract Length_cod" was created, set to 1 if the original value in "Contract Length" is "Annual" or "Quarterly," and 0 for "Monthly." This allows binary representation of the contract duration. This action was taken because the unmodified variable had a negative impact on the model, and it was observed that there were almost no differences between annual and quarterly contracts, so they were grouped into 1 and monthly contracts into another group.')
    st.write('3. Standardization of Numeric Variables: The numeric columns "Age," "Support Calls," "Payment Delay," "Total Spend," and "Last Interaction" were selected. MinMaxScaler was then used to standardize these variables, ensuring that they all have a similar value range (between 0 and 1) for better performance in machine learning models.')


elif section == 'Model Training':
    st.write('Welcome to the Model Training section.')

    # Load model training results
    path = 'C:/Users/Hp/Desktop/ProyectoML_churn/src/data/processed/df_resultados.csv'
    results_df = pd.read_csv(path)

    # Information about model training
    st.write('(We did not use the train-test split as the test dataset was provided by the company.)')

    # Used Models
    st.subheader('Used Models')
    st.write('To train the model, we used the following classifiers:')
    st.write('- Logistic Regression')
    st.write('- Decision Tree')
    st.write('- Random Forest')
    st.write('- Gradient Boosting')
    st.write('- K-Nearest Neighbor')
    st.write('- Gaussian Naive Bayes')

    # Display model training results
    st.subheader('Model Training Results')
    st.dataframe(results_df)

    st.subheader('Hyperparameter Results')
    st.title("Model Results")

    # Dictionaries
    results = {
        "Random Forest": {
            "Accuracy": 0.6311,
            "Classification Report": """
                precision    recall  f1-score   support
            0.0       0.98      0.10      0.18     21097
            1.0       0.62      1.00      0.76     30493
        accuracy                           0.63     51590
    macro avg       0.80      0.55      0.47     51590
    weighted avg       0.76      0.63      0.52     51590
            """
        },
        "Decision Tree": {
            "Accuracy": 0.6381,
            "Classification Report": """
                precision    recall  f1-score   support
            0.0       0.98      0.12      0.21     21097
            1.0       0.62      1.00      0.77     30493
        accuracy                           0.64     51590
    macro avg       0.80      0.56      0.49     51590
    weighted avg       0.77      0.64      0.54     51590
            """
        },
        "Logistic Regression": {
            "Accuracy": 0.7198,
            "Classification Report": """
                precision    recall  f1-score   support
            0.0       0.94      0.33      0.49     21097
            1.0       0.68      0.99      0.81     30493
        accuracy                           0.72     51590
    macro avg       0.81      0.66      0.65     51590
    weighted avg       0.79      0.72      0.68     51590
            """
        },
        "K-Nearest Neighbors": {
            "Accuracy": 0.6423,
            "Classification Report": """
                precision    recall  f1-score   support
            0.0       0.96      0.13      0.23     21097
            1.0       0.62      1.00      0.77     30493
        accuracy                           0.64     51590
    macro avg       0.79      0.56      0.50     51590
    weighted avg       0.76      0.64      0.55     51590
            """
        },
        "Gaussian Naive Bayes": {
            "Accuracy": 0.6581,
            "Classification Report": """
                precision    recall  f1-score   support
            0.0       0.98      0.17      0.29     21097
            1.0       0.63      1.00      0.78     30493
        accuracy                           0.66     51590
    macro avg       0.81      0.58      0.53     51590
    weighted avg       0.78      0.66      0.58     51590
            """
        }
    }

    for model, data in results.items():
        st.subheader(model)
        st.markdown(f"Accuracy: {data['Accuracy']:.4f}")
        st.write(f"Classification Report:\n{data['Classification Report']}")

    st.write('Reasons for choosing Logistic Regression:')
    st.write('- Reliable Performance (ROC-AUC: 0.76):')
    st.write('- Like other models, it predicts almost 0.1 for positives (customers who leave).')
    st.write('- Balance Between Sensitivity and Specificity:')
    st.write('- This model achieves an accuracy of 0.71, an effective balance between detecting customers who will churn and those who will not, which is crucial in a churn problem.')

elif section == 'Prediction':
    st.write('Welcome to the Prediction section.')

    # Upload CSV file
    upload_file = st.file_uploader("Upload CSV file", type=["csv"])

    feature_variables = ['Age', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction', 'Gender_Male', 'Contract Length_cod']
    target_variable = "Churn"

    if upload_file is not None:
        # Read the CSV file
        df = pd.read_csv(upload_file)
        df_copy = df.copy()

        # Load the model and make predictions:
        if st.button("Load Model and Make Predictions"):
            # Perform data scaling and encoding
            df = scale_and_encode(df)

            X_test = df[feature_variables]

            # Make predictions and get probabilities
            prediction_results, probability_results, importance_results = load_and_predict_model(X_test)

            # Create a new DataFrame with predictions and probabilities
            predictions_df = pd.DataFrame({'Predictions': prediction_results, 'Probability 0': probability_results[:, 0], 'Probability 1': probability_results[:, 1]})

            # Concatenate the original DataFrame with the predictions DataFrame
            result_df = pd.concat([df_copy, predictions_df], axis=1)

            # Display the results
            st.write("Prediction Results:")
            st.write(result_df)

            # Create a histogram to show probabilities
            import matplotlib.pyplot as plt

            # Histogram of probabilities
            st.subheader('Probability Histogram')
            plt.hist(probability_results[:, 1], bins=20, color='blue', alpha=0.7)
            plt.xlabel('Churn Probability')
            plt.ylabel('Frequency')
            st.pyplot(plt)

            # Create a DataFrame with features and their importances
            feature_importance = pd.DataFrame({'Feature': feature_variables, 'Importance': importance_results})
            feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

            # Plot importances
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance')
            plt.xlabel('Importance')
            plt.ylabel('Feature')

            # Display the feature importance plot
            st.subheader('Feature Importance')
            st.pyplot(plt)

            # Add space
            st.markdown('<br>', unsafe_allow_html=True)

            st.write(
                '<style>div.Widget.row-widget.stButton > div{background-color: #3498db; color: white; text-align: center}</style>',
                unsafe_allow_html=True
            )

            # Add space
            st.markdown('<br>', unsafe_allow_html=True)

elif section == 'Conclusions and Recommendations':
    st.title('Conclusions and Recommendations')

    # Conclusions and recommendations

    st.write('The created model has an accuracy of 70%, so we request more information from the company '
             '(additional variables and data) to attempt to improve the model and its predictions.')

    st.write('The variables that most affect customer churn are support calls, '
             'suggesting a need to pay attention to the quality of service provided by the company to prevent customer churn. '
             'Customers making more calls likely have unresolved issues.')

    st.write('Additionally, it was observed that monthly contracts have a high average churn rate. '
             'It is recommended to explore strategies to retain these customers, such as offering longer-term contracts.')

    st.write('We suggest generating a retention campaign using predictive analysis to evaluate its effectiveness. '
             'In the meantime, we look forward to obtaining new data to improve the model and, therefore, the predictions.')

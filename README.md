# TASK_04_DA
📊 Customer Churn Prediction in Telecom


📌 Project Overview

Customer churn is a major challenge in the telecom industry, where retaining existing customers is often more cost-effective than acquiring new ones.
This project builds a machine learning pipeline to predict whether a customer is likely to churn based on demographic, service usage, and account information.

We use the Telco Customer Churn dataset (Kaggle) and apply different ML models to classify customers into churn (Yes/No) categories.

🎯 Objectives

Perform data preprocessing (handle missing values, encode categorical variables, scale numerical features).

Build classification models (Logistic Regression & Random Forest).

Compare performance of models.

Identify key features influencing churn using feature importance.

Save the trained model for future deployment.

🛠️ Tools & Libraries

Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (Preprocessing, Logistic Regression, Random Forest, Pipelines, Evaluation)

Joblib (for saving trained model)

🔑 Workflow
1. Data Preprocessing

Dropped irrelevant columns (customerID).

Converted TotalCharges to numeric & filled missing values.

Encoded categorical variables using OneHotEncoder.

Scaled numerical features using StandardScaler.

2. Model Training

Logistic Regression (baseline model).

Random Forest Classifier (improved performance & feature importance).

3. Model Evaluation

Metrics: Accuracy, Precision, Recall, F1-score.

Confusion Matrix to analyze predictions.

Feature Importance visualization for churn drivers.

4. Model Saving

Final model (Random Forest) saved as customer_churn_model.pkl for future deployment.

📊 Results

Logistic Regression: Good baseline but lower recall.

Random Forest: Higher accuracy & better feature importance insights.

📌 Top churn indicators (from feature importance):

Contract Type

Tenure (length of stay)

MonthlyCharges

InternetService type

PaymentMethod

📈 Future Improvements

Try XGBoost / LightGBM for better accuracy.

Implement hyperparameter tuning with GridSearchCV.

Build a Flask/Django API for deployment.

Deploy as a web app (Streamlit/Gradio) for business users.

📚 Dataset


Dataset used: Telco Customer Churn
📎 Download from Kaggle

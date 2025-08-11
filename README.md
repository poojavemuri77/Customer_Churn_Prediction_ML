# Customer_Churn_Prediction_ML

# Telco Customer Churn Prediction

## 📊 Project Overview
A comprehensive machine learning project to predict customer churn for a telecommunications company using the Telco Customer Churn dataset. This project implements multiple classification algorithms and uses advanced techniques like SMOTE for handling class imbalance.

## 🎯 Objective
Predict whether a customer will churn (leave the service) based on various customer attributes and service usage patterns, helping the business take proactive measures to retain customers.

## 📁 Dataset
- **Source**: Telco Customer Churn Dataset from kaggle
- **Size**: 7,043 customers with 21 features
- **Target Variable**: Churn (Yes/No)
- **Features**: Demographics, account information, and service usage patterns

## 🔍 Key Features Analyzed
- **Demographics**: Gender, Senior Citizen status, Partner, Dependents
- **Account Info**: Tenure, Contract type, Payment method, Monthly/Total charges
- **Services**: Phone service, Internet service, Online security, Tech support, etc.

## 🛠️ Technologies Used
- **Python Libraries**: 
  - Data Analysis: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `xgboost`
  - Data Balancing: `imbalanced-learn` (SMOTE)
- **Development Environment**: Jupyter Notebook

## 📈 Methodology

### 1. Data Exploration & Preprocessing
- **Data Cleaning**: Handled missing values in TotalCharges column
- **Feature Engineering**: Removed irrelevant features (CustomerID)
- **Encoding**: Applied Label Encoding for categorical variables
- **Class Imbalance**: Identified and addressed using SMOTE technique

### 2. Exploratory Data Analysis (EDA)
- Comprehensive visualization of feature distributions
- Correlation analysis between features and target variable
- Identification of key churn indicators

### 3. Data Preprocessing
- **Feature Scaling**: Standardized numerical features
- **Train-Test Split**: 80-20 split for model validation
- **SMOTE Implementation**: Balanced the dataset using Synthetic Minority Oversampling

### 4. Model Development
Implemented and compared multiple classification algorithms:
- **Decision Tree Classifier**
- **Random Forest Classifier** 
- **XGBoost Classifier**

### 5. Model Evaluation
- **Cross-Validation**: 5-fold cross-validation for robust performance assessment
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Detailed performance analysis

## 🏆 Results
- **Best Performing Model**: Random Forest Classifier
- **Model Performance**: Achieved highest accuracy among tested algorithms
- **Model Deployment**: Saved trained model using pickle for future predictions

## 📊 Key Insights
1. Successfully identified key factors contributing to customer churn
2. Addressed class imbalance effectively using SMOTE
3. Random Forest provided the most reliable predictions
4. Model ready for deployment with real-time prediction capabilities

## 🚀 Model Deployment
- Trained model saved as `customer_churn_model.pkl`
- Includes feature names for consistent prediction input
- Ready for integration into production systems

## 📝 Usage
```python
# Load the saved model
import pickle
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Make predictions on new data
prediction = loaded_model.predict(new_customer_data)
```

## 🔮 Future Enhancements
- Hyperparameter tuning for optimal performance
- Feature importance analysis
- Advanced ensemble methods
- Real-time prediction API development
- A/B testing framework for model validation

## 📋 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
```

## 🤝 Feedback
Feel free to reach out for any feedback and project collaboration.Reach me out at 
Linkedin: www.linkedin.com/in/pooja-vemuri77
I am interested in Data analysis,Datascience projects and Creating compelling Dashboards.

---
*This project demonstrates end-to-end machine learning workflow from data exploration to model deployment for customer churn prediction.*

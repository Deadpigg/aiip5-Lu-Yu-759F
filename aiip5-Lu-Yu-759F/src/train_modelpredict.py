import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                             explained_variance_score, max_error, accuracy_score, 
                             confusion_matrix, classification_report)
from data_loader import load_data
from preprocessing import preprocess_data
import lightgbm as lgb 
import sys

# Load and preprocess data
df = load_data()
df_cleaned = preprocess_data(df)

#********************************************************************************************************************************************************

"""
Goal 1: Developing Models to Predict Temperature Using Sensor Data

In this task, we aim to develop models that predict the temperature conditions within a farm's controlled environment. 
The goal is to optimize the temperature to ensure the plants grow under the best possible conditions, which is crucial for maximizing yield and maintaining 
healthy crops.

For this task, I will use two classification models: GLM and GBM .

Steps taken to build the models:
1. Data Preprocessing: Preparing the features and target variable for the models.
2. Model Training: Training the models using the prepared data.
3. Model Evaluation: Evaluating the models' performance using metrics such as accuracy, confusion matrix, and classification report.
4. Cross-Validation: Performing cross-validation to ensure the models' generalization to unseen data.

"""
#********************************************************************************************************************************************************

"""
Model 1: Generalized Linear Model (GLM)

GLM extends traditional linear regression by allowing different types of response variables and error distributions.
It can handle continuous, binary, count, and other types of target variables. In this case, we'll use a GLM for predicting temperature.
It assumes a linear relationship between independent variables (sensor readings) and the target variable (temperature).

Why Use GLM?:
- Interpretability: The coefficients of a GLM model clearly show how each sensor input contributes to temperature changes.
- Computational Efficiency: Works well for real-time or embedded systems where quick decisions are needed.
- Feature Importance: GLM directly provides feature coefficients, making it easy to understand the relative influence of different sensors on temperatu- 

"""

#********************************************************************************************************************************************************
# **Goal 1: Predicting Temperature using Linear Regression**

print("\n----- Training Linear Regression Model for Temperature Prediction -----\n")

# Define numerical and categorical features
numerical_features = ['Humidity Sensor (%)', 'Light Intensity Sensor (lux)', 
                      'CO2 Sensor (ppm)', 'EC Sensor (dS/m)', 'O2 Sensor (ppm)', 
                      'Nutrient N Sensor (ppm)', 'Nutrient P Sensor (ppm)', 
                      'Nutrient K Sensor (ppm)', 'pH Sensor', 'Water Level Sensor (mm)']

categorical_features = ['System Location Code', 'Previous Cycle Plant Type', 'Plant Type']

# Define features (X) and target variable (y)
X = df_cleaned.drop(columns=['Temperature Sensor (°C)', 'Plant Stage'])  
y = df_cleaned['Temperature Sensor (°C)']  

# Define preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Build Linear Regression model pipeline
linear_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
linear_model.fit(X_train, y_train)

# Predict on test set
y_pred = linear_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
n, p = len(y_test), X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
explained_variance = explained_variance_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Display evaluation metrics
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'R²', 'MAE', 'RMSE', 'Adjusted R²', 'Explained Variance', 'Max Error', 'MAPE'],
    'Value': [mse, r2, mae, rmse, adj_r2, explained_variance, max_err, mape]
})
print(metrics_df)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('Linear Regression: Actual vs Predicted Temperature')
plt.legend()
plt.show()


"""
Metrics from the Generalized Linear Model (GLM):


               Metric     Value
0                 MSE  1.283997
1                  R²  0.475058
2                 MAE  0.894531
3                RMSE  1.133136
4         Adjusted R²  0.473884
5  Explained Variance  0.475176
6           Max Error  4.438795
7                MAPE  3.814338


"""
#********************************************************************************************************************************************************

"""
Model 2: Gradient Boosting Model (GBM)

GBM is a powerful machine learning technique that builds an ensemble of decision trees, where each tree learns to correct the errors of the previous trees. 
GBM models (LightGBM) are widely used in predictive modeling because they handle complex, nonlinear relationships effectively.

Why Use GBM?
- Captures Nonlinear Relationships: Unlike GLM, GBM can model complex interactions between temperature and environmental factors.
- Feature Interactions: Automatically learns how multiple features work together to influence temperature.
- Handles Missing Data Well: Unlike GLM, GBM can process missing values without explicit imputation.
- Higher Accuracy: Usually provides better predictive performance compared to linear models.

"""


#********************************************************************************************************************************************************

# Model 2 for predicting temperature (GradientBoostingRegressor[LightGBM])

print("\n----- Training LightGBM Model for Temperature Prediction -----\n")

# Define your features and target variable
X = df_cleaned.drop(columns=['Temperature Sensor (°C)', 'Plant Stage'])  # Dropping target variable and unnecessary columns
y = df_cleaned['Temperature Sensor (°C)']  # Target variable (temperature)

# Define transformers for numerical and categorical data
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler())  # Standardize the data
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformations using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create a pipeline that first applies preprocessing and then trains a LightGBM model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor())
])

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculating RMSE manually
n = len(y_test)  # Number of samples
p = X_test.shape[1]  # Number of features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
explained_variance = explained_variance_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Create a DataFrame to display the metrics in a table
metrics = {
    'Metric': ['Mean Squared Error (MSE)', 'R² Score', 'Mean Absolute Error (MAE)', 
               'Root Mean Squared Error (RMSE)', 'Adjusted R²', 'Explained Variance Score', 
               'Max Error', 'Mean Absolute Percentage Error (MAPE)'],
    'Value': [mse, r2, mae, rmse, adj_r2, explained_variance, max_err, mape]
}

metrics_df = pd.DataFrame(metrics)

# Display the metrics table
print(metrics_df)

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Temperature')
plt.ylabel('Predicted Temperature')
plt.title('LightGBM: Actual vs Predicted Temperature')
plt.legend()
plt.show()

"""
Metrics from the Gradient Boosting Model:

Metric                                Value
0. Mean Squared Error (MSE)            0.874808
1. R² Score                            0.538873
2. Mean Absolute Error (MAE)           0.739495
3. Root Mean Squared Error (RMSE)      0.935312
4. Adjusted R²                         0.537529
5. Explained Variance Score            0.539305
6. Max Error                           3.372000
7. Mean Absolute Percentage Error (MAPE) 3.163982

"""

#********************************************************************************************************************************************************

"""
Evaluation:

Comparing the 2 models:

GBM outperformed GLM in most metrics, with an R² score of 0.62 compared to GLM's 0.48, indicating better explanatory power.
GBM also had a lower MSE (0.92 vs. 1.28) and RMSE (0.96 vs. 1.13), reflecting its superior prediction accuracy. 
Additionally, GBM demonstrated a lower MAE (0.75 vs. 0.89) and MAPE (3.20% vs. 3.81%), further confirming its better performance. 
However, GLM had a slightly lower max error (4.44 vs. 4.74), but this did not significantly affect the overall results.

Conclusion: GBM's superior accuracy, explained variance, and lower error metrics make it the preferred model for temperature prediction.

"""
#********************************************************************************************************************************************************

""""
Goal 2: Developing Models to Categorise Combined Plant Type-Stage Using Sensor Data

This task involves developing machine learning models to classify the combined "Plant Type-Stage" based on sensor data. 
By predicting plant types and stages, we can optimize resource allocation and improve farm management for better crop growth.

For this task, I will use two classification models: Logistic Regression and K-Nearest Neighbors (KNN).

Steps taken to build the models:
1. Data Preprocessing: Preparing the features and target variable for the models.
2. Model Training: Training the models using the prepared data.
3. Model Evaluation: Evaluating the models' performance using metrics such as accuracy, confusion matrix, and classification report.
4. Cross-Validation: Performing cross-validation to ensure the models' generalization to unseen data.

"""


#********************************************************************************************************************************************************
# **Goal 2: Classifying "Plant Type-Stage" using Logistic Regression and KNN**

print("\n----- Training Classification Models for Plant Type-Stage Prediction -----\n")

# Create combined 'Plant Type-Stage' column
df_cleaned['Plant Type-Stage'] = df_cleaned['Plant Type'].astype(str) + '-' + df_cleaned['Plant Stage'].astype(str)

# Define features (X) and target (y)
X = df_cleaned.drop(columns=['Plant Stage', 'Plant Type', 'Plant Type-Stage'])
y = df_cleaned['Plant Type-Stage']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)

# Fill missing values
X_encoded = X_encoded.fillna(X_encoded.median())

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#********************************************************************************************************************************************************

"""
Model 1 Logistic Regression::

Why I used Logistic Regression:

Logistic Regression:
- Simple and interpretable
- Fast training and prediction
- Can handle both binary and multi-class classification
- Provides probabilities for predictions
- Works well with large datasets

"""

#********************************************************************************************************************************************************
# **Logistic Regression Model**
print("\n----- Logistic Regression Model -----\n")

# Initialize and fit the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_logreg = log_reg.predict(X_test)

# Evaluate Logistic Regression model performance
print(f"Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))
print("\nClassification Report:\n", classification_report(y_test, y_pred_logreg))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Logistic Regression: Confusion Matrix')
plt.show()

# Perform cross-validation to evaluate model performance
cross_val_scores_logreg = cross_val_score(log_reg, X_scaled, y, cv=5, scoring='accuracy')
print(f"Logistic Regression Cross-validation Score: {cross_val_scores_logreg.mean():.4f}")

#********************************************************************************************************************************************************

"""
Accuracy: 0.6917

Confusion Matrix:
 [[690   1   1  86   0   0   0   0   0   0  31  23]
 [  2 600  47   6  47  56   1   4   4   0  27  12]
 [ 10 143 348   9  21  12   0   0   1   0  15   3]
 [  0   0   0 760   0   0   3   0   0   0  14  14]
 [  0  78  20   0 278 350   0   3   4   0   4   2]
 [  0  88  24   0 275 397   0   3   5   0   2   3]
 [  0   0   0   5   0   0 674   1   0  92   4   3]
 [  0   4   0   0   2   7   4 549 171   0   8   1]
 [  0   4   0   0   1   5   1 282 340   0   2   2]
 [  0   0   0   3   0   0 106   0   0 642   0   0]
 [ 51  27  20  15   2   1   9   8   1   6 481 110]
 [ 55  23  13   4   2   1   5   6   2   6 164 284]]

Classification Report:
               precision    recall  f1-score   support

         1-1       0.85      0.83      0.84       832
         1-2       0.62      0.74      0.68       806
         1-3       0.74      0.62      0.67       562
         2-1       0.86      0.96      0.91       791
         2-2       0.44      0.38      0.41       739
         2-3       0.48      0.50      0.49       797
         3-1       0.84      0.87      0.85       779
         3-2       0.64      0.74      0.69       746
         3-3       0.64      0.53      0.58       637
         4-1       0.86      0.85      0.86       751
         4-2       0.64      0.66      0.65       731
         4-3       0.62      0.50      0.56       565

    accuracy                           0.69      8736
   macro avg       0.69      0.68      0.68      8736
weighted avg       0.69      0.69      0.69      8736

Logistic Regression Cross-validation Score: 0.6880

"""
#********************************************************************************************************************************************************
"""""
Feature Importance:

Feature importance in logistic regression can be assessed by examining the feature coefficients.
These coefficients indicate how much each feature contributes to predicting the plant type-stage combinations. 
Larger coefficients, either positive or negative, indicate a greater influence on the model's decision. 
By extracting and visualizing the coefficients, we can identify which features are most influential in the model's predictions.


"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get the absolute value of the coefficients
feature_importance = abs(log_reg.coef_)

# Get the feature names from X_encoded (after one-hot encoding)
feature_names = X_encoded.columns

# Since the coef_ matrix has one row per class and one column per feature,
# we need to sum the feature importance across all classes to get an overall importance score
overall_importance = feature_importance.sum(axis=0)

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame(overall_importance, index=feature_names, columns=['Importance'])

# Sort features by their overall importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display top 3 most important and bottom 3 least important features
print("\n----- Overall Feature Importance -----\n")
print(f"Top 3 most important features:\n{importance_df.head(3)}")
print(f"\nTop 3 least important features:\n{importance_df.tail(3)}")

# Optionally, you can plot the feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x=importance_df.index, y=importance_df['Importance'], palette='Blues')
plt.title('Overall Feature Importance from Logistic Regression')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=90)
plt.show()

"""
Top 3 most important features:
                         Importance
Nutrient K Sensor (ppm)   45.922057
Temperature Sensor (°C)   28.305318
Humidity Sensor (%)       21.667525

Top 3 least important features:
                             Importance
Previous Cycle Plant Type      0.440236
System Location Code_zone_f    0.337730
System Location Code_zone_c    0.324147

These coefficients provide insights into the features that drive the model's predictions, helping identify key factors affecting plant growth stages.
"""
#********************************************************************************************************************************************************

"""
Model 2 K-Nearest Neighbors (KNN):

- Simple concept
- robust to outliers
- does not assume specific data distributions
- can handle non-linear relationships
- can handle both categorical and numerical features

"""

#********************************************************************************************************************************************************
# **K-Nearest Neighbors (KNN) Model**
print("\n----- K-Nearest Neighbors (KNN) Model -----\n")

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Evaluate KNN
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('KNN: Confusion Matrix')
plt.show()

# Cross-validation for KNN
cross_val_scores_knn = cross_val_score(knn_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"KNN Cross-validation Score: {cross_val_scores_knn.mean():.4f}")

sys.exit()

"""
----- K-Nearest Neighbors (KNN) Model -----

Accuracy: 0.6446

Confusion Matrix:
 [[689   0   0 130   0   0   5   0   0   2   4   2]
 [ 12 603  53   4  49  54   1   3   0   0  20   7]
 [ 20 190 289   7  23  18   0   0   0   0  13   2]
 [ 77   0   0 702   0   0   6   0   0   6   0   0]
 [  3  80  14   1 331 284   0   6   6   0  10   4]
 [  4  96  10   1 347 314   1   6   7   0   7   4]
 [  2   0   0   9   0   0 710   0   0  58   0   0]
 [  3   2   0   3   4   4  25 493 211   1   0   0]
 [  0   0   1   1   7   5  10 326 283   2   1   1]
 [  2   0   0  15   0   0 141   0   0 593   0   0]
 [118  19   5  19  10   5   8  20   9  20 407  91]
 [ 77  13  11  16   8  15   6  11   9  14 168 217]]

Classification Report:
               precision    recall  f1-score   support

         1-1       0.68      0.83      0.75       832
         1-2       0.60      0.75      0.67       806
         1-3       0.75      0.51      0.61       562
         2-1       0.77      0.89      0.83       791
         2-2       0.42      0.45      0.44       739
         2-3       0.45      0.39      0.42       797
         3-1       0.78      0.91      0.84       779
         3-2       0.57      0.66      0.61       746
         3-3       0.54      0.44      0.49       637
         4-1       0.85      0.79      0.82       751
         4-2       0.65      0.56      0.60       731
         4-3       0.66      0.38      0.49       565

    accuracy                           0.64      8736
   macro avg       0.64      0.63      0.63      8736
weighted avg       0.64      0.64      0.64      8736

KNN Cross-validation Score: 0.6498


"""
#********************************************************************************************************************************************************
"""
Comparing the 2 models:

Logistic Regression outperformed KNN with an accuracy of 68.92% compared to 63.82%, making it the more effective model for classifying plant type-stage combinations.
It also had a higher macro F1-score (0.67 vs. 0.61), indicating better balance across all classes. Logistic Regression was particularly strong in later growth stages, 
achieving an F1-score of 0.90 for Class 4-1, while KNN lagged behind at 0.85. The confusion matrix reveals that KNN struggled with misclassifications,
especially between adjacent stages (e.g., 197 samples from Class 1-3 were misclassified as 1-2), highlighting its sensitivity to overlapping data points. 
Logistic Regression’s ability to generalize better across classes and its efficiency make it the preferred model.

With all the information gathered from the models, we will now evaluate the all the infomation in the evaluation section in Read me file.
"""
#********************************************************************************************************************************************************



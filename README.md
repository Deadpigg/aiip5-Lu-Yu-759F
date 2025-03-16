# AIIP README:

**Name**: Lu Yu

**Email**: luyu8833@gmail.com

**NRIC**: T0271759F

## Overview

**README.md**: Provides an overview and instructions for running the pipeline.

**eda.ipynb**: Jupyter Notebook that contains the exploratory data analysis of the dataset, which helps understand the data structure and identify trends and patterns.

**model.py**: Python file containing the implementation of machine learning models, including training, evaluation, and selection of the best model.

**requirements.txt**: A file listing the dependencies needed to run the project, ensuring compatibility across environments.

## Execution Instructions

To execute the pipeline, follow these steps:

- Clone the repository to your own device.
- Ensure you have the required dependencies listed in requirements.txt installed
- Open eda.ipynb and run all to perform exploratory data analysis
- Open models.py and execute the code cells to build and evaluate machine learning models
- Modify parameters or model configurations as needed for experimentation

## Pipeline Flow

The pipeline consists of the following steps:

**Data Cleaning and Preprocessing**: 
-Cleaned the data by standardizing text columns, ensuring sensor values were non-negative, and handling missing values by replacing numerical NaNs with medians and categorical NaNs with "unknown.". Mapped categorical variables to numerical values for consistency and removed duplicates and outliers using the IQR method. 

**Exploratory Data Analysis (EDA)**: 
- The eda.ipynb file presents a detailed analysis of the dataset using visualizations like histograms, boxplots, and correlation matrices. It highlights key trends, such as the distribution of temperature conditions and the relationships between features.

**Key Findings for EDA**
- The EDA highlights different resource needs at each plant stage. Water and nutrients increase as plants grow, with nitrogen and potassium peaking in vegetative and maturity stages, and phosphorus being most important in the vegetative phase.
- Oxygen and CO₂ rise with growth, while pH remains stable for nutrient absorption.
- Light intensity increases with growth, and seedlings need more stable, slightly higher temperatures.
- Spatial analysis shows Zone D is best for seedlings, Zone C for vegetative growth, and Zone E for mature plants.

**Feature Processing**
The features in the dataset are processed as follows:

| Feature | Processing Steps |
| ----------- | ----------- |
| Categorical | One-hot encoding to convert to numerical form |
| numerical | Scaling using StandardScaler |
| Missing Values | Imputation with mean or mode |

**Model Building**:

For Task 1:
- Choice of models:
**GLM**: Easy to interpret, handles various data distributions, and suitable for both categorical and numerical features.
**GBM**: Ensemble learning approach with high performance and the ability to capture complex relationships among features.

For Task 2:
- Choice of models:
**KNN**: Simple concept, robust to outliers, and capable of handling non-linear relationships.
**Logistic regression**: simple, efficient model that provides probabilistic outputs, works well with linearly separable data, and can be regularized to prevent overfitting.


**Model Evaluation**: 

For Task 1:
The Gradient Boosting Model (GBM) outperforms the Generalized Linear Model (GLM) across all metrics, with lower MSE (0.8748 vs 1.0912), higher R² (0.5389 vs 0.4248), and better MAE (0.7395 vs 0.8417). It also has a lower RMSE (0.9353 vs 1.0446) and higher Adjusted R² (0.5375 vs 0.4232), showing better predictive accuracy and variance explanation. Overall, the GBM is more reliable and effective in capturing data patterns.

For Task 2:
Logistic Regression outperformed KNN with higher accuracy (68.92% vs. 63.82%) and a better macro F1-score (0.67 vs. 0.61). It excelled in later growth stages, with an F1-score of 0.90 for Class 4-1, while KNN scored 0.85. The confusion matrix showed KNN struggled with misclassifications, especially between adjacent stages. Overall, Logistic Regression is more effective and efficient.

## Conclusion
The pipeline provides a comprehensive approach to analyzing and modeling plant growth data. Through data cleaning, exploratory data analysis (EDA), and feature processing, we gain valuable insights into the resource needs at different plant stages and their optimal environmental conditions. The model building and evaluation steps demonstrate the effectiveness of machine learning models, with Gradient Boosting outperforming Generalized Linear Models (GLM) in Task 1 and Logistic Regression outperforming KNN in Task 2. These results highlight the importance of model selection and optimization for improving predictive accuracy and overall efficiency. The pipeline is designed to be flexible for experimentation, enabling further improvements and adaptations as needed.
  

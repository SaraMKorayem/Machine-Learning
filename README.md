Sara Mohamed Korayem
# **Machine Learning Stroke Prediction Using Random Forest**
![image](https://github.com/SaraMKorayem/Machine-Learning/assets/169392560/77e0ac16-72db-4fb8-b5ed-41e8727f34f1)


Stroke is the second deadliest disease globally and the third leading cause of disability. However, most stroke-related deaths can be prevented by recognizing its symptoms and implementing preventive measures through information technology(Eldora et al., 2024). This machine learning project focuses on predictive modeling for Stroke Prediction using two different models to compare and use the better approach. The two machine learning employed were Logistic Regression and Random Forest Classifier. Data preprocessing involved handling missing values and redundant columns. Exploratory Data Analysis (EDA) scrutinized data patterns and outliers. Feature engineering introduced new features to enhance predictive power.  Logistic Regression achieved an AUC accuracy of 88%, while Random Forest Classifier achieved 99%. Model evaluation included classification reports, confusion matrices, and feature importance analysis. The final models were fine-tuned and are ready for deployment, potentially providing valuable insights into stroke prediction in clinical settings.

1: Data Exploration and Preprocessing
Dataset Characteristics
- The dataset consists of health-related features such as age, gender, BMI, smoking status, residence type, work type, ever married status, average glucose level, and stroke occurrence.
- The dataset has been sourced from Kaggle's Stroke Prediction Dataset and link is provided in code (.ipynb) file.

Data Cleaning
- Identified and handled missing values in the 'bmi' column.
- Dropped the 'id' column as it was not needed for further analysis.
- No duplicate rows were found in the dataset.


Exploratory Data Analysis (EDA)
- Visualized missing values using the missingno library.
- Explored the distributions of categorical variables like gender, residence type, smoking status, ever married status, and work type.
- Checked for outliers and inconsistencies.

2: Feature Engineering

- Added new features 'age_squared' (squared age) and 'age_bmi_interaction' (age multiplied by BMI) to enhance predictive power.

3: Machine Learning Model Development

Selected Model: Logistic Regression
- Split the dataset into training and testing sets.
- Utilized SMOTE for oversampling due to class imbalance.
- Trained a logistic regression model and evaluated its performance.
- Achieved an accuracy of AUC 88% on the test data.

Model Evaluation Metrics
- Used classification report and confusion matrix to evaluate model performance.
- Explored feature importance using coefficients from the logistic regression model.

Second Model: Random Forest Classifier
- Trained a random forest classifier and evaluated its performance.
- Achieved an accuracy of [insert accuracy percentage here] on the test data.
- Calculated the AUC score to assess model performance.
- AUC was 99%.

4: Model Evaluation and Fine-tuning

Evaluation Metrics
- Compared the performance of logistic regression and random forest classifiers.
- Cross-validated the models to assess generalization performance.

  
5: Model Deployment
-Used the random forest model, which had an AUC of 99% this was around10% higher than logistic regression model.
Streamlit was used for the interface.
- Created a new (.py) file for the Random Forest model alone.


Limitations and Future Improvements
- Limited dataset size and class imbalance may have affected model performance.
- Further exploration could be helpful, more of feature selection techniques, and trying other classification algorithms like Support Vector Machines (SVM) or Gradient Boosting Machines (GBM).

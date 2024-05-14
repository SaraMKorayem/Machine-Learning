
import pandas as pd
import sklearn as sk

# from google.colab import files
# uploaded = files.upload()

#df = pd.read_csv("E:\Junior year uni\semester 2\ML & DM\assignment\healthcare-dataset-stroke-data.csv")


df = pd.read_csv(r"E:\Junior year uni\semester 2\ML & DM\assignment\healthcare-dataset-stroke-data.csv")

# Your data processing code continues...


df.head()

df = df.drop(['id'], axis =1)

df.isnull()

df = df.dropna()

df['Residence_type'] = df['Residence_type'].replace(['Rural', 'Urban'], [0, 1])

df['gender'] = df['gender'].replace(['Female', 'Male'], [0, 1])

df.info()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

#will drop this other to have accurate results for my model
df = df[df['gender'] != 'Other']
sum_other_values = (df['gender'] == 'Other').sum()
sum_other_values

df['work_type'] = le.fit_transform(df['work_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])
df['ever_married'] = le.fit_transform(df['ever_married'])

df['gender'] = df['gender'].replace(['Female', 'Male'], [0, 1]).astype(int)

# Assume 'df' is your DataFrame
# Create a new feature 'age_squared' by squaring the 'age' column
df['age_squared'] = df['age'] ** 2

# Create an interaction feature 'age_bmi_interaction' by multiplying 'age' and 'bmi'
df['age_bmi_interaction'] = df['age'] * df['bmi']

columns_to_exclude = ['stroke']
new_df = df.drop(columns=columns_to_exclude)

x = new_df
print("X's shape: ", x.shape)

y = df['stroke']
print("Y's shape: ", y.shape)

from imblearn.over_sampling import SMOTE

os = SMOTE(random_state = 42)
x_os, y_os = os.fit_resample(x, y)

df_os = pd.DataFrame(x_os)
df_os['stroke'] = y_os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, Y_train, y_test = train_test_split(x_os, y_os, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

#Train the model on the training data
rf_model.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test)


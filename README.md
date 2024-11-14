# Customer-Churn-Prediction-Model


### Project Overview

The objective of this project is to build a machine learning model to predict customer churn for a telecom company. The goal is to create a model that predicts whether a customer is likely to churn (leave the service), and also, deploy the model on Heroku, allowing for real-time predictions.
 
### Data Source
The dataset used for this project was gotten from Kaggle. It is a telecom company's customer data. including features like gender, tenure, internet service, payment method, and churn status. 

### Tools

- Excel
- Python libraries Pandas, Scikit-learn, Joblib, Flask, Gunicorn, etc.
- Jupyter Notebook


### Steps

The project includes several  data preprocessing, model training, and deploying. the model on Heroku, making it accessible online.


#### Data Exploration
The initial step was to install and import all the required python libraries/modules that are to be used in the course of the analysis.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
```


```python
df.info() # details of the columns
df.head() # display of the first 5 rows across the columns
```

In a bid to know the number of Customers that would likely leave the service (churn) and those that would not.
```python
df['Churn'].value_counts()
```
Having run this code, a total of 5174 are not likely to churn while 1869 customers are likely to churn the telecommunication services.


The charts below visualize the count of customers across the columns, displaying the number of customers that are males and females; those that are senior citizens and those that are not; those that have partner and those that don't have; and those that have dependents and those that don't have.
```python
fig, axes = plt.subplots(1, 4, figsize = (12, 4))

sns.countplot(x = 'gender', data = df, ax = axes[0])
axes[0].set_title('gender')

sns.countplot(x = 'SeniorCitizen', data = df, ax = axes[1])
axes[1].set_title('SeniorCitizen')

sns.countplot(x = 'Partner', data = df, ax = axes[2])
axes[2].set_title('Partner')

sns.countplot(x = 'Dependents', data = df, ax = axes[3])
axes[3].set_title('Dependents')

plt.tight_layout()
```



A boxplot is used to visualize the Monthly Charges and the churn status of the customers. This shows that
```python
sns.boxplot(data = df, x = "Churn", y = "MonthlyCharges")
plt.show()
```

This reveals the churn status of customers across columns like Internet Service, Online Security, Device Protection and Tech Support. This will help investigate if a lag in any of these services contribute to a customer churning.
```python
fig, axes = plt.subplots(1, 4, figsize = (12, 4))

sns.countplot(x = 'Churn', hue = 'InternetService', data = df, ax = axes[0])
axes[0].set_title('InternetService')

sns.countplot(x = 'Churn', hue = 'OnlineSecurity', data = df, ax = axes[1])
axes[1].set_title('OnlineSecurity')

sns.countplot(x = 'Churn', hue = 'DeviceProtection', data = df, ax = axes[2])
axes[2].set_title('DeviceProtection')

sns.countplot(x = 'Churn', hue = 'TechSupport', data = df, ax = axes[3])
axes[3].set_title('TechSupport')

plt.tight_layout()
```

#### Data Preprocessing

It was observed that TotalCharges column which ought to be in float data type carries object datatype. So, there is a need to convert the data type to float. 

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
```

```python
# importing essential libraries 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
```

Here, we need to encode the categorical values, replacing them with 0-2. In a bid to achieve this we will have to first drop the non-categorical columns.


```python
# dropping the non-categorical columns in the dataset
dfa = df.drop(['customerID', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'], axis=1)
dfa.head()
```

```python
# encoding the categorical variables 
le = preprocessing.LabelEncoder()
dfb = dfa.apply(le.fit_transform)
dfb.head()
```

Having encode them, then, we have to merge both non-categorical and categorical columns together before taking further steps
```python
dfc = df[['customerID', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
new_df = pd.merge(dfc, dfb, left_index = True, right_index = True)
```


The next step now is oversampling which is 
This is to avoid a bias in our result, having observed that the count of the churn status of the customers are quite far apart (Yes = 1869; No = 5174).

```python
# Before oversampling, let's do a train-test split
from sklearn.model_selection import train_test_split
new_df = new_df.dropna()
new_df = new_df.drop(['customerID'], axis=1) # dropping CustomerID column because it is irrelevant here

X = new_df.drop(['Churn'], axis=1) # dropping the Churn column from the dataset to create X(feature; independent variable) for model training
y = new_df['Churn'] # assigning the Churn column to y(target; dependent variable) that is to be predicted

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
``` 

```python
# This is as a result of the imbalance in the proportion of the customers that churn and the ones that didn't
from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=5)
X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_resampled, y_resampled # This replaces the original training data with the oversampled data
y_train.value_counts()
```

#### Model Building and Evaluation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
```

```python
# Model building
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
```

```python
y_pred = clf.predict(X_test) # Make predictions
```

```python
# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

y_prob = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)

print(f'Accuracy: {accuracy:.2f}\n'
     f'Precision: {precision:.2f}\n'
     f'Recall: {recall:.2f}\n'
     f'F1 Score: {f1:.2f}\n'
     f'ROC AUC: {roc_auc:.2f}')
```


#### Model Deployment
```python
# Saving the model
import joblib
joblib.dump(clf, 'clf.pk1')
```

```python
pip install Flask gunicorn
```

```python
#Creating a New Directory
import os
directory_name = 'Model Deployment'

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    print(f'Directory "{directory_name}" created.')
else:
    print(f'Directory "{directory_name}" already exists.')
```

```python
#Creating a virtual environment
!python -m venv "Model Deployment"
```

```python
from flask import Flask, request, jsonify
```

```python
#Loading the model
model = joblib.load('clf.pk1')
```

```python
# Defining a route to handle the predictions
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data'] #Get the data from the request
    
    prediction = model.predict([data])
    return jsonify({'prediction': prediction.tolist()})
```


#### Insights

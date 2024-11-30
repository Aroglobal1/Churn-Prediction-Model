# Customer-Churn-Prediction-Model


### Project Overview

The objective of this project is to build a machine learning model to predict customer churn for a telecom company. The goal is to create a model that predicts whether a customer is likely to churn (leave the service) and deploy it on Heroku, allowing for real-time predictions.
 
### Data Source
The dataset used for this project was sourced from Kaggle. It consists of a telecom company's customer data, including features like gender, tenure, internet service, payment method, and churn status.

### Tools

- Excel
- Python libraries Pandas, Scikit-learn, Joblib, Flask, Gunicorn, etc.
- Jupyter Notebook


### Steps

The project includes several steps: data preprocessing, model training, and deployment on Heroku, making the model accessible online.


#### Data Exploration
The first step was to install and import all the required Python libraries/modules for the analysis.

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
![1a](https://github.com/user-attachments/assets/592aa256-707e-4388-bbe3-5864c94a36b2)


This shows the number of Customers that have churned (left the service) versus those who have not.

```python
df['Churn'].value_counts()
```
![2a](https://github.com/user-attachments/assets/548d57c5-9c03-4a3d-a8a0-9d5c26cb0a08)

This shows that about 1869 (27%) of the customers have churned. This indicates that there is an imbalanced classification problem due to the large difference in the number of churned and non-churned customers. 


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
![1](https://github.com/user-attachments/assets/500ebe31-c40b-4545-bfeb-65309b1bf7e4)

From the above charts, there is an equal distribution of both genders. Also, there are equal proportions of customers with and without partners. The majority of customers are not senior citizens, and most of them do not have dependents.


Next, I analyzed the relationship between monthly charges and churn status, as customers might churn if the cost of their monthly charges is too high.

```python
sns.boxplot(data = df, x = "Churn", y = "MonthlyCharges")
plt.show()
```
![2](https://github.com/user-attachments/assets/abc41357-18af-4a5c-8988-44dd2070a39f)

The boxplot shows that the median of those who churned is higher, supporting the assumption that customers tend to churn when monthly charges are high.


I also explored the relationship between customer's churn status and a few other categorical variables, such as Internet Service, Online Security, Device Protection and Tech Support, to investigate if gaps in these services contribute to customer churn.
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
![3](https://github.com/user-attachments/assets/0dd15908-4071-475f-8fe4-98e51296a24f)

- **Internet Service:** Customers with Fiber optic service churned more, possibly due to its higher cost or poor network coverage.
- **Online Security:** More customers who churned did not opt for online security, possibly leaving them vulnerable to cyberattacks.
- **Device Protection:** Customers without device protection churned the most, which could be due to the service's cost or lack of awareness.
- **Tech Support:** A larger percentage of customers who churned did not subscribe to tech support, which might indicate dissatisfaction with the service.


#### Data Preprocessing

It was observed that the TotalCharges column, which should be a float, is incorrectly classified as an object. Therefore, we need to convert the data type to float.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')
```

Next, we import the necessary libraries for data preprocessing.
```python
# importing essential libraries 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
```

We need to encode the categorical variables using Scikit-learn's LabelEncoder to convert them into numeric format, making them compatible with the machine learning model. First, we drop non-categorical variables (columns).


```python
# dropping the non-categorical columns in the dataset
dfa = df.drop(['customerID', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'], axis=1)
dfa.head()
```
![4](https://github.com/user-attachments/assets/2de42982-0ed1-4285-a803-fe7e7adbb7f9)


```python
# encoding the categorical variables 
le = preprocessing.LabelEncoder()
dfb = dfa.apply(le.fit_transform)
dfb.head()
```
![5](https://github.com/user-attachments/assets/9263b831-74f2-4c78-8923-4b43bc597b68)


After encoding, what follows is to merge both non-categorical and categorical variables.
```python
dfc = df[['customerID', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']]
new_df = pd.merge(dfc, dfb, left_index = True, right_index = True)
```


#### Oversampling
Since only 27% of the customers have churned, the dataset is imbalanced, which could lead to an underperforming machine learning model. To address this, we use oversampling to balance the class distribution by increasing the number of samples in the minority class (churned customers).

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
from imblearn.over_sampling import SMOTE
oversample = SMOTE(k_neighbors=5)
X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_resampled, y_resampled # This replaces the original training data with the oversampled data
y_train.value_counts()
```
![6](https://github.com/user-attachments/assets/1db56482-6e68-401b-8f83-267299370390)


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

After training the model, it was evaluated using performance metrics like accuracy, precision, recall, F1 score and ROC AUC.
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
![7](https://github.com/user-attachments/assets/c36cc1d1-3d60-4fee-bfd6-33b2e52528b8)

This shows that the model correctly classifies 

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
It was observed that 27% of the customers have churned, highlighting a need for immediate action to investigate the factors driving this churn. This indicates a potential issue with customer retention, which could negatively impact the company's revenue, reputation and customer's loyalty.

The analysis also revealed a relationship between customer churn and high monthly charges. This suggests that price sensitivity is a major factor contributing to churn. To address this, the company may need to review its pricing structure or introduce discounts and incentives to retain more customers.


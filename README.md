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


#### Model Training and Evaluation


#### Deployment


#### Insights

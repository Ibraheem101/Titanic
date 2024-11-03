#!/usr/bin/env python
# coding: utf-8

# Load data
# Import libraries

import os
import boto3
import numpy as np
import pandas as pd
import seaborn as sns

# Load Data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

# Exploratory Data Analysis

# The 'Cabin' column has a lot of Nan values, so we won't be using it as a feature.

# Dropping the cabin ane Name column
train_df = train_data.drop(['Cabin', 'Name'], axis = 1)

# Set index to PassengerID
train_df.set_index(['PassengerId'], inplace = True)

# Peplace rows with missing age values with median
#'S' is the most common element in the 'Embarked' column, we'll replace missing values in that column with 'S'
 
train_df['Age'] = train_df['Age'].replace(np.nan, train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].replace(np.nan, 'S')


# We have 891 entries which we will use for training.
# <p>We'll select our features

# #### One-hot encoding the 'Sex' and 'Embarked' columns
train_onehot = pd.get_dummies(train_df[['Sex', 'Embarked']], prefix="", prefix_sep="")

# New train dataframe
new_train_df = pd.merge(left = train_df, right = train_onehot, left_index=True, right_on='PassengerId')
new_train_df.drop('Sex', axis = 1, inplace = True)

# Rename columns
new_train_df = new_train_df.rename(columns = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'})

# Selecting features
features = ["Pclass", "Age", "SibSp", "Parch", "female", "male", "Cherbourg", "Queenstown", "Southampton"]
X_train = new_train_df[features]
y_train = new_train_df['Survived']

#Scaling the features
from sklearn.preprocessing import StandardScaler
standard_scale = StandardScaler()
scaled_feature = standard_scale.fit_transform(X_train)
X_train = pd.DataFrame(scaled_feature, columns = features)

# ### TEST DATA

### The age column has Nan values

test_data["Age"] = test_data["Age"].replace(np.nan, train_df['Age'].median())

# One-hot encoding
test_onehot = pd.get_dummies(test_data[['Sex', 'Embarked']], prefix="", prefix_sep="")

# New test dataframe
new_test_df = pd.merge(left = test_data, right = test_onehot, left_index=True, right_on=test_data.index)
new_test_df.drop('Sex', axis = 1, inplace = True)
new_test_df.drop(columns = ['key_0'], axis=1, inplace=True)

# Rename columns
new_test_df.rename(columns = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}, inplace = True)
new_test_df.set_index('PassengerId', inplace = True)

X_test = new_test_df[features]


scaled_test_features = standard_scale.transform(X_test)
X_test_scaled = pd.DataFrame(scaled_test_features, columns=features)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

import pickle
filename = 'titanic_model.pkl'

with open(filename, 'wb') as file:  
    pickle.dump(LR, file)

X_test.dropna(axis = 0, inplace = True)
yhat = LR.predict(X_test_scaled)
yhat_prob = LR.predict_proba(X_test)

submission = pd.DataFrame(X_test.index.values, columns = ["PassengerId"])
submission["Survived"] = yhat
submission.to_csv('my_submission.csv',index=False)
print("Your submission was successfully saved!")

# Initialize an S3 client
s3 = boto3.client('s3')

# Set bucket and model path
bucket_name = 'my-titanic-project-bucket'
model_file_name = filename
local_model_path = f'{model_file_name}'

# Upload model to S3
try:
    s3.upload_file(local_model_path, bucket_name, f'models/{model_file_name}')
    print(f"Model successfully uploaded to s3://{bucket_name}/models/{model_file_name}")
except Exception as e:
    print(f"Failed to upload model to S3: {e}")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Load your dataset
df = pd.read_csv('/kaggle/input/combined-dataset/combined_data.csv')  # Adjust the dataset path as needed

# Vectorize the ciphertext column
vectorizer = TfidfVectorizer(max_features=100)  # Limit to top 100 features
ciphertext_features = vectorizer.fit_transform(df['ciphertext']).toarray()

# Convert the ciphertext features into a DataFrame with column names
ciphertext_feature_df = pd.DataFrame(ciphertext_features, columns=[f"ciphertext_feature_{i}" for i in range(ciphertext_features.shape[1])])

# Concatenate with other numeric features
X = pd.concat([df[['key_length', 'block_size', 'ciphertext_length']], ciphertext_feature_df], axis=1)   
y = df['algorithm']

# Encode the target variable (y) using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the features for models that require scaling (Logistic Regression, SVM, etc.)
scaler = StandardScaler()

# Ensure that X contains only numeric columns, and column names are strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------ Handle Missing Values ------

# Impute missing values in the scaled training and test data
imputer = SimpleImputer(strategy='median')
X_train_scaled_imputed = imputer.fit_transform(X_train_scaled)
X_test_scaled_imputed = imputer.transform(X_test_scaled)

# ------ XGBoost Model with Regularization ------

# Set up the XGBoost model with regularization
xgb_model = XGBClassifier(
    objective='multi:softmax', 
    num_class=len(label_encoder.classes_),  # Set the number of classes to match the encoded labels
    max_depth=4,  # Limit the depth of trees
    learning_rate=0.01,  # Lower learning rate for more stable convergence
    n_estimators=100,  # Number of boosting rounds
    min_child_weight=1,  # Regularization parameter for leaf nodes
    subsample=0.8,  # Randomly sample 80% of the data for each tree
    colsample_bytree=0.8,  # Randomly sample features for each tree
    gamma=1,  # Minimum loss reduction for further partitioning
    reg_alpha=0.01,  # L1 regularization
    reg_lambda=0.1   # L2 regularization
)

# Train the XGBoost model with early stopping
eval_set = [(X_test_scaled_imputed, y_test)]
xgb_model.fit(
    X_train_scaled_imputed, y_train, 
    eval_set=eval_set, 
    early_stopping_rounds=10, 
    eval_metric='mlogloss', 
    verbose=True
)

# Predict with the model
y_pred_xgb = xgb_model.predict(X_test_scaled_imputed)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Test Accuracy: {xgb_accuracy:.4f}")

# ------ Logistic Regression Model ------

# Set up Logistic Regression with regularization
log_reg_model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', C=0.1)

# Train the Logistic Regression model
log_reg_model.fit(X_train_scaled_imputed, y_train)

# Predict with the model
y_pred_log_reg = log_reg_model.predict(X_test_scaled_imputed)
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Test Accuracy: {log_reg_accuracy:.4f}")

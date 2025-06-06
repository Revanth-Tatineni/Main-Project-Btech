import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# Standardize the features for Random Forest (optional but improves consistency)
scaler = StandardScaler()

# Ensure that X contains only numeric columns, and column names are strings
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle Missing Values
imputer = SimpleImputer(strategy='median')
X_train_scaled_imputed = imputer.fit_transform(X_train_scaled)
X_test_scaled_imputed = imputer.transform(X_test_scaled)

# ------ Random Forest Model ------

# Set up the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of trees
    random_state=42,       # For reproducibility
    class_weight='balanced'  # Handles imbalanced classes
)

# Train the Random Forest model
rf_model.fit(X_train_scaled_imputed, y_train)

# Predict with the Random Forest model
y_pred_rf = rf_model.predict(X_test_scaled_imputed)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")


# Classification report for additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=label_encoder.classes_))

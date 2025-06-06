import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

# Convert the target variable to one-hot encoding for deep learning
y_one_hot = to_categorical(y_encoded)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Standardize the features for neural networks
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

# ------ Deep Learning Model ------

# Set up a simple feedforward neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled_imputed.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train_scaled_imputed, y_train,
    validation_data=(X_test_scaled_imputed, y_test),
    epochs=50,  # Adjust epochs as needed
    batch_size=32,  # Adjust batch size as needed
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled_imputed, y_test, verbose=0)
print(f"Deep Learning Test Accuracy: {test_accuracy:.4f}")

# Predict with the model
y_pred_dl = model.predict(X_test_scaled_imputed)
y_pred_classes = np.argmax(y_pred_dl, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
dl_accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Deep Learning Test Accuracy (from predictions): {dl_accuracy:.4f}")

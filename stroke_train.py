# stroke_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle
import os

# Load dataset
df = pd.read_csv('dataset/brain_stroke.csv')
df = df.dropna()

# Encode categorical variables
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['work_type'] = le.fit_transform(df['work_type'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])

# Features and Target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Check original class distribution
print("Original Class Distribution:\n", y.value_counts())

# Split data BEFORE applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to training data only (SMOTE (Synthetic Minority Oversampling Technique)  )
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nBalanced Class Distribution (After SMOTE):\n", y_train_smote.value_counts())

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

# Evaluate the model
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
os.makedirs('model', exist_ok=True)
with open('model/stroke_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    print("\n✅ Model trained and saved as 'model/stroke_model.pkl'")

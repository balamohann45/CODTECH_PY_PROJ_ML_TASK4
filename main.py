import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv('student_data.csv', encoding='latin1')

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('passed', axis=1)
y = df['passed']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Create side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Overall Passed vs Not Passed
sns.countplot(x='passed', data=df, ax=axs[0])
axs[0].set_title("Overall Distribution: Passed vs Not Passed")
axs[0].set_xlabel("Passed (1 = Yes, 0 = No)")
axs[0].set_ylabel("Count")

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[1])
axs[1].set_title("Confusion Matrix (Test Set)")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

# Show combined plot
plt.tight_layout()
plt.show()





    

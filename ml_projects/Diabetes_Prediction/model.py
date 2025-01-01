# Importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Load the dataset
dataset_path = 'diabetes_prediction_dataset.csv'  # Ensure the file exists in the working directory
dataset = pd.read_csv(dataset_path)

# Encode categorical features
label_encoders = {}
for column in ['gender', 'smoking_history']:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

# Feature matrix and target vector
X = dataset[["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]]
Y = dataset["diabetes"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=30)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model
Y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)

# Extract confusion matrix values
TN, FP, FN, TP = cm.ravel()

# Calculate evaluation metrics
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)  # Sensitivity
f1 = f1_score(Y_test, Y_pred)
specificity = TN / (TN + FP)
npv = TN / (TN + FN)
fpr = FP / (FP + TN)
fnr = FN / (TP + FN)
fdr = FP / (FP + TP)
mcc = matthews_corrcoef(Y_test, Y_pred)

# Print evaluation metrics
print("Confusion Matrix:")
print(cm)
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"NPV: {npv:.2f}")
print(f"FPR: {fpr:.2f}")
print(f"FNR: {fnr:.2f}")
print(f"FDR: {fdr:.2f}")
print(f"MCC: {mcc:.2f}")

# Save the model and preprocessors
import pickle
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))

# Visualizations
plt.figure(figsize=(15, 8))

# Correlation Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')

# Age vs. Diabetes Distribution
plt.subplot(2, 2, 2)
sns.histplot(data=dataset, x="age", hue="diabetes", kde=True, bins=30, palette='Set2')
plt.title('Age vs Diabetes')

# BMI vs Diabetes
plt.subplot(2, 2, 3)
sns.boxplot(data=dataset, x="diabetes", y="bmi", palette="pastel")
plt.title('BMI Distribution by Diabetes')

# Smoking History vs Diabetes
plt.subplot(2, 2, 4)
sns.countplot(data=dataset, x="smoking_history", hue="diabetes", palette="viridis")
plt.title('Smoking History by Diabetes')

plt.tight_layout()
plt.show()

# Visualize Results: FDR, FNR, NPV, and FPR
metrics = {
    "FDR": fdr,
    "FNR": fnr,
    "NPV": npv,
    "FPR": fpr
}

plt.figure(figsize=(8, 6))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red'])
plt.title('Results for FDR, FNR, NPV, and FPR')
plt.ylabel('Values')
plt.ylim(0, 1)  # Since these metrics are ratios, the range is 0 to 1
plt.xlabel('Metrics')
for index, value in enumerate(metrics.values()):
    plt.text(index, value + 0.02, f"{value:.2f}", ha='center', fontsize=12)
plt.show()

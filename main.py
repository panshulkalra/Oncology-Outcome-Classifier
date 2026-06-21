import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

try:
    df = pd.read_csv('dataset.csv')
    print("Clinical dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset not found. Please check the file name.")
    exit()

# Drop corrupted column if present
if 'Unnamed: 3' in df.columns:
    df = df.drop(columns=['Unnamed: 3'])

# Feature Engineering: The Lymph Node Ratio (LNR)
df['Lymph_Node_Ratio'] = df['Reginol Node Positive'] / (df['Regional Node Examined'] + 1)

# Target Variable: 1 for Alive, 0 for Dead
y = df['Status'].apply(lambda x: 1 if x == 'Alive' else 0)

# Features: Drop Status and Survival Months (Preventing Data Leakage)
X = df.drop(columns=['Status', 'Survival Months'])

# 2. PEARSON CORRELATION (BIOLOGICAL MAP)
print("\nGenerating Pearson Correlation Matrix...")
numerical_cols = ['Age', 'Tumor Size', 'Regional Node Examined', 'Reginol Node Positive', 'Lymph_Node_Ratio']
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt=".2f", linewidths=1)
plt.title('Pearson Correlation: Biological Marker Map', fontsize=15)
plt.savefig('correlation_map.png')
print("Correlation map saved as 'correlation_map.png'")


# 3. DATA PREPROCESSING
X_encoded = pd.get_dummies(X, drop_first=True)

# Regex Fix for LightGBM (Removing brackets, spaces, commas from column names)
X_encoded = X_encoded.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

# Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Standard Scaling for distance-based models
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_encoded.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_encoded.columns)


models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', random_state=42),
    "FFNN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
}

print("\n" + "="*50)
print("ALGORITHM ACCURACY REPORT")
print("="*50)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<30} | {acc * 100:.2f}%")


# 5. THE CLINICAL DEEP DIVE (VISUALS & METRICS)
print("\n" + "="*50)
print("XGBoost")
print("="*50)

champion_model = models["XGBoost"]
y_pred_champ = champion_model.predict(X_test_scaled)

# Feature Importance Bar Chart
champ_importances = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': champion_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=champ_importances.head(15), palette='magma')
plt.title('Top 15 Predictors of Mortality', fontsize=16)
plt.xlabel('Importance Score')
plt.ylabel('Biological Marker')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance chart saved as 'feature_importance.png'")

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred_champ)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Dead', 'Alive'], yticklabels=['Dead', 'Alive'])
plt.title('Clinical Confusion Matrix: Prediction Errors', fontsize=15)
plt.xlabel('Model Prediction')
plt.ylabel('Actual Patient Outcome')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Medical Classification Report
print("\nCLASSIFICATION REPORT")
print("-" * 50)
print(classification_report(y_test, y_pred_champ, target_names=['Dead', 'Alive']))
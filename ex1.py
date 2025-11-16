
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


sns.set_style("whitegrid")


FILE_NAME = 'cardio_data.csv'
try:
    df = pd.read_csv(FILE_NAME)
except FileNotFoundError:
    print(f"Error: {FILE_NAME} not found. Please check the file name.")
    exit()


if 'id' in df.columns:
    df = df.drop('id', axis=1)

X = df.drop('cardio', axis=1)
y = df['cardio']


X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"Data split and scaled. Training set size: {X_train.shape[0]} samples")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


models = {
    "Logistic Regression (LR)": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbor (KNN)": KNeighborsClassifier(),
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "Decision Tree (DT)": DecisionTreeClassifier(random_state=42),
    "Random Forest (RF)": RandomForestClassifier(random_state=42)
}

results = []
print("\n--- Model Training and Accuracy Comparison ---")

for name, model in models.items():
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)


    results.append({
        'Model': name,
        'Accuracy': accuracy
    })

accuracy_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)

print("\nAccuracy Comparison Table:")

print(accuracy_df.to_markdown(index=False, floatfmt=".4f"))

# Select
best_model_name = accuracy_df.iloc[0]['Model']
best_model_accuracy = accuracy_df.iloc[0]['Accuracy']

print(
    f"\n✅ The final model selected is **{best_model_name}** with an accuracy of **{best_model_accuracy:.4f}**. (This fulfills Step 5)")
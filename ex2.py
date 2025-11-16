

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


models = {
    "Logistic Regression (LR)": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbor (KNN)": KNeighborsClassifier(),
    "Support Vector Machine (SVM)": SVC(random_state=42),
    "Decision Tree (DT)": DecisionTreeClassifier(random_state=42),
    "Random Forest (RF)": RandomForestClassifier(random_state=42)
}

results = []

print("\n--- PHASE 2: Model Training and Evaluation ---")

for name, model in models.items():
    print(f"\nTraining {name}...")

   
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)

   
    print(f"Accuracy: {accuracy:.4f}")

   

    results.append({
        'Model': name,
        'Accuracy': accuracy
    })


print("\n--- Comparison of Model Accuracies (Step 4 Output) ---")


accuracy_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)


print(accuracy_df.to_markdown(index=False))


best_model_row = accuracy_df.iloc[0]
best_model_name = best_model_row['Model']
best_model_accuracy = best_model_row['Accuracy']

print(
    f"\nDecision for Step 5: The best performing model is **{best_model_name}** with an accuracy of **{best_model_accuracy:.4f}**.")
print("This model would be selected as the final Machine Learning model for heart disease detection.")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the preprocessed dataset
dataframe = pd.read_csv("preprocessed_data.csv")

# Reconstruct original demand class
def class_reconstruction(row):
    if row['demand_class_High'] == 1:
        return 'High'
    elif row['demand_class_Medium'] == 1:
        return 'Medium'
    else:
        return 'Low'

y = dataframe.apply(class_reconstruction, axis=1)
X = dataframe.drop(columns=['demand_class_Medium', 'demand_class_High'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define parameter grid
parameter_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

# GridSearchCV setup
print("Grid search started")

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=parameter_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

labels = ['Low', 'Medium', 'High']

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, labels=labels, target_names=labels, digits=3)
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print("Best Parameters Found:")
print(grid_search.best_params_)
print(f"\nModel Accuracy {accuracy:.4f}")
print("\nClassification Report:\n")
print(report)

# Confusion Matrix
display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
display.plot(cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.grid(False)
plt.tight_layout()
plt.show()
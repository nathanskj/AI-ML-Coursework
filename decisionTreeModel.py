import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# Train Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

y_tree_pred = decision_tree_model.predict(X_test)

# Evaluation
labels = ['Low', 'Medium', 'High']
tree_accuracy = accuracy_score(y_test, y_tree_pred)
tree_report = classification_report(y_test, y_tree_pred, labels=labels, target_names=labels, digits=3)
tree_matrix = confusion_matrix(y_test, y_tree_pred, labels=labels)

print(f"Decision Tree Accuracy: {tree_accuracy:.4f}")
print("\nClassification Report:\n")
print(tree_report)

# Confusion Matrix Plot
ConfusionMatrixDisplay(confusion_matrix=tree_matrix, display_labels=labels).plot(cmap='Oranges')
plt.title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.show()

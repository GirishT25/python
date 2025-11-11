# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Step 1: Load the dataset
# -------------------------------
# Replace the path with the downloaded CSV file location
dataset = pd.read_csv("iris.csv")  

print("📄 Dataset Preview:\n", dataset.head(), "\n")

# -------------------------------
# Step 2: Split into features and target
# -------------------------------
X = dataset.iloc[:, :-1]  # All columns except last (features)
y = dataset.iloc[:, -1]   # Last column (target)

# -------------------------------
# Step 3: Split into training and testing sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Step 4: Train Naive Bayes Classifier
# -------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -------------------------------
# Step 5: Make Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 6: Evaluate Model
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.2f}\n")
print("📊 Confusion Matrix:\n", conf_matrix, "\n")
print("📝 Classification Report:\n", report)

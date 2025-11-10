# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create the dataset
# -------------------------------
data = {
    'Age': ['<21', '<21', '21-35', '>35', '>35', '>35', '21-35', '<21', '<21', '>35'],
    'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium'],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Female', 'Female', 'Female', 'Female'],
    'MaritalStatus': ['Single', 'Married', 'Single', 'Single', 'Single', 'Married', 'Married', 'Single', 'Married', 'Married'],
    'Buys': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df, "\n")

# -------------------------------
# Step 2: Encode categorical data
# -------------------------------
le = LabelEncoder()
encoded_df = df.copy()

for column in encoded_df.columns:
    encoded_df[column] = le.fit_transform(encoded_df[column])

print("Encoded Dataset:\n", encoded_df, "\n")

# -------------------------------
# Step 3: Split into features and target
# -------------------------------
X = encoded_df.drop('Buys', axis=1)
y = encoded_df['Buys']

# -------------------------------
# Step 4: Train the Decision Tree
# -------------------------------
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)

# -------------------------------
# Step 5: Visualize the Decision Tree
# -------------------------------
plt.figure(figsize=(12,8))
plot_tree(
    model,
    feature_names=['Age', 'Income', 'Gender', 'MaritalStatus'],
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Customer Buying Behavior")
plt.show()

# -------------------------------
# Step 6: Test data prediction
# -------------------------------
# Test case: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]
test_data = pd.DataFrame([['<21', 'Low', 'Female', 'Married']],
                         columns=['Age', 'Income', 'Gender', 'MaritalStatus'])

# Encode test data using the same encoding method
# Note: We encode each feature using the categories seen in training
for col in test_data.columns:
    # Create a new encoder and fit on the original data for that column
    le.fit(df[col])
    test_data[col] = le.transform(test_data[col])

print("Encoded Test Data:\n", test_data, "\n")

# Predict
prediction = model.predict(test_data)
print("Prediction (0=No, 1=Yes):", prediction)

# Interpret prediction
if prediction[0] == 1:
    print("\n✅ Decision: The customer is likely to BUY the saree.")
else:
    print("\n❌ Decision: The customer is NOT likely to buy the saree.")

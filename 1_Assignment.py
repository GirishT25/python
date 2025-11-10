# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the dataset
iris = pd.read_csv('iris.csv')
X = iris.iloc[:, :-1].values      # All columns except last one
y = iris.iloc[:, -1].values       # Last column (class labels)
feature_names = iris.columns[:-1]

# Convert to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("🔹 First 5 rows of dataset:")
print(df.head())

# Step 2: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Create a new DataFrame for PCA results
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Step 5: Visualize PCA result
plt.figure(figsize=(8,6))
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}

for species, color in colors.items():
    plt.scatter(df_pca.loc[df_pca['target'] == species, 'PC1'],
                df_pca.loc[df_pca['target'] == species, 'PC2'],
                label=species, color=color)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Feature Reduction on Iris Dataset')
plt.legend()
plt.show()

# Step 6: Explained Variance Ratio
print("\n🔹 Explained Variance Ratio:")
print(pca.explained_variance_ratio_)
print(f"\nTotal Variance Explained by 2 Components: {sum(pca.explained_variance_ratio_):.2f}")

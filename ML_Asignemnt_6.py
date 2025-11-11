import numpy as np

# Step 1: Define points
points = np.array([[0.1,0.6], [0.15,0.71], [0.08,0.9], [0.16,0.85],
                   [0.2,0.3], [0.25,0.5], [0.24,0.1], [0.3,0.2]])

# Step 2: Initial centroids
m1 = np.array([0.1,0.6])   # Cluster 1
m2 = np.array([0.3,0.2])   # Cluster 2

# Step 3: Assign points to nearest centroid
clusters = []
for p in points:
    d1 = np.linalg.norm(p - m1)  # distance to m1
    d2 = np.linalg.norm(p - m2)  # distance to m2
    clusters.append(1 if d1 < d2 else 2)

# Step 4: Update centroids
m1_new = np.mean(points[np.array(clusters)==1], axis=0)
m2_new = np.mean(points[np.array(clusters)==2], axis=0)

# Step 5: Print results
print("Cluster assignment:", clusters)
print("Updated m1:", m1_new)
print("Updated m2:", m2_new)
print("Population of cluster 2 (around m2):", clusters.count(2))
print("P6 belongs to cluster:", clusters[5])

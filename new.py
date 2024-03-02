from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Example data
X_train = [[0, 1], [1, 1], [2, 2], [3, 3]]
y_train = [0, 0, 1, 1]  # Example labels

# Initialize and fit the classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Example data for which we want prediction probabilities
X_test = [[1, 2], [2, 3]]

# Get the prediction probabilities
probabilities = knn.predict_proba(X_test)

# Output the prediction probabilities
print(probabilities)
output=np.argmax(probabilities[1])
print(output)
print(max(probabilities[0]))

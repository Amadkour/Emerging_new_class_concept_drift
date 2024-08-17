import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance


class KNNENS:
    def __init__(self, num_subsets, Ka, L, threshold):
        self.num_subsets = num_subsets  # Number of subsets b
        self.Ka = Ka  # Number of instances from each class
        self.L = L  # Number of nearest neighbors for radius calculation
        self.threshold = threshold  # Threshold t for classifying new classes
        self.hyperspheres = []  # To store the hypersphere ensembles

    def fit(self, X, y):
        """Train the KNNENS model by generating hypersphere ensembles."""
        classes = np.unique(y)
        for _ in range(self.num_subsets):
            subset_indices = np.array([], dtype=int)
            for cls in classes:
                cls_indices = np.where(y == cls)[0]
                if len(cls_indices) < self.Ka:
                    # If there are not enough instances, sample with replacement
                    selected_indices = np.random.choice(cls_indices, self.Ka, replace=True)
                else:
                    # Otherwise, sample without replacement
                    selected_indices = np.random.choice(cls_indices, self.Ka, replace=False)
                subset_indices = np.concatenate((subset_indices, selected_indices))
            X_subset = X[subset_indices]
            y_subset = y[subset_indices]
            self.hyperspheres.append(self._create_hyperspheres(X_subset, y_subset))

    def _create_hyperspheres(self, X, y):
        """Create a hypersphere ensemble for a given subset."""
        hyperspheres = []
        for i, c in enumerate(X):
            class_indices = np.where(y == y[i])[0]
            X_class = X[class_indices]
            nbrs = NearestNeighbors(n_neighbors=self.L).fit(X_class)
            distances, _ = nbrs.kneighbors([c])
            radius = np.mean(distances)
            hyperspheres.append((c, radius, y[i]))  # Store center, radius, and class label
        return hyperspheres

    def predict(self, X_test):
        """Classify and detect new classes for the test instances."""
        predictions = []
        for x in X_test:
            votes = []
            for hyperspheres in self.hyperspheres:
                class_prediction = self._classify(x, hyperspheres)
                votes.append(class_prediction)
            if votes:
                final_prediction = max(set(votes), key=votes.count)
            else:
                # Handle the case when there are no votes
                final_prediction = -1
            predictions.append(final_prediction)
        return predictions

    def _classify(self, x, hyperspheres):
        """Classify a single instance using hypersphere ensemble."""
        best_dist_ratio = float('inf')
        best_class = -1

        for c, r, cls in hyperspheres:
            dist = distance.euclidean(x, c)
            dist_ratio = dist / (r + 0.0001)

            if dist_ratio <= self.threshold and dist_ratio < best_dist_ratio:
                best_dist_ratio = dist_ratio
                best_class = cls

        return best_class

    def score(self, X_test, y_test):
        """Evaluate the model's accuracy."""
        predictions = self.predict(X_test)
        correct = np.sum(predictions == y_test)
        return correct / len(y_test)


# Example Usage:
# Generate some random data for demonstration
np.random.seed(42)
X_train = np.random.rand(100, 2)  # 100 instances with 2 features
y_train = np.random.choice([0, 1], 100)  # Binary classes

X_test = np.random.rand(20, 2)  # 20 test instances
y_test = np.random.choice([0, 1], 20)  # Ground truth for test

# Initialize the KNNENS model
knnens = KNNENS(num_subsets=5, Ka=10, L=5, threshold=1.5)

# Train the model
knnens.fit(X_train, y_train)

# Predict the classes for test data
predictions = knnens.predict(X_test)
print("Predictions:", predictions)

# Evaluate the model
accuracy = knnens.score(X_test, y_test)
print("Accuracy:", accuracy)

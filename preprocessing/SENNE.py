import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from collections import deque


class SENNE:
    def __init__(self, K, psi, p, buffer_size, threshold):
        """
        Initialize SENNE model.

        Parameters:
        K (int): Number of known classes.
        psi (int): Number of instances to subsample.
        p (int): Number of subsets (p times).
        buffer_size (int): Size of buffer for new class instances.
        threshold (float): Threshold for new class detection.
        """
        self.K = K
        self.psi = psi
        self.p = p
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.ensembles = {k: [] for k in range(1, K + 1)}  # Ensemble models for each class
        self.buffer = deque(maxlen=buffer_size)  # Buffer for new class instances

    def _build_hyperspheres(self, D):
        """Build isolation hyperspheres for a given subset D."""
        hyperspheres = []
        nbrs = NearestNeighbors(n_neighbors=2).fit(D)
        for c in D:
            _, indices = nbrs.kneighbors([c])
            eta_c = D[indices[0][1]]  # Nearest neighbor in the same subset
            radius = np.linalg.norm(c - eta_c)
            hyperspheres.append((c, radius))
        return hyperspheres

    def fit(self, X, y):
        """Fit the SENNE model."""
        for k in range(1, self.K + 1):
            D_k = X[y == k]
            for _ in range(self.p):
                D_k_i = D_k[np.random.choice(D_k.shape[0], self.psi, replace=False)]
                self.ensembles[k].append(self._build_hyperspheres(D_k_i))

    def _new_class_score(self, x, hyperspheres):
        """Calculate new-class score N(k)(x) for a single class."""
        scores = []
        for hs in hyperspheres:
            for c, radius in hs:
                dist = np.linalg.norm(x - c)
                if dist <= radius:
                    scores.append(1 - radius / (dist + 1e-9))
                    break
            else:
                scores.append(1)
        return np.mean(scores)

    def _known_class_score(self, x, hyperspheres):
        """Calculate known-class score P(k)(x) for a single class."""
        scores = []
        for hs in hyperspheres:
            for c, radius in hs:
                dist = np.linalg.norm(x - c)
                if dist <= radius:
                    scores.append(1)
                    break
            else:
                scores.append(0)
        return np.mean(scores)

    def predict(self, X):
        """Predict class for each instance in X."""
        predictions = []
        for x in X:
            new_class_scores = [self._new_class_score(x, self.ensembles[k]) for k in range(1, self.K + 1)]
            if all(score >= self.threshold for score in new_class_scores):
                predictions.append(-1)
                self.buffer.append(x)
                if len(self.buffer) == self.buffer_size:
                    self.update_model()
            else:
                known_class_scores = [self._known_class_score(x, self.ensembles[k]) for k in range(1, self.K + 1)]
                predictions.append(np.argmax(known_class_scores) + 1)
        return predictions

    def update_model(self):
        """Update the model with new class data from the buffer."""
        self.K += 1
        self.ensembles[self.K] = []
        new_class_data = np.array(self.buffer)
        for _ in range(self.p):
            D_new_i = new_class_data[np.random.choice(new_class_data.shape[0], self.psi, replace=False)]
            self.ensembles[self.K].append(self._build_hyperspheres(D_new_i))
        self.buffer.clear()


# Example Usage:

# Generating some random data for demonstration
np.random.seed(42)
X_train = np.random.rand(100, 2)  # 100 instances with 2 features
y_train = np.random.choice([1, 2], 100)  # Binary classes 1 and 2

X_test = np.random.rand(20, 2)  # 20 test instances

# Initialize the SENNE model
senne = SENNE(K=2, psi=5, p=3, buffer_size=5, threshold=0.5)

# Train the model
senne.fit(X_train, y_train)

# Predict the classes for test data
predictions = senne.predict(X_test)
print("Predictions:", predictions)

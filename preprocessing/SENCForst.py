import random
import numpy as np
from scipy.spatial.distance import euclidean


class SENCTreeNode:
    def __init__(self, is_leaf=False, size=None, class_distribution=None, center=None, radius=None, split_attr=None,
                 split_value=None):
        self.is_leaf = is_leaf
        self.size = size
        self.class_distribution = class_distribution
        self.center = center
        self.radius = radius
        self.split_attr = split_attr
        self.split_value = split_value
        self.left = None
        self.right = None


class SENCTree:
    def __init__(self, min_size=5):
        self.min_size = min_size
        self.root = None

    def build_tree(self, X, class_labels):
        if len(X) < self.min_size:
            center = np.mean(X, axis=0)
            radius = 0.00001 if len(X) == 0 else max([euclidean(center, x) for x in X])
            class_distribution = {cls: sum([1 for label in class_labels if label == cls]) for cls in set(class_labels)}
            return SENCTreeNode(is_leaf=True, size=len(X), class_distribution=class_distribution, center=center,
                                radius=radius)
        else:
            attr = random.choice(range(X.shape[1]))
            split_value = random.uniform(X[:, attr].min(), X[:, attr].max())
            left_mask = X[:, attr] <= split_value
            right_mask = X[:, attr] > split_value
            left_node = self.build_tree(X[left_mask], class_labels[left_mask])
            right_node = self.build_tree(X[right_mask], class_labels[right_mask])
            node = SENCTreeNode(split_attr=attr, split_value=split_value)
            node.left = left_node
            node.right = right_node
            return node

    def fit(self, X, class_labels):
        self.root = self.build_tree(X, class_labels)

    def predict(self, x):
        node = self.root
        if node is None:
            return -1 # or handle the None case as needed
        while not node.is_leaf:
            if x[node.split_attr] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return node.class_distribution


class SENCForest:
    def __init__(self, n_trees=10, subsample_size=256, min_size=5):
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.trees = [SENCTree(min_size=min_size) for _ in range(n_trees)]
        self.buffer = []
        self.buffer_size = 5
        self.known_classes = set()

    def fit(self, X, class_labels):
        self.known_classes = set(class_labels)
        for tree in self.trees:
            subsample_indices = np.random.choice(len(X), self.subsample_size, replace=len(X)<self.subsample_size)
            tree.fit(X[subsample_indices], class_labels[subsample_indices])

    def predict(self, x):
        votes = []
        for tree in self.trees:
            class_distribution = tree.predict(x)
            if class_distribution:
                majority_class = max(class_distribution, key=class_distribution.get)
                if majority_class in self.known_classes:
                    votes.append(majority_class)
                else:
                    votes.append(-1)
            else:
                votes.append(-1)

        final_prediction = max(set(votes), key=votes.count)
        if final_prediction == -1:
            self.buffer.append(x)
        return final_prediction

    def update_forest(self):
        # Update the forest with new instances from the buffer
        # For simplicity, let's re-train trees from scratch using buffer data
        print(f"Updating SENCForest with buffer of size {len(self.buffer)}")
        self.buffer = []

# Example usage:
# Assume X is a numpy array of training instances and y are the corresponding class labels
# X = np.array([...])
# y = np.array([...])
# forest = SENCForest()
# forest.fit(X, y)

# Now we can predict on new instances:
# new_instance = np.array([...])
# prediction = forest.predict(new_instance)
# print(f"Predicted class: {prediction}")

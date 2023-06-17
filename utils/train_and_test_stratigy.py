import math

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
import pandas as pd


class MyTestThenTrain:

    def __init__(
            self, metrics=(accuracy_score, balanced_accuracy_score), verbose=False
    ):
        self.buffer_y = []
        self.buffer_x = []
        self.centroids_distance = []
        self.balanced_methods = []
        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]
        self.verbose = verbose

    def process(self, stream, clfs, concept_drift_method=None):
        """
        Perform learning procedure on data stream.

        :param stream: Data stream as an object
        :type stream: object
        :param clfs: scikit-learn estimator of list of scikit-learn estimators.
        :type clfs: tuple or function

        Parameters
        ----------
        use_concept_drift
        """
        # Verify if pool of classifiers or one
        if isinstance(clfs, ClassifierMixin):
            self.clfs_ = [clfs]
        else:
            self.clfs_ = clfs

        # Assign parameters
        self.stream_ = stream

        # Prepare scores table
        self.scores = np.zeros(
            (len(self.clfs_), ((self.stream_.n_chunks - 1)), len(self.metrics))
        )
        index = 0
        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
        have_drift = 0
        while True:
            chunk = stream.get_chunk()
            index += 1
            X, y = chunk
            self.calculate_centroid(X, y)
            self.centroids_distance = self.calculate_centroid_distances(centroids)

            if self.verbose:
                pbar.update(1)

            # Test
            y_prediction = np.zeros(y.shape)
            if stream.previous_chunk is not None:
                for clfid, clf in enumerate(self.clfs_):
                    y_pred = clf.predict(X)
                    for i in range(len(y_prediction)):
                        y_prediction[i] = y_prediction[i] + y_pred[i]

                    self.scores[clfid, stream.chunk_id - 1] = [
                        metric(y, y_pred) for metric in self.metrics]
            if stream.previous_chunk is not None and concept_drift_method is not None:
                '''concept drift'''
                for index in range(len(y_prediction)):
                    concept_drift_method.add_element(y_prediction[index])
                    if concept_drift_method.detected_change():
                        self.calculations(X[index], y_prediction[index], concept_drift_method)
            else:
                print(index)
                # Train
                [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]
                print(index, ' is finish')

            if stream.is_dry():
                break

    def calculations(self, X, y, concept_drift_method):
        """calculate distance between X and centroids"""
        distance = []
        for centroid in centroids:
            distance.append(self.calculate_distance(centroid, X))
        if np.min(distance) > np.max(self.centroids_distance):
            if len(self.buffer_y) < buffer_length:
                '''detect new object'''
                print(f"detect new object ${y}")
                self.buffer_x.append(X)
                self.buffer_y.append(y)
            else:
                '''detect new object'''
                print('buffer is full')
                print('this points used as a new class',self.buffer_y)
                # Train
                print("======================[training]")
                print(self.buffer_x)
                print(self.buffer_y)
                [clf.partial_fit(self.buffer_x, self.buffer_y, self.stream_.classes_) for clf in self.clfs_]
                self.buffer_x.clear()
                self.buffer_y.clear()

        else:
            '''update model'''
            concept_drift_method.reset()
            # Train
            [clf.partial_fit(X, y, self.stream_.classes_) for clf in self.clfs_]

    def calculate_distance(self, point_a, point_b):
        point_a = np.array(point_a)
        point_b = np.array(point_b)
        distance = np.linalg.norm(point_b - point_a)
        return distance

    def calculate_center(self, points):
        num_points = len(points)
        num_dimensions = len(points[0])

        center = [0] * num_dimensions

        for point in points:
            for i in range(num_dimensions):
                center[i] += point[i]

        center = [coord / num_points for coord in center]
        return center

    def calculate_centroid(self, X, y):
        for i in np.unique(y):
            subset = X[y == i]
            centroids.append(self.calculate_center(subset))

    def calculate_centroid_distances(self, points):
        num_points = len(points)
        num_dimensions = len(points[0])

        distances = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                point1 = np.array(points[i])
                point2 = np.array(points[j])
                distance = np.sqrt(np.sum((point1 - point2) ** 2))
                distances[i][j] = distance
                distances[j][i] = distance

        return distances
buffer_length = 10
centroids = []

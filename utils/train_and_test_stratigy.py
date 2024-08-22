import math

import numpy
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd

from preprocessing.KENNE import KNNENS
from preprocessing.SENCForst import SENCForest
from preprocessing.SENNE import SENNE


class MyTestThenTrain:

    def __init__(
            self,classifier,algoritnmsName, metrics=(accuracy_score, balanced_accuracy_score), concept_drift_method=None, thresold=10,
            verbose=False
    ):
        self.stream_ = None
        self.buffer_x = []
        self.concept_drift_method = concept_drift_method
        self.thresold = thresold
        self.centroids_distance = []
        self.balanced_methods = []
        self.classifier = classifier
        self.KENNE = KNNENS(num_subsets=5, Ka=10, L=5, threshold=1.5)
        self.SENCForst = SENCForest()
        self.SENNE = SENNE(K=2, psi=5, p=3, buffer_size=5, threshold=0.5)

        if isinstance(metrics, (list, tuple)):
            self.metrics = metrics
        else:
            self.metrics = [metrics]

        self.verbose = verbose
        self.scores = np.zeros(
            (len(algoritnmsName), 200, len(self.metrics))
        )
    def score_of_relatedWorks(self,X, y, y_pred, stream_index,algorithmIndex):
        indices = [i for i, y in enumerate(y_pred) if y == -1]
        self.buffer_x.append(X[indices])
        indices2 = [i for i, x in enumerate(y_pred) if x != -1]
        # print(np.unique(y[indices2]))
        # print(np.unique(y_pred[indices2],))
        self.scores[algorithmIndex, stream_index] = [
            metric(y[indices2], y_pred[indices2], average='macro') if metric == f1_score else metric(y[indices2],
                                                                                                   y_pred[indices2])
            for metric in self.metrics
        ]

    def process(self, stream, algorithm='pa', algorithmIndex=0 ):
        """
        Perform learning procedure on data stream.

        :param concept_drift_method: 
        :param stream: Data stream as an object
        :type stream: object
        :param classifier: scikit-learn estimator of list of scikit-learn estimators.
        :type classifier: tuple or function

        Parameters
        ----------
        use_concept_drift
        """
        drift_count = 0
        # Verify if pool of classifiers or one
        if isinstance(self.classifier, ClassifierMixin):
            self.clfs_ = [self.classifier]
        else:
            self.clfs_ = self.classifier

        # Assign parameters
        self.stream_ = stream

        # Prepare scores table
        if self.verbose:
            pbar = tqdm(total=stream.n_chunks)
        have_drift = 0
        while True:

            chunk = stream.get_chunk()
            print(stream.chunk_id)
            X, y = chunk
            if algorithm == 'KENNE':
                if stream.previous_chunk is not None:
                    y_pred = numpy.array(self.KENNE.predict(X))
                    self.score_of_relatedWorks(X,y, y_pred, min(stream.chunk_id, 199),algorithmIndex=algorithmIndex)
                else:
                    self.KENNE.fit(X, y)
            elif algorithm == 'SENCForst':
                if stream.previous_chunk is not None:
                    y_pred = numpy.array([self.SENCForst.predict(x) for x in X])
                    self.score_of_relatedWorks(X,y, y_pred, min(stream.chunk_id, 199),algorithmIndex=algorithmIndex)
                else:
                    self.SENCForst.fit(X, y)
            elif algorithm == 'SENNE':
                if stream.previous_chunk is not None:
                    y_pred = numpy.array(self.SENNE.predict(X))
                    self.score_of_relatedWorks(X,y, y_pred, min(stream.chunk_id, 199),algorithmIndex=algorithmIndex)

                else:
                    self.SENNE.fit(X, y)

            else:
                self.calculate_centroid(X, y)
                # calculate centriod of each class
                self.centroids_distance = self.calculate_centroid_distances(centroids)

                if self.verbose:
                    pbar.update(1)

                # Test
                y_prediction = np.zeros(y.shape)
                if stream.previous_chunk is not None:
                    y_pred = self.clfs_[0].predict(X)
                    for i in range(len(y_prediction)):
                        y_prediction[i] = y_prediction[i] + y_pred[i]
                    '''concept drift'''
                    for index2 in range(len(y_prediction)):
                        # self.concept_drift_method.add_element(y_prediction[index2])
                        self.concept_drift_method.update(y_prediction[index2])
                        # if self.concept_drift_method.detected_change():
                        if self.concept_drift_method.drift_detected:
                            # self.calculations(X[index2], y_prediction[index2], concept_drift_method,clf)
                            self.calculations(X, y, index2, self.concept_drift_method)
                        self.scores[algorithmIndex, stream.chunk_id - 1] = [
                            metric(y, y_pred, average='macro') if metric == f1_score else metric(y, y_pred)
                            for metric in self.metrics
                        ]
                else:
                    # Train
                    [c.partial_fit(X, y, self.stream_.classes_) for c in self.clfs_]
                    print(min(stream.chunk_id, 199), ' is finish')

            if len(self.buffer_x) >= 5:
                drift_count+=1
                try:
                    [c.partial_fit(X, y, self.stream_.classes_) for c in self.clfs_]
                    self.buffer_x.clear()
                except:
                    print("=================> error here: ")
            if stream.is_dry():
                print("==================[number of updates: %s]"%drift_count)
                drift_count=0
                break

    def calculations(self, X, y, index, concept_drift_method):
        """calculate distance between X and centroids"""
        distance = []
        for centroid in centroids:
            distance.append(self.calculate_distance(centroid, X))
        # print(self.centroids_distance)
        if np.min(distance) > np.max(self.centroids_distance):
            '''detect new object'''
            self.buffer_x.append(X[index])
        else:

            '''update model'''
            # concept_drift_method.reset()
            # # Train
            # print(X)
            # print(y)

    def calculate_distance(self, point_a, point_b):
        point_a = np.array(point_a)
        point_b = np.array(point_b)
        distance = np.sqrt(np.sum((point_a - point_b) ** 2))
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
        centroids.clear()
        for i in np.unique(y):
            subset = X[y == i]
            center = self.calculate_center(subset)
            centroids.append(center)

    def calculate_centroid_distances(self, points):
        num_points = len(points)
        distances = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(i + 1, num_points):
                point1 = np.array(points[i])
                point2 = np.array(points[j])
                distance = np.sqrt(np.sum((point1 - point2) ** 2))
                distances[i][j] = distance
                distances[j][i] = distance
        return distances


buffer_length = 2
centroids = []

import time
from sklearn.metrics import balanced_accuracy_score, f1_score

import numpy as np
from sklearn.naive_bayes import GaussianNB
from skmultiflow.drift_detection import ADWIN, DDM
from sklearn.base import clone
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Select streams and methods
from strlearn.metrics import geometric_mean_score_1, precision, recall

from utils import helper
from utils.SEA import SEA
from utils.StratifiedBagging import StratifiedBagging
from utils.train_and_test_stratigy import MyTestThenTrain

streams = helper.realstreams()

gnb = SEA(base_estimator=StratifiedBagging(base_estimator=GaussianNB()), des="None", oversampled='None')
knn = SEA(base_estimator=StratifiedBagging(base_estimator=KNeighborsClassifier(n_neighbors=1)), des="None",
           oversampled='None')
svc = SEA(base_estimator=StratifiedBagging(base_estimator=SVC(probability=True, random_state=42)), des="None",
           oversampled=None)
ht = SEA(base_estimator=StratifiedBagging(base_estimator=HoeffdingTreeClassifier()), des="None", oversampled='None')


# Define worker
def worker(i, stream_n):
    stream = streams[stream_n]
    drift_methods = [
        ADWIN(),
        # DDM()
    ]
    classifiers_name = [
        # 'gnb',
        # 'knn',
        # 'svc',
        # 'ht',
        'mix'
    ]
    classifiers = [

        (
            ht,
            gnb,
            knn,
            svc
        )]

    threshold = [10]

    for t in threshold:
        for index, classifer_name in enumerate(classifiers_name):
            cclfs = [clone(clf) for clf in classifiers[index]]

            for method in drift_methods:
                print("Starting stream %i/%i" % (i + 1, len(streams)))
                eval = MyTestThenTrain(metrics=(
                    balanced_accuracy_score,
                    geometric_mean_score_1,
                    f1_score,
                    precision,
                    recall,
                ))
                eval.process(
                    stream,
                    cclfs,
                    concept_drift_method=method,
                    thresold=t

                )

                print("Done stream %i/%i" % (i + 1, len(streams)))

                results = eval.scores
                # np.save(f"output/result1/%s/cover_type/%s" % (classifiers_name[index], t), results)
                # np.save(f"output/result1/%s/synthetic/%s" % (classifiers_name[index], t), results)
                np.save(f"output/result1/%s/cover_type/%s" % (classifiers_name[index], t), results)


jobs = []
if __name__ == '__main__':
    from joblib import Parallel, delayed

    # for i, stream_n in enumerate(streams):
    # print('chunk number', i)
    # worker(i, stream_n)
    start_time = time.perf_counter()
    result = Parallel(n_jobs=4, prefer="threads")(delayed(worker)(i, stream_n) for i, stream_n in enumerate(streams))
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    print(result)

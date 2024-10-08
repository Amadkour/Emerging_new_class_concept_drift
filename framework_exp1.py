import time

import numpy as np
from sklearn.naive_bayes import GaussianNB
from river.drift import ADWIN
from sklearn.base import clone
from river.tree import HoeffdingTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Select streams and methods
from strlearn.metrics import geometric_mean_score_1, precision, recall,balanced_accuracy_score, f1_score

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

algorithms= ['SENCForst',
             'KENNE', 'SENNE',
             "PA", ]
# Define worker
def worker(i, stream_n):
    drift_methods = [
        ADWIN(),
        # DDM()
    ]
    classifiers_name = [
        # 'gnb',
        # 'knn',
        # 'svc',
        # 'ht',
        'gnb'
    ]
    classifiers = [

        (
            gnb,
            # svc
            # ht,

            # knn,

        )]

    thresholds = ['adaptive','5','10']
    # thresholds = ['20']
    cclfs = [clone(clf) for clf in classifiers[0]]
    for threshold in thresholds:

        eval = MyTestThenTrain(
            cclfs,
            algoritnms_count=len(algorithms),
            buffer_type=threshold,
            metrics=(balanced_accuracy_score,geometric_mean_score_1,f1_score,precision,recall),
            concept_drift_method=drift_methods[0],
        )
        for index,algorithm in enumerate(algorithms):
                start_time = time.perf_counter()
                streams = helper.realstreams()
                print("Starting stream %i/%i" % (i + 1, len(streams)))
                eval.process(
                    streams[stream_n],
                    algorithm=algorithm,
                    algorithmIndex=index
                )

                finish_time = time.perf_counter()
                print(f"==========(%s)===with threshold (%s)======finished in {finish_time - start_time} seconds"% (algorithm,threshold),)

        results = eval.scores
        # np.save(f"output/result1/%s/cover_type/%s" % (classifiers_name[index], t), results)
        # np.save(f"output/result1/%s/synthetic/%s" % (classifiers_name[index], t), results)
        np.save(f"output/result1/%s/cover_type/%s" % (classifiers_name[0],threshold), results)


if __name__ == '__main__':
    from joblib import Parallel, delayed

    # for i, stream_n in enumerate(streams):
    # print('chunk number', i)
    # worker(i, stream_n)
    start_time = time.perf_counter()
    result = Parallel(n_jobs=4, prefer="threads")(delayed(worker)(i, stream_n) for i, stream_n in enumerate(streams))
    print(result)

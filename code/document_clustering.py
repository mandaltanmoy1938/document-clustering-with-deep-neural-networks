import logging as log
import time

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

import global_variables as gv
import object_pickler as op
import timer

log.basicConfig(filename='document_clustering.log', level=log.DEBUG, filemode="w")


def load_pickle(filename):
    # filename without extension
    return op.load_object(gv.prj_src_path + "python_objects/" + filename)


def train_test():
    data_label = [{"data": "train_vector", "label": "train_labels",
                   "test_data": "test_vector", "test_label": "test_labels"}
                  # ,
                  #           {"data": "train_data_transformed", "label": "train_labels",
                  #            "test_data": "test_data_transformed", "test_label": "test_labels"}
                  ]

    try_algorithms = {
        "supervised": {"SVC_linear": svm.SVC(kernel='linear', C=1, random_state=0),
                       "SVC_poly": svm.SVC(kernel='poly', C=1, random_state=0),
                       "SVC_rbf": svm.SVC(kernel='rbf', C=1, random_state=0),
                       "LogisticRegression": LogisticRegression()},
        "unsupervised": {"KMeans": KMeans(n_clusters=15)}
    }
    for dl in data_label:
        data = load_pickle(dl["data"])
        test_data = load_pickle(dl["test_data"])
        labels = load_pickle(dl["label"])
        labels = [gv.translation[x] for x in labels]

        for s_u, algos in try_algorithms.items():
            s_u_time = time.time()
            log.info("%s  starts at %s" % (s_u, time.localtime(s_u_time)))

            for algo, clf in algos.items():
                algo_time = time.time()
                log.info("\tAlgorithm: %s \n\t\ttraining starts at %s" % (algo, time.localtime(algo_time)))
                clf.fit(data, labels)
                timer.time_executed(algo_time, "\t\ttraining")

                predict_time = time.time()
                log.info("\t\t%s predict starts at %s" % (algo, time.localtime(predict_time)))
                predict = clf.predict(test_data)
                timer.time_executed(predict_time, "\t\tpredict")
                op.save_object(predict, gv.prj_src_path + "python_objects/%s_%s_predict" % (algo, dl["test_data"]))


def run():
    train_test()


def main():
    run()


if __name__ == '__main__':
    start = time.time()
    log.info(("Document clustering started: ", time.localtime(start)))
    try:
        main()
    except Exception as ex:
        log.exception(ex)
    timer.time_executed(start, "Document clustering")

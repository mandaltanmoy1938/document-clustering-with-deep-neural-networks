import time
import logging as log
import object_pickler as op
import global_variables as gv
import timer

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans, MeanShift

log.basicConfig(filename='document_clustering.log', level=log.DEBUG, filemode="w")


def loadPickle(filename):
    # filename without extension
    return op.load_object(gv.prj_src_path + "python_objects/" + filename)


def train_test():
    data_label = [{"data": "train_data_transformed", "label": "train_labels",
                   "test_data": "test_data_transformed", "test_label": "test_labels"}]

    try_algorithms = {"supervised": {"SVC": svm.SVC(kernel='linear', C=1, random_state=0), "NB": MultinomialNB(),
                                     "LogisticRegression": LogisticRegression()},
                      "unsupervised": {"KMeans": KMeans(n_clusters=15), "MeanShift": MeanShift()}
                      }
    for dl in data_label:
        data = loadPickle(dl["data"])
        test_data = loadPickle(dl["test_data"])
        labels = loadPickle(dl["label"])
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

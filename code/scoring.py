import time
import logging as log
import object_pickler as op
import global_variables as gv
import timer
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score, silhouette_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

log.basicConfig(filename='scoring.log', level=log.DEBUG, filemode="w")


def loadPickle(filename):
    # filename without extension
    return op.load_object(gv.prj_src_path + "python_objects/" + filename)


def run():
    predicted_label = [{"unsupervised": ["KMeans_"],
                        "supervised": ["LogisticRegression_", "NB_", "SVC_"]}]

    y_true = loadPickle("test_labels")
    for algo in predicted_label["supervised"]:
        predict = loadPickle(algo + "test_data_transformed_predict")
        accuracy = accuracy_score(y_true, predict)
        f1 = f1_score(y_true, predict, average='macro')
        recall = recall_score(y_true, predict, average='macro')
        precision = precision_score(y_true, predict, average='macro')
        log.info("Algorithm supervised: %s \n\taccuracy:\t%s"
                 "\n\t f1_macro:\t%s\n\trecall_macro:\t%s\n\tprecision_macro:\t%s" %
                 (algo, accuracy, f1, recall, precision))

    for algo in predicted_label["unsupervised"]:
        predict = loadPickle(algo + "test_data_transformed_predict")
        score_h = homogeneity_score(y_true, predict)
        score_c = completeness_score(y_true, predict)
        score_v = v_measure_score(y_true, predict)
        score_a = adjusted_rand_score(y_true, predict)
        score_am = adjusted_mutual_info_score(y_true, predict)
        log.info("Algorithm unsupervised: %s \n\thomogeneity:\t%s\n\t completeness:\t%s\n\tv_measure:\t%s"
                 "\n\tadjusted_rand:\t%s\n\tadjusted_mutual_info:\t%s" %
                 (algo, score_h, score_c, score_v, score_a, score_am))


def main():
    run()


if __name__ == '__main__':
    start = time.time()
    log.info(("Scoring started: ", time.localtime(start)))
    try:
        main()
    except Exception as ex:
        log.error(ex)
    timer.time_executed(start, "Scoring")

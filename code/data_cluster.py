import time
import logging as log
import object_pickler as op
import global_variables as gv
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate


def main():
    clf = KMeans(n_clusters=15)
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']

    #################Test##############
    test_data_transformed = op.load_object(gv.prj_src_path + "python_objects/test_data_transformed")
    test_labels = op.load_object(gv.prj_src_path + "python_objects/test_labels")
    scores = cross_validate(clf, test_data_transformed, test_labels, scoring=scoring, cv=5, return_train_score=False)
    print("test")
    for k in sorted(scores.keys()):
        print("\t%s: %0.2f (+/- %0.2f)" % (k, scores[k].mean(), scores[k].std() * 2))

    #################Train##############
    train_data_transformed = op.load_object(gv.prj_src_path + "python_objects/train_data_transformed")
    train_labels = op.load_object(gv.prj_src_path + "python_objects/train_labels")
    scores = cross_validate(clf, test_data_transformed, test_labels, scoring=scoring, cv=5, return_train_score=False)
    print("train")
    for k in sorted(scores.keys()):
        print("\t%s: %0.2f (+/- %0.2f)" % (k, scores[k].mean(), scores[k].std() * 2))

    #################Val##############
    val_data_transformed = op.load_object(gv.prj_src_path + "python_objects/val_data_transformed")
    val_labels = op.load_object(gv.prj_src_path + "python_objects/val_labels")
    scores = cross_validate(clf, test_data_transformed, test_labels, scoring=scoring, cv=5, return_train_score=False)
    print("val")
    for k in sorted(scores.keys()):
        print("\t%s: %0.2f (+/- %0.2f)" % (k, scores[k].mean(), scores[k].std() * 2))


if __name__ == '__main__':
    start_time = time.time()
    log.info(("Data processor started: ", time.localtime(start_time)))
    main()
    end_time = time.time()
    log.info(("Data processor ended: ", time.localtime(end_time)))
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log.info(("Data processor executed for {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))

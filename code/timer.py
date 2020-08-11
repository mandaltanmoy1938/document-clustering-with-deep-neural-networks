import time
import pandas as pd
import logging as log
import object_pickler as op
import global_variables as gv

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift


def time_executed(start_time, process_name):
    end_time = time.time()
    log.info("%s ended: %s" % (process_name, time.localtime(end_time)))
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log.info((process_name, " executed for {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))

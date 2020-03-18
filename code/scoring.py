import time
import pandas as pd
import logging as log
import object_pickler as op
import global_variables as gv
import timer

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift

def run():
    print("scoring")

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
import time
import spacy
import logging as log

nlp = spacy.load("en")


def main():
    return


if __name__ == '__main__':
    start_time = time.time()
    log.info(("Statistics started: ", time.localtime(time.time())))
    main()
    log.info(("Statistics ended: ", time.localtime(time.time())))
    end_time = time.time()
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    log.info(("Statistics executed for {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)))

import pickle
import os
import logging as log
from os import path


def load_object(file_name):
    object = dict()
    try:
        with open(file_name + '.p', 'rb') as handle:
            try:
                object = pickle.load(handle)
            except Exception as e:
                log.warning(("Error load file: ", file_name))
                log.error(e)
    except Exception as ex:
        log.warning(("In load error open file: ", file_name))
        log.error(ex)
    return object


def save_object(object, file_name):
    try:
        if not path.exists(file_name + '.p'):
            os.mknod(file_name + '.p')

        with open(file_name + '.p', 'wb+') as fp:
            try:
                pickle.dump(object, fp, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                log.warning(("Error dump file: ", file_name))
                log.error(e)
    except Exception as ex:
        log.warning(("In save error open file: ", file_name))
        log.error(ex)

import os
import logging as log
import global_variables as gv


def read_file(file_name):
    # global error_file_count
    lines = []
    with open(file_name, 'rt', encoding="utf-8") as f:
        try:
            lines = f.readlines()
        except Exception as e:
            gv.error_file_count += 1
            log.warning(("Error file: ", file_name))
            log.error(e)
    return lines


def get_all_files_from_directory(dir_path):
    data_paths = list()
    for path, sub_dirs, files in os.walk(dir_path):
        for name in files:
            if name not in ["text_test.txt", "text_train.txt", "text_val.txt"]:
                data_paths.append(os.path.join(path, name))
    return data_paths

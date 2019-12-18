import time
import spacy
import logging as log
import data_labeller as dl
import file_collector as fc
import global_variables as gv

log.basicConfig(filename='data_processor.log', level=log.DEBUG, filemode="w")


def tokenizer(file_paths, nlp, skipped_files, required_files):
    document_meta = dict()
    for file_path in file_paths:
        file_path_ = file_path.replace(gv.src_path, "").replace("\\", "/")
        if file_path_ not in skipped_files and file_path_ in required_files:
            lines = fc.read_file(file_path)
            text = "".join(lines)
            parsed_text = nlp(text)
            document_meta['{}'.format(file_path.replace(gv.src_path, "").replace("\\", "/"))] = dict()
            for token in parsed_text:
                document_meta['{}'.format(file_path.replace(gv.src_path, "").replace("\\", "/"))][token.text] = 1.0
    return document_meta


def main():
    # test_paths_by_label, test_labels_by_path = dl.get_labels(fc.read_file(gv.src_path + gv.test_label_file_name),
    #                                                          gv.test_label_file_name)
    # train_paths_by_label, train_labels_by_path = dl.get_labels(fc.read_file(gv.src_path + gv.train_label_file_name),
    #                                                            gv.train_label_file_name)
    val_paths_by_label, val_labels_by_path = dl.get_labels(fc.read_file(gv.src_path + gv.val_label_file_name),
                                                           gv.val_label_file_name)

    nlp = spacy.load("en_core_web_sm")
    file_paths = fc.get_all_files_from_directory(gv.src_path)
    # test_document_meta = tokenizer(file_paths=file_paths, nlp=nlp, skipped_files=test_paths_by_label["3"],
    #                           required_files=test_labels_by_path)
    # train_document_meta = tokenizer(file_paths=file_paths, nlp=nlp, skipped_files=train_paths_by_label["3"],
    #                           required_files=train_labels_by_path)
    val_document_meta = tokenizer(file_paths=file_paths, nlp=nlp, skipped_files=val_paths_by_label["3"],
                                  required_files=val_labels_by_path)


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

import time
import spacy
import logging as log
import data_labeller as dl
import file_collector as fc
import object_pickler as op
import global_variables as gv

from sklearn.feature_extraction import DictVectorizer

log.basicConfig(filename='data_processor.log', level=log.DEBUG, filemode="w")


def dict_vectorizer(data_dict, label_dct):
    labels = list()
    data = list()
    for file_path, value in data_dict.items():
        labels.append(label_dct[file_path])
        data.append(value)
    dv = DictVectorizer(sparse=False)
    data_transformed = dv.fit_transform(data)
    return data_transformed, labels


def tokenizer(nlp, required_files):
    document_meta = dict()
    parsed_documents = dict()
    for file_path_ in required_files:
        file_path = gv.data_src_path + file_path_
        lines = fc.read_file(file_path)
        text = "".join(lines)
        parsed_text = nlp(text)
        parsed_documents[file_path_] = parsed_text
        document_meta[file_path_] = dict()
        for token in parsed_text:
            if token.pos_ is "NUM":
                token_key = "<NUM>"
            else:
                token_key = token.text.strip()
            if len(token_key) > 0:
                document_meta[file_path_][token_key] = 1.0
    return document_meta, parsed_documents


def run():
    test_paths_by_label, test_labels_by_path = dl.get_labels(fc.read_file(gv.data_src_path + gv.test_label_file_name),
                                                             gv.test_label_file_name)

    # train_paths_by_label, train_labels_by_path = dl.get_labels(
    #     fc.read_file(gv.data_src_path + gv.train_label_file_name), gv.train_label_file_name)

    val_paths_by_label, val_labels_by_path = dl.get_labels(fc.read_file(gv.data_src_path + gv.val_label_file_name),
                                                           gv.val_label_file_name)

    nlp = spacy.load("en_core_web_sm")
    # file_paths = fc.get_all_files_from_directory(gv.src_path)
    test_document_meta, test_parsed_documents = tokenizer(nlp=nlp, required_files=test_labels_by_path)
    op.save_object(test_document_meta, gv.prj_src_path + "python_objects/test_document_meta")
    op.save_object(test_parsed_documents, gv.prj_src_path + "python_objects/test_parsed_documents")

    # train_document_meta, train_parsed_documents = tokenizer(nlp=nlp, required_files=train_labels_by_path)
    # op.save_object(train_document_meta, gv.prj_src_path + "python_objects/train_document_meta")
    # op.save_object(train_parsed_documents, gv.prj_src_path + "python_objects/train_parsed_documents")

    # val_document_meta, val_parsed_documents = tokenizer(nlp=nlp, required_files=val_labels_by_path)
    # op.save_object(val_document_meta, gv.prj_src_path + "python_objects/val_document_meta")
    # op.save_object(val_parsed_documents, gv.prj_src_path + "python_objects/val_parsed_documents")

    test_document_meta = op.load_object(gv.prj_src_path + "python_objects/test_document_meta.p")
    # train_document_meta = op.load_object(gv.prj_src_path+"python_objects/train_document_meta.p")
    val_document_meta = op.load_object(gv.prj_src_path + "python_objects/val_document_meta.p")

    test_data_transformed, test_labels = dict_vectorizer(data_dict=test_document_meta, label_dct=test_labels_by_path)
    # train_data_transformed, train_labels = dict_vectorizer(data_dict=train_document_meta,
    #                                                        label_dct=train_labels_by_path)
    val_data_transformed, val_labels = dict_vectorizer(data_dict=val_document_meta, label_dct=val_labels_by_path)


def main():
    run()


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

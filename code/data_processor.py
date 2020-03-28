import time
import json
import spacy
import logging as log
import data_labeller as dl
import file_collector as fc
import object_pickler as op
import global_variables as gv

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import DictVectorizer

log.basicConfig(filename='data_processor.log', level=log.DEBUG, filemode="w")


# !vectorizer is must transform the test and val data not fit transform!
def load_stop_words():
    with open(gv.prj_src_path + "data/stopwords-en.txt", "rt", encoding="utf-8-sig") as infile:
        stopwords_en = json.load(infile)["en"]
        return stopwords_en


def hashing_vectorizer(data_dict, label_dict, lower_case, stop_words, n_features):
    labels = list()
    data = list()
    for file_path, value in data_dict.items():
        if len(value.strip()) is not 0:
            labels.append(label_dict[file_path])
            data.append(value)
    vectorizer = HashingVectorizer(lowercase=lower_case, stop_words=stop_words, n_features=n_features, analyzer='word')
    data_transformed = vectorizer.fit_transform(data_dict)
    return data_transformed, labels


def dict_vectorizer(data_dict, label_dict):
    labels = list()
    data = list()
    for file_path, value in data_dict.items():
        if len(value) is not 0:
            labels.append(label_dict[file_path])
            data.append(value)
    dv = DictVectorizer(sparse=True)
    data_transformed = dv.fit_transform(data)
    return data_transformed, labels


def tokenizer(required_files):
    stopwords_en = load_stop_words()

    nlp = spacy.load("en_core_web_sm")
    document_meta = dict()
    modified_texts = dict()
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
                text = text.replace(token.text, token_key)
            else:
                token_key = token.text.strip()
            if len(token_key) > 0 and token_key not in stopwords_en:
                document_meta[file_path_][token_key] = 1.0
        modified_texts[file_path_] = text
    return document_meta, parsed_documents, modified_texts


def run():
    #############################Test#####################
    # test labels
    test_paths_by_label, test_labels_by_path = dl.get_labels(
        fc.read_file(gv.data_src_path + gv.test_label_file_name),
        gv.test_label_file_name)
    # test dataset processing
    test_document_meta, test_parsed_documents, test_modified_texts = tokenizer(required_files=test_labels_by_path)
    op.save_object(test_document_meta, gv.prj_src_path + "python_objects/test_document_meta")
    # op.save_object(test_parsed_documents, gv.prj_src_path + "python_objects/test_parsed_documents")

    stopwords_en = load_stop_words()
    # test dataset vectorize
    test_data_transformed, test_labels = hashing_vectorizer(data_dict=test_modified_texts,
                                                            label_dict=test_labels_by_path, lower_case=True,
                                                            stop_words=stopwords_en, n_features=2 ** 20)
    op.save_object(test_data_transformed, gv.prj_src_path + "python_objects/test_data_hash_transformed")
    # dict_vectorizer
    test_document_meta = op.load_object(gv.prj_src_path + "python_objects/test_document_meta")
    test_data_transformed, test_labels = dict_vectorizer(data_dict=test_document_meta,
                                                          label_dict=test_labels_by_path)
    op.save_object(test_data_transformed, gv.prj_src_path + "python_objects/test_data_transformed")
    op.save_object(test_labels, gv.prj_src_path + "python_objects/test_labels")

    #############################Val#####################
    # val labels
    val_paths_by_label, val_labels_by_path = dl.get_labels(fc.read_file(gv.data_src_path + gv.val_label_file_name),
                                                           gv.val_label_file_name)
    # val dataset processing
    val_document_meta, val_parsed_documents, val_modified_texts = tokenizer(required_files=val_labels_by_path)
    op.save_object(val_document_meta, gv.prj_src_path + "python_objects/val_document_meta")
    # op.save_object(val_parsed_documents, gv.prj_src_path + "python_objects/val_parsed_documents")
    # val dataset vectorize
    val_data_transformed, val_labels = hashing_vectorizer(data_dict=val_modified_texts,
                                                          label_dict=val_labels_by_path, lower_case=True,
                                                          stop_words=stopwords_en, n_features=2 ** 20)
    op.save_object(val_data_transformed, gv.prj_src_path + "python_objects/val_data_hash_transformed")
    # dict_vectorizer
    val_document_meta = op.load_object(gv.prj_src_path + "python_objects/val_document_meta")
    val_data_transformed, val_labels = dict_vectorizer(data_dict=val_document_meta, label_dict=val_labels_by_path)
    op.save_object(val_data_transformed, gv.prj_src_path + "python_objects/val_data_transformed")
    op.save_object(val_labels, gv.prj_src_path + "python_objects/val_labels")

    #############################Train#####################
    # train labels
    train_paths_by_label, train_labels_by_path = dl.get_labels(
        fc.read_file(gv.data_src_path + gv.train_label_file_name), gv.train_label_file_name)
    # train dataset processing
    train_document_meta, train_parsed_documents, train_modified_texts = tokenizer(required_files=train_labels_by_path)
    op.save_object(train_document_meta, gv.prj_src_path + "python_objects/train_document_meta")
    # op.save_object(train_parsed_documents, gv.prj_src_path + "python_objects/train_parsed_documents")
    # train dataset vectorize
    train_data_transformed, train_labels = hashing_vectorizer(data_dict=train_modified_texts,
                                                              label_dict=train_labels_by_path, lower_case=True,
                                                              stop_words=stopwords_en, n_features=2 ** 20)
    op.save_object(train_data_transformed, gv.prj_src_path + "python_objects/train_data_hash_transformed")
    # dict_vectorizer
    train_document_meta = op.load_object(gv.prj_src_path + "python_objects/train_document_meta")
    train_data_transformed, train_labels = dict_vectorizer(data_dict=train_document_meta,
                                                           label_dict=train_labels_by_path)
    op.save_object(train_data_transformed, gv.prj_src_path + "python_objects/train_data_transformed")
    op.save_object(train_labels, gv.prj_src_path + "python_objects/train_labels")


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

import time
import json
import spacy
import timer
import gensim
import logging as log
import file_collector as fc
import object_pickler as op
import global_variables as gv
import data_labeller as dl

from random import randrange
from sklearn.feature_extraction import DictVectorizer
from gensim.models.doc2vec import Doc2Vec

log.basicConfig(filename='data_processor.log', level=log.DEBUG, filemode="w")


def load_stop_words():
    with open(gv.prj_src_path + "data/stopwords-en.txt", "rt", encoding="utf-8-sig") as infile:
        stopwords_en = json.load(infile)["en"]
        return stopwords_en


def load_doc2vec_model():
    return Doc2Vec.load(gv.prj_src_path + "python_objects/doc2vec_model")


def preprocess_doc2vec(train_corpus_list, tokens_only=False):
    for i, tcd in enumerate(train_corpus_list):
        tokens = gensim.utils.simple_preprocess(tcd)
        tcd = tcd.strip()
        if len(tcd) is not 0:
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def generate_doc2vec_model(train_corpus):
    doc2vec_model = Doc2Vec(vector_size=300, min_count=2, epochs=50)
    doc2vec_model.build_vocab(train_corpus)
    doc2vec_model.train(train_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.iter)
    doc2vec_model.save(gv.prj_src_path + "python_objects/document_model.doc2vec")


def dict_vectorizer(data_dict, label_dict, test_data_dict, test_label_dict, val_data_dict, val_label_dict):
    labels = list()
    test_labels = list()
    val_labels = list()
    data = list()
    test_data = list()
    val_data = list()
    for file_path, value in data_dict.items():
        value = value.strip()
        if len(value) is not 0:
            labels.append(label_dict[file_path])
            data.append(value)

    for file_path, value in test_data_dict.items():
        value = value.strip()
        if len(value) is not 0:
            test_labels.append(test_label_dict[file_path])
            test_data.append(value)

    for file_path, value in val_data_dict.items():
        value = value.strip()
        if len(value) is not 0:
            val_labels.append(val_label_dict[file_path])
            val_data.append(value)

    dv = DictVectorizer(sparse=True)
    data_transformed = dv.fit_transform(data)
    test_data_transformed = dv.transform(test_data)
    val_data_transformed = dv.transform(val_data)
    return data_transformed, labels, test_data_transformed, test_labels, val_data_transformed, val_labels


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
    return document_meta, modified_texts


def run():
    # train labels
    label_start = time.time()
    log.info(("Get train labels: ", time.localtime(label_start)))
    train_paths_by_label, train_labels_by_path = dl.get_labels(
        fc.read_file(gv.data_src_path + gv.train_label_file_name), gv.train_label_file_name)
    log.debug("train_paths_by_label: " + str(len(train_paths_by_label)))
    log.debug("train_labels_by_path: " + str(len(train_labels_by_path)))
    timer.time_executed(label_start, "Get train labels")
    # test labels
    label_start = time.time()
    log.info(("Get test labels: ", time.localtime(label_start)))
    test_paths_by_label, test_labels_by_path = dl.get_labels(fc.read_file(gv.data_src_path + gv.test_label_file_name),
                                                             gv.test_label_file_name)
    log.debug("test_paths_by_label: " + str(len(test_paths_by_label)))
    log.debug("test_labels_by_path: " + str(len(test_labels_by_path)))
    timer.time_executed(label_start, "Get test labels")

    # val labels
    label_start = time.time()
    log.info(("Get val labels: ", time.localtime(label_start)))
    val_paths_by_label, val_labels_by_path = dl.get_labels(fc.read_file(gv.data_src_path + gv.val_label_file_name),
                                                           gv.val_label_file_name)
    log.debug("val_paths_by_label: " + str(len(val_paths_by_label)))
    log.debug("val_labels_by_path: " + str(len(val_labels_by_path)))
    timer.time_executed(label_start, "Get val labels")

    # train dataset processing
    process_start = time.time()
    log.info(("Process train data: ", time.localtime(process_start)))
    train_document_meta, train_modified_texts = tokenizer(required_files=train_labels_by_path)
    log.debug("train_document_meta: " + str(len(train_document_meta)))
    log.debug("train_modified_texts: " + str(len(train_modified_texts)))
    timer.time_executed(process_start, "Process train data")
    op.save_object(train_document_meta, gv.prj_src_path + "python_objects/train_document_meta")
    op.save_object(train_modified_texts, gv.prj_src_path + "python_objects/train_modified_texts")

    # test dataset processing
    process_start = time.time()
    log.info(("Process test data: ", time.localtime(process_start)))
    test_document_meta, test_modified_texts = tokenizer(required_files=test_labels_by_path)
    log.debug("test_document_meta: " + str(len(test_document_meta)))
    log.debug("test_modified_texts: " + str(len(test_modified_texts)))
    timer.time_executed(process_start, "Process test data")
    op.save_object(test_document_meta, gv.prj_src_path + "python_objects/test_document_meta")
    op.save_object(test_modified_texts, gv.prj_src_path + "python_objects/test_modified_texts")

    # val dataset processing
    process_start = time.time()
    log.info(("Process val data: ", time.localtime(process_start)))
    val_document_meta, val_modified_texts = tokenizer(required_files=val_labels_by_path)
    log.debug("val_document_meta: " + str(len(val_document_meta)))
    log.debug("val_modified_texts: " + str(len(val_modified_texts)))
    timer.time_executed(process_start, "Process val data")
    op.save_object(val_document_meta, gv.prj_src_path + "python_objects/val_document_meta")
    op.save_object(val_modified_texts, gv.prj_src_path + "python_objects/val_modified_texts")

    # stopwords_en = load_stop_words()

    # load document meta
    train_document_meta = op.load_object(gv.prj_src_path + "python_objects/train_document_meta")
    test_document_meta = op.load_object(gv.prj_src_path + "python_objects/test_document_meta")
    val_document_meta = op.load_object(gv.prj_src_path + "python_objects/val_document_meta")

    # dict_vectorizer
    process_start = time.time()
    log.info(("Dictvectorizer: ", time.localtime(process_start)))
    train_data_transformed, train_labels, test_data_transformed, test_labels, val_data_transformed, val_labels = dict_vectorizer(
        data_dict=train_document_meta, label_dict=train_labels_by_path, test_data_dict=test_document_meta,
        test_label_dict=test_labels_by_path, val_data_dict=val_document_meta, val_label_dict=val_labels_by_path)
    log.debug("train_data_transformed: " + str(len(train_data_transformed)))
    log.debug("train_labels: " + str(len(train_labels)))
    log.debug("test_data_transformed: " + str(len(test_data_transformed)))
    log.debug("test_labels: " + str(len(test_labels)))
    log.debug("val_data_transformed: " + str(len(val_data_transformed)))
    log.debug("val_labels: " + str(len(val_labels)))
    timer.time_executed(process_start, "Dictvectorizer")

    op.save_object(train_data_transformed, gv.prj_src_path + "python_objects/train_data_transformed")
    op.save_object(train_labels, gv.prj_src_path + "python_objects/train_labels")

    op.save_object(test_data_transformed, gv.prj_src_path + "python_objects/test_data_transformed")
    op.save_object(test_labels, gv.prj_src_path + "python_objects/test_labels")

    op.save_object(val_data_transformed, gv.prj_src_path + "python_objects/val_data_transformed")
    op.save_object(val_labels, gv.prj_src_path + "python_objects/val_labels")

    # load modified texts
    train_modified_texts = op.load_object(gv.prj_src_path + "python_objects/train_modified_texts")
    test_modified_texts = op.load_object(gv.prj_src_path + "python_objects/test_modified_texts")
    val_modified_texts = op.load_object(gv.prj_src_path + "python_objects/val_modified_texts")

    # generate preprocessed train corpus
    process_start = time.time()
    log.info(("Train corpus: ", time.localtime(process_start)))
    train_corpus_list = [tcd for tcd in train_modified_texts]
    train_corpus_preprocessed = preprocess_doc2vec(train_corpus_list)
    log.info("train_corpus size: " + str(len(train_corpus_list)))
    timer.time_executed(process_start, "Train corpus")

    # generate tokens only train corpus
    process_start = time.time()
    log.info(("Train corpus: ", time.localtime(process_start)))
    train_corpus_tokens_only = preprocess_doc2vec(train_corpus_list, tokens_only=True)
    timer.time_executed(process_start, "Train corpus")

    # generate tokens only Test corpus
    process_start = time.time()
    log.info(("Test corpus: ", time.localtime(process_start)))
    test_corpus_list = [tcd for tcd in test_modified_texts]
    test_corpus_tokens_only = preprocess_doc2vec(test_corpus_list, tokens_only=True)
    log.info("test_corpus size: " + str(len(test_corpus_list)))
    timer.time_executed(process_start, "Test corpus")

    # generate tokens only val corpus
    process_start = time.time()
    log.info(("Val corpus: ", time.localtime(process_start)))
    val_corpus_list = [tcd for tcd in val_modified_texts]
    val_corpus_tokens_only = preprocess_doc2vec(val_corpus_list, tokens_only=True)
    log.info("val_corpus size: " + str(len(val_corpus_list)))
    timer.time_executed(process_start, "Val corpus")

    # generate doc2vec model
    process_start = time.time()
    log.info(("Doc2Vec: ", time.localtime(process_start)))
    generate_doc2vec_model(train_corpus_preprocessed)
    timer.time_executed(process_start, "Doc2Vec")

    # load doc2vec model
    model = Doc2Vec.load(gv.prj_src_path + "python_objects/document_model.doc2vec")

    # get train vector from the doc2vec
    infer_vector_start = time.time()
    log.info(("Infer vector train: ", time.localtime(process_start)))
    train_vector = list(map(model.infer_vector, train_corpus_tokens_only))
    log.info("train_vector size: " + str(len(train_vector)))
    rand_index = randrange(len(train_vector))
    log.info("train_vector[" + str(rand_index) + "] feature size: " + str(len(train_vector[rand_index - 1])))
    log.info(train_vector[rand_index])
    timer.time_executed(infer_vector_start, "Infer vector train")
    op.save_object(train_vector, gv.prj_src_path + "python_objects/train_vector")

    # get test vector from the doc2vec
    infer_vector_start = time.time()
    log.info(("Infer vector test: ", time.localtime(process_start)))
    test_vector = list(map(model.infer_vector, test_corpus_tokens_only))
    log.info("test_vector size: " + str(len(test_vector)))
    rand_index = randrange(len(test_vector))
    log.info("test_vector[" + str(rand_index) + "] feature size: " + str(len(test_vector[rand_index - 1])))
    log.info(test_vector[rand_index])
    timer.time_executed(infer_vector_start, "Infer vector test")
    op.save_object(test_vector, gv.prj_src_path + "python_objects/test_vector")

    # get val vector from the doc2vec
    infer_vector_start = time.time()
    log.info(("Infer vector val: ", time.localtime(process_start)))
    val_vector = list(map(model.infer_vector, val_corpus_tokens_only))
    log.info("val_vector size: " + str(len(val_vector)))
    rand_index = randrange(len(val_vector))
    log.info("val_vector[" + str(rand_index) + "] feature size: " + str(len(val_vector[rand_index - 1])))
    log.info(val_vector[rand_index])
    timer.time_executed(infer_vector_start, "Infer vector val")
    op.save_object(val_vector, gv.prj_src_path + "python_objects/val_vector")


def main():
    run()


if __name__ == '__main__':
    start_time = time.time()
    log.info(("Data processor started: ", time.localtime(start_time)))
    try:
        main()
    except Exception as ex:
        log.exception(ex)
    timer.time_executed(start_time, "Data processor")

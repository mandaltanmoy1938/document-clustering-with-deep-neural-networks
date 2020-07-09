import logging as log
import time

import pyLDAvis
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel

import global_variables as gv
import object_pickler as op
import timer

log.basicConfig(filename='statistics.log', level=log.DEBUG, filemode="w")


def load_lda_model(num_topics, passes):
    return LdaModel.load("%spython_objects/document_model_%s_%s.lda" % (gv.prj_src_path, str(num_topics), str(passes)))


def load_lda_dictionary(dict_name):
    return Dictionary.load_from_text("%spython_objects/%s.dict" % (gv.prj_src_path, dict_name))


def visualize(corpus, dictionary, num_topics_passes_tuple_list):
    texts = corpus
    corpus = list(map(dictionary.doc2bow, corpus))
    model_list = dict()
    for num_topics_passes_tuple in num_topics_passes_tuple_list:
        model_list["%s_%s" % (str(num_topics_passes_tuple[0]), str(num_topics_passes_tuple[1]))] = load_lda_model(
            num_topics=num_topics_passes_tuple[0], passes=num_topics_passes_tuple[1])

    for key, model in model_list.items():
        ldavis_start = time.time()
        log.info("LDAVis_%s started: %s" % (key, time.localtime(ldavis_start)))

        log.info("[perplexity_%s] train data %s" % (key, model.log_perplexity(corpus)))

        coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        log.info("[perplexity_%s] train data %s" % (key, coherence_lda))

        ldavis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
        pyLDAvis.save_html(ldavis, "%sgenerated_plots/lda_%s.html" % (gv.prj_src_path, key))
        timer.time_executed(ldavis_start, "LDAVis_%s" % key)


def process_data(modified_texts):
    text_list = [tcd.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").replace("\t", " ").split(" ") for
                 key, tcd in modified_texts.items()]
    corpus = list()
    for text in text_list:
        token_list = [token.rstrip() for token in text.split(" ") if len(token.rstrip()) > 0]
        corpus.append(token_list)
    return corpus


def main():
    train_modified_texts = op.load_object(gv.prj_src_path + "python_objects/train_modified_texts")
    dictionary = load_lda_dictionary("dataset")
    corpus = process_data(train_modified_texts)
    num_topics_passes_tuple_list = [(20, 20), (50, 10), (30, 20)]

    visualize(corpus=corpus, dictionary=dictionary, num_topics_passes_tuple_list=num_topics_passes_tuple_list)


if __name__ == '__main__':
    start = time.time()
    log.info(("Document clustering started: ", time.localtime(start)))
    try:
        main()
    except Exception as ex:
        log.exception(ex)
    timer.time_executed(start, "Document clustering")

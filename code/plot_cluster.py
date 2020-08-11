import time
import timer
import pandas as pd
import logging as log
import object_pickler as op
import global_variables as gv
import graph_generator as gg

from sklearn.manifold import TSNE

log.basicConfig(filename='plot_cluster.log', level=log.DEBUG, filemode="w")


def load_pickle(filename):
    # filename without extension
    return op.load_object(gv.prj_src_path + "python_objects/" + filename)


def run():
    predicted_label = {"unsupervised": ["KMeans_"],
                       "supervised": ["LogisticRegression_", "SVC_linear_", "SVC_poly_", "SVC_rbf_"]}
    target_names = [gv.label_name[i] for i in gv.translation_rev]

    # # dimension reduction dictvectorizer
    # test_transformed = load_pickle("test_data_transformed")
    # test_transformed_embedded = TSNE(n_components=2).fit_transform(test_transformed)
    # op.save_object(test_transformed_embedded, gv.prj_src_path + "python_objects/test_2d_data_transformed")
    #
    # # dimension reduction doc2vec
    # test_vector = load_pickle("test_vector")
    # test_vector_embedded = TSNE(n_components=2).fit_transform(test_vector)
    # op.save_object(test_vector_embedded, gv.prj_src_path + "python_objects/test_2d_data_vector")

    test_embedded_dictvectorizer = load_pickle("test_2d_data_transformed")
    test_embedded_doc2vec = load_pickle("test_2d_data_vector")
    fig_num = 0
    df_dictvectorizer = pd.DataFrame(test_embedded_dictvectorizer, columns=["x", "y"])
    for algo in predicted_label["supervised"]:
        log.debug("plotting: " + algo + "test_dictvectorizer_predict")
        fig_num += 1
        predict = load_pickle(algo + "test_data_transformed_predict")
        df_dictvectorizer[algo + "prediction"] = [target_names[p] for p in predict]
        gg.plot_cluster(title=algo + "test_dictvectorizer_predict", data=df_dictvectorizer, pad=30,
                        plot_name=gv.prj_src_path + "generated_plots/" + algo + "test_dictvectorizer_predict",
                        fig_num=fig_num, l_col=3, hue=algo + "prediction")

    # for algo in predicted_label["unsupervised"]:
    #     fig_num += 1
    #     predict = load_pickle(algo + "test_data_transformed_predict")
    #     df_dictvectorizer[algo + "prediction"] = predict
    #     gg.plot_cluster(title=algo + "test_dictvectorizer_predict", data=df_dictvectorizer, pad=30,
    #                     plot_name=gv.prj_src_path + "generated_plots/" + algo + "test_dictvectorizer_predict",
    #                     fig_num=fig_num, l_col=2, hue=algo + "prediction")

    labels = load_pickle("test_labels")
    labels = [gv.translation[x] for x in labels]
    log.debug(df_dictvectorizer.shape)
    df_dictvectorizer["ground_truth"] = [target_names[label] for label in labels]
    fig_num += 1
    gg.plot_cluster(title="Ground truth Dictvectorizer", data=df_dictvectorizer, pad=30,
                    plot_name=gv.prj_src_path + "generated_plots/ground_truth_dictvectorizer", fig_num=fig_num, l_col=3,
                    hue="ground_truth")

    df_doc2vec = pd.DataFrame(test_embedded_doc2vec, columns=["x", "y"])
    for algo in predicted_label["supervised"]:
        log.debug("plotting: " + algo + "test_doc2vec_predict")
        fig_num += 1
        predict = load_pickle(algo + "test_vector_predict")
        df_doc2vec[algo + "prediction"] = [target_names[p] for p in predict]
        gg.plot_cluster(title=algo + "test_doc2vec_predict", data=df_doc2vec, pad=30,
                        plot_name=gv.prj_src_path + "generated_plots/" + algo + "test_doc2vec_predict",
                        fig_num=fig_num, l_col=3, hue=algo + "prediction")

    # for algo in predicted_label["unsupervised"]:
    #     fig_num += 1
    #     predict = load_pickle(algo + "est_vector_predict")
    #     df_doc2vec[algo + "prediction"] = predict
    #     gg.plot_cluster(title=algo + "test_doc2vec_predict", data=df_doc2vec, pad=30,
    #                     plot_name=gv.prj_src_path + "generated_plots/" + algo + "test_doc2vec_predict",
    #                     fig_num=fig_num, l_col=2, hue=algo + "prediction")

    labels = load_pickle("test_labels")
    labels = [gv.translation[x] for x in labels]
    df_doc2vec["ground_truth"] = [target_names[label] for label in labels]
    fig_num += 1
    gg.plot_cluster(title="Ground truth Doc2Vec", data=df_doc2vec, pad=30,
                    plot_name=gv.prj_src_path + "generated_plots/ground_truth_doc2vec", fig_num=fig_num, l_col=3,
                    hue="ground_truth")


def main():
    run()


if __name__ == '__main__':
    start = time.time()
    log.info(("PLot cluster started: ", time.localtime(start)))
    try:
        main()
    except Exception as ex:
        log.exception(ex)
    timer.time_executed(start, "Plot cluster")

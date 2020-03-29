import time
import timer
import random
import pandas as pd
import logging as log
import object_pickler as op
import global_variables as gv
import graph_generator as gg

from sklearn.manifold import TSNE

log.basicConfig(filename='plot_cluster.log', level=log.DEBUG, filemode="w")


def loadPickle(filename):
    # filename without extension
    return op.load_object(gv.prj_src_path + "python_objects/" + filename)


def run():
    predicted_label = {"unsupervised": ["KMeans_"],
                       "supervised": ["LogisticRegression_", "NB_", "SVC_"]}
    target_names = [gv.label_name[i] for i in gv.translation_rev]

    # test_transformed = loadPickle("test_data_transformed")
    # test_embedded = TSNE(n_components=2).fit_transform(test_transformed)
    # op.save_object(test_embedded, gv.prj_src_path + "python_objects/test_2d_data_transformed")
    test_embedded = loadPickle("test_2d_data_transformed")
    fig_num = 0
    df = pd.DataFrame(test_embedded, columns=["x", "y"])
    for algo in predicted_label["supervised"]:
        fig_num += 1
        predict = loadPickle(algo + "test_data_transformed_predict")
        df[algo + "prediction"] = [target_names[p] for p in predict]
        gg.plot_cluster(title=algo, data=df, pad=30, plot_name=gv.prj_src_path + "generated_plots/" + algo,
                        fig_num=fig_num, l_col=3, hue=algo + "prediction")

    for algo in predicted_label["unsupervised"]:
        fig_num += 1
        predict = loadPickle(algo + "test_data_transformed_predict")
        df[algo + "prediction"] = predict
        gg.plot_cluster(title=algo, data=df, pad=30, plot_name=gv.prj_src_path + "generated_plots/" + algo,
                        fig_num=fig_num, l_col=2, hue=algo + "prediction")

        labels = loadPickle("test_labels")
        labels = [gv.translation[x] for x in labels]
        df["ground_truth"] = [target_names[l] for l in labels]
        fig_num += 1
        gg.plot_cluster(title="Ground truth", data=df, pad=30,
                        plot_name=gv.prj_src_path + "generated_plots/ground_truth", fig_num=fig_num, l_col=3,
                        hue="ground_truth")


def main():
    run()
    # df1 = pd.DataFrame([[random.randint(10000, 10000000) for j in range(2)] for i in range(15)], columns=["x", "y"])
    # df2 = pd.DataFrame([[random.randint(10000, 100000) for j in range(2)] for i in range(15)], columns=["x", "y"])
    # df3 = pd.DataFrame([[random.randint(100000, 1000000) for j in range(2)] for i in range(15)], columns=["x", "y"])
    # df4 = pd.DataFrame([[random.randint(1000, 1000000) for j in range(2)] for i in range(15)], columns=["x", "y"])
    # labels = [i for i in range(15)]
    # target_names = [gv.label_name[i] for i in gv.translation_rev]
    # df1['label_a'] = [target_names[l] for l in labels]
    # df2['label_b'] = [target_names[l] for l in labels]
    # df3['label_c'] = [target_names[l] for l in labels]
    # df4['label_d'] = [target_names[l] for l in labels]
    # gg.plot_cluster(title="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", data=df1, pad=30,
    #                 plot_name="test_a", fig_num=1, l_col=3, hue="label_a")
    # gg.plot_cluster(title="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", data=df2, pad=30,
    #                 plot_name="test_b", fig_num=2, l_col=3, hue="label_b")
    # gg.plot_cluster(title="ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc", data=df3, pad=30,
    #                 plot_name="test_c", fig_num=3, l_col=3, hue="label_c")
    # gg.plot_cluster(title="ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", data=df4, pad=30,
    #                 plot_name="test_d", fig_num=4, l_col=3, hue="label_d")


if __name__ == '__main__':
    start = time.time()
    log.info(("PLot cluster started: ", time.localtime(start)))
    try:
        main()
    except Exception as ex:
        log.exception(ex)
    timer.time_executed(start, "Plot cluster")

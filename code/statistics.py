import logging as log
import time

import pandas as pd

import data_labeller as dl
import file_collector as fc
import global_variables as gv
import graph_generator as gg
import object_pickler as op
import timer

log.basicConfig(filename='statistics.log', level=log.DEBUG, filemode="w")


def get_file_content_meta(lines):
    empty_line_count = 0
    word_count = 0
    space_count = 0
    total_character_count = 0
    for line in lines:
        line = line.strip('\n').strip()
        if len(line) == 0:
            empty_line_count += 1
        else:
            word_count += len(line.split())
            space_count += line.count(' ')
            total_character_count += len(line)
    return len(lines), empty_line_count, word_count, space_count, total_character_count


def get_all_file_content_meta(file_paths, src_path, required_files):
    empty_file_count = 0
    empty_file_count_by_class = dict()
    log.info(("Total number of file:", len(file_paths)))
    all_file_content_meta = dict(('{}'.format(file_path.replace(src_path, "").replace("\\", "/")),
                                  {"total_line_count": 0, "empty_line_count": 0, "word_count": 0, "space_count": 0,
                                   "total_character_count": 0}) for file_path in file_paths)
    for file_path in file_paths:
        file_path_ = file_path.replace(src_path, "").replace("\\", "/")
        if file_path_ in required_files:
            all_file_content_meta[file_path_]["total_line_count"], \
            all_file_content_meta[file_path_]["empty_line_count"], \
            all_file_content_meta[file_path_]["word_count"], \
            all_file_content_meta[file_path_]["space_count"], \
            all_file_content_meta[file_path_]["total_character_count"] = get_file_content_meta(fc.read_file(file_path))

            if all_file_content_meta[file_path_]["total_line_count"] == all_file_content_meta[file_path_] \
                    ["empty_line_count"]:
                del all_file_content_meta[file_path_]
                empty_file_count_by_class.setdefault(required_files[file_path_], 0)
                empty_file_count_by_class[required_files[file_path_]] += 1
                empty_file_count += 1
    return all_file_content_meta, empty_file_count, empty_file_count_by_class


def get_all_label_content_meta(meta_dict, label_dict_list, empty_document_class):
    label_content_meta = dict(('{}'.format(k),
                               {"total_line_count": 0, "empty_line_count": 0, "word_count": 0, "space_count": 0,
                                "total_character_count": 0}) for k in range(16))
    for label_dict in label_dict_list:
        for label, paths in label_dict.items():
            label_content_meta[label]['label'] = gv.label_name[label]
            label_content_meta[label]['number_of_documents'] = len(paths)
            label_content_meta[label]['number_of_empty_documents'] = empty_document_class[label]
            for path in paths:
                if path in meta_dict:
                    label_content_meta[label]['total_line_count'] += meta_dict[path]['total_line_count']
                    label_content_meta[label]['empty_line_count'] += meta_dict[path]['empty_line_count']
                    label_content_meta[label]['word_count'] += meta_dict[path]['word_count']
                    label_content_meta[label]['space_count'] += meta_dict[path]['space_count']
                    label_content_meta[label]['total_character_count'] += meta_dict[path]['total_character_count']
    return label_content_meta


def plot_statistics(label_content_meta_object_name, labels_by_path, empty_file_count_object_name, data_type, fig_num=0):
    # load objects
    label_content_meta = op.load_object("%spython_objects/%s" % (gv.prj_src_path, label_content_meta_object_name))
    empty_file_count = op.load_object("%spython_objects/%s" % (gv.prj_src_path, empty_file_count_object_name))

    label_content_meta_pd = pd.DataFrame.from_dict(label_content_meta, orient='index')
    label_content_meta_pd["class_avg_line"] = label_content_meta_pd["total_line_count"] \
                                              / label_content_meta_pd["number_of_documents"]
    label_content_meta_pd["class_avg_word"] = label_content_meta_pd["word_count"] \
                                              / label_content_meta_pd["number_of_documents"]
    label_content_meta_pd["class_avg_line_wo_empty_documents"] = \
        label_content_meta_pd["total_line_count"] / (label_content_meta_pd["number_of_documents"]
                                                     - label_content_meta_pd["number_of_empty_documents"])
    label_content_meta_pd["class_avg_word_wo_empty_documents"] = \
        label_content_meta_pd["word_count"] / (label_content_meta_pd["number_of_documents"]
                                               - label_content_meta_pd["number_of_empty_documents"])
    label_content_meta_pd = label_content_meta_pd.round(
        {"class_avg_line": 2, "class_avg_word": 2, "class_avg_line_wo_empty_documents": 2,
         "class_avg_word_wo_empty_documents": 2, })
    fig_num += 1
    gg.plot_chart(y="number_of_documents", y_label="number of documents",
                  title="Number of documents vs Classes for\n" + str(
                      len(labels_by_path)) + " val documents\nincluding " + str(
                      empty_file_count) + "documents", kind="bar", data=label_content_meta_pd, pad=40,
                  plot_name="%sgenerated_plots/%s_document_number" % (gv.prj_src_path, data_type), fig_num=fig_num)
    fig_num += 1
    gg.plot_chart(y="number_of_empty_documents", y_label="number of empty documents",
                  title="Number of empty documents vs Classes for\n" + str(
                      len(labels_by_path)) + " val documents\nincluding " + str(
                      empty_file_count) + "documents", kind="bar", data=label_content_meta_pd, pad=40,
                  plot_name="%sgenerated_plots/%s_empty_document_number" % (gv.prj_src_path, data_type),
                  fig_num=fig_num)
    fig_num += 1
    gg.plot_chart(y="total_line_count", y_label="Total number of lines", title="Total number of lines vs Classes",
                  kind="bar", data=label_content_meta_pd, pad=20,
                  plot_name="%sgenerated_plots/%s_total_line_count" % (gv.prj_src_path, data_type),
                  fig_num=fig_num)
    fig_num += 1
    gg.plot_chart(y="word_count", y_label="Total number of words", title="Total number of words vs Classes", kind="bar",
                  data=label_content_meta_pd, pad=20,
                  plot_name="%sgenerated_plots/%s_word_count" % (gv.prj_src_path, data_type),
                  fig_num=fig_num)
    fig_num += 1
    gg.plot_chart(y="class_avg_line", y_label="Average number of lines", title="Average number of lines vs Classes",
                  kind="bar", data=label_content_meta_pd, pad=20,
                  plot_name="%sgenerated_plots/%s_avg_line_count" % (gv.prj_src_path, data_type), fig_num=fig_num)
    fig_num += 1
    gg.plot_chart(y="class_avg_word", y_label="Average number of words", title="Average number of words vs Classes",
                  kind="bar", data=label_content_meta_pd, pad=20,
                  plot_name="%sgenerated_plots/%s_avg_word_count" % (gv.prj_src_path, data_type), fig_num=fig_num)
    fig_num += 1
    gg.plot_chart(y="class_avg_line_wo_empty_documents", y_label="Average number of lines",
                  title="Average number of lines vs Classes\n(excluding empty documents)", kind="bar",
                  data=label_content_meta_pd, pad=30,
                  plot_name="%sgenerated_plots/%s_avg_line_count_wo_empty_documents" % (gv.prj_src_path, data_type),
                  fig_num=fig_num)
    fig_num += 1
    gg.plot_chart(y="class_avg_word_wo_empty_documents", y_label="Average number of words",
                  title="Average number of words vs Classes\n(excluding empty documents)", kind="bar",
                  data=label_content_meta_pd, pad=30,
                  plot_name="%sgenerated_plots/%s_avg_word_count_wo_empty_documents" % (gv.prj_src_path, data_type),
                  fig_num=fig_num)
    return fig_num


def main():
    # test
    test_paths_by_label, test_labels_by_path = dl.get_labels_w_3(
        fc.read_file(gv.data_src_path + gv.test_label_file_name), gv.test_label_file_name)
    file_paths = fc.get_all_files_from_directory(gv.data_src_path)
    test_meta_dict, test_empty_file_count, test_empty_file_count_by_class = \
        get_all_file_content_meta(file_paths, gv.data_src_path, test_labels_by_path)
    op.save_object(test_empty_file_count, gv.prj_src_path + "python_objects/test_empty_file_count")
    test_label_content_meta = get_all_label_content_meta(test_meta_dict, [test_paths_by_label],
                                                         test_empty_file_count_by_class)
    op.save_object(test_label_content_meta, gv.prj_src_path + "python_objects/test_label_content_meta")
    fig_num = plot_statistics(label_content_meta_object_name="test_label_content_meta",
                              labels_by_path=test_labels_by_path, empty_file_count_object_name="test_empty_file_count",
                              data_type="test")

    # train
    train_paths_by_label, train_labels_by_path = dl.get_labels_w_3(
        fc.read_file(gv.data_src_path + gv.train_label_file_name), gv.train_label_file_name)
    train_meta_dict, train_empty_file_count, train_empty_file_count_by_class = \
        get_all_file_content_meta(file_paths, gv.data_src_path, train_labels_by_path)
    op.save_object(train_empty_file_count, gv.prj_src_path + "python_objects/train_empty_file_count")
    train_label_content_meta = get_all_label_content_meta(train_meta_dict, [train_paths_by_label],
                                                          train_empty_file_count_by_class)
    op.save_object(train_label_content_meta, gv.prj_src_path + "python_objects/train_label_content_meta")
    fig_num = plot_statistics(label_content_meta_object_name="train_label_content_meta",
                              labels_by_path=train_labels_by_path,
                              empty_file_count_object_name="train_empty_file_count", data_type="train", fig_num=fig_num)

    # val
    val_paths_by_label, val_labels_by_path = dl.get_labels_w_3(fc.read_file(gv.data_src_path + gv.val_label_file_name),
                                                               gv.val_label_file_name)
    val_meta_dict, val_empty_file_count, val_empty_file_count_by_class = \
        get_all_file_content_meta(file_paths, gv.data_src_path, val_labels_by_path)
    op.save_object(val_empty_file_count, gv.prj_src_path + "python_objects/val_empty_file_count")
    val_label_content_meta = get_all_label_content_meta(val_meta_dict, [val_paths_by_label],
                                                        val_empty_file_count_by_class)
    op.save_object(val_label_content_meta, gv.prj_src_path + "python_objects/val_label_content_meta")
    plot_statistics(label_content_meta_object_name="val_label_content_meta", labels_by_path=val_labels_by_path,
                    empty_file_count_object_name="val_empty_file_count", data_type="val", fig_num=fig_num)

    log.error(("Number of error file:", gv.error_file_count))


if __name__ == '__main__':
    start_time = time.time()
    log.info(("Statistics started: ", time.localtime(start_time)))
    try:
        main()
    except Exception as ex:
        log.exception(ex)
    timer.time_executed(start_time, "Statistics")

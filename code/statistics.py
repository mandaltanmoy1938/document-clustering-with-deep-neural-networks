import time
import pandas as pd
import logging as log
import plot_generator as pg
import file_collector as fc
import global_variables as gv
import data_labeller as dl

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


def get_all_file_content_meta(file_paths, src_path, skipped_files, required_files):
    empty_file_count = 0
    log.info(("Total number of file:", len(file_paths)))
    all_file_content_meta = dict(('{}'.format(file_path.replace(src_path, "").replace("\\", "/")),
                                  {"total_line_count": 0, "empty_line_count": 0, "word_count": 0, "space_count": 0,
                                   "total_character_count": 0}) for file_path in file_paths)
    for file_path in file_paths:
        file_path_ = file_path.replace(src_path, "").replace("\\", "/")
        if file_path_ not in skipped_files and file_path_ in required_files:
            all_file_content_meta[file_path_]["total_line_count"], \
            all_file_content_meta[file_path_]["empty_line_count"], \
            all_file_content_meta[file_path_]["word_count"], \
            all_file_content_meta[file_path_]["space_count"], \
            all_file_content_meta[file_path_]["total_character_count"] = get_file_content_meta(fc.read_file(file_path))

            if all_file_content_meta[file_path_]["total_line_count"] == all_file_content_meta[file_path_] \
                    ["empty_line_count"]:
                del all_file_content_meta[file_path_]
                empty_file_count += 1
    return all_file_content_meta, empty_file_count


def get_all_label_content_meta(meta_dict, label_dict_list):

    label_content_meta = dict(('{}'.format(k),
                               {"total_line_count": 0, "empty_line_count": 0, "word_count": 0, "space_count": 0,
                                "total_character_count": 0}) for k in range(16))
    for label_dict in label_dict_list:
        for label, paths in label_dict.items():
            label_content_meta[label]['label'] = gv.label_name[label]
            label_content_meta[label]['number_of_documents'] = len(paths)
            for path in paths:
                if path in meta_dict:
                    label_content_meta[label]['total_line_count'] += meta_dict[path]['total_line_count']
                    label_content_meta[label]['empty_line_count'] += meta_dict[path]['empty_line_count']
                    label_content_meta[label]['word_count'] += meta_dict[path]['word_count']
                    label_content_meta[label]['space_count'] += meta_dict[path]['space_count']
                    label_content_meta[label]['total_character_count'] += meta_dict[path]['total_character_count']
                # else:
                # log.debug(("mismatch path wth meta dict: ", path))
    return label_content_meta


def main():
    test_paths_by_label, test_labels_by_path = dl.getpy_labels(fc.read_file(gv.data_src_path + gv.test_label_file_name),
                                                             gv.test_label_file_name)
    log.info(("number of handwritten files in test: ", len(test_paths_by_label["3"])))

    train_paths_by_label, train_labels_by_path = dl.get_labels(fc.read_file(gv.data_src_path + gv.train_label_file_name),
                                                               gv.train_label_file_name)
    log.info(("number of handwritten files in train: ", len(train_paths_by_label["3"])))

    val_paths_by_label, val_labels_by_path = dl.get_labels(fc.read_file(gv.data_src_path + gv.val_label_file_name),
                                                           gv.val_label_file_name)
    log.info(("number of handwritten files in val: ", len(val_paths_by_label["3"])))

    file_paths = fc.get_all_files_from_directory(gv.data_src_path)
    test_meta_dict, test_empty_file_count = get_all_file_content_meta(file_paths, gv.data_src_path,
                                                                      test_paths_by_label["3"], test_labels_by_path)
    test_label_content_meta = get_all_label_content_meta(test_meta_dict, [test_paths_by_label])
    del test_label_content_meta["3"]
    test_label_content_meta_pd = pd.DataFrame.from_dict(test_label_content_meta, orient='index')
    test_label_content_meta_pd["class_avg_line"] = test_label_content_meta_pd["total_line_count"] / \
                                                   test_label_content_meta_pd["number_of_documents"]
    test_label_content_meta_pd["class_avg_word"] = test_label_content_meta_pd["word_count"] / \
                                                   test_label_content_meta_pd["number_of_documents"]
    test_label_content_meta_pd = test_label_content_meta_pd.round({"class_avg_line": 0, "class_avg_word": 0})
    pg.plot_chart(y="total_line_count", y_label="Total number of lines",
                  title="Total number of lines vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
                  kind="bar", data=test_label_content_meta_pd, pad=30, plot_name="test_total_line_count", fig_num=1)
    pg.plot_chart(y="word_count", y_label="Total number of words",
                  title="Total number of words vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
                  kind="bar", data=test_label_content_meta_pd, pad=30, plot_name="test_word_count", fig_num=2)
    pg.plot_chart(y="class_avg_line", y_label="Average number of lines",
                  title="Average number of lines vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
                  kind="bar", data=test_label_content_meta_pd, pad=30, plot_name="test_avg_line_count", fig_num=3)
    pg.plot_chart(y="class_avg_word", y_label="Average number of words",
                  title="Average number of words vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
                  kind="bar", data=test_label_content_meta_pd, pad=30, plot_name="test_avg_word_count", fig_num=4)

    train_meta_dict, train_empty_file_count = get_all_file_content_meta(file_paths, gv.data_src_path,
                                                                        train_paths_by_label["3"], train_labels_by_path)
    train_label_content_meta = get_all_label_content_meta(train_meta_dict, [train_paths_by_label])
    del train_label_content_meta["3"]
    train_label_content_meta_pd = pd.DataFrame.from_dict(train_label_content_meta, orient='index')
    train_label_content_meta_pd["class_avg_line"] = train_label_content_meta_pd["total_line_count"] / \
                                                    train_label_content_meta_pd["number_of_documents"]
    train_label_content_meta_pd["class_avg_word"] = train_label_content_meta_pd["word_count"] / \
                                                    train_label_content_meta_pd["number_of_documents"]
    train_label_content_meta_pd = train_label_content_meta_pd.round({"class_avg_line": 0, "class_avg_word": 0})
    pg.plot_chart(y="total_line_count", y_label="Total number of lines",
                  title="Total number of lines vs Classes for\n" + str(len(train_labels_by_path)) + " train documents",
                  kind="bar", data=train_label_content_meta_pd, pad=30, plot_name="train_total_line_count", fig_num=5)
    pg.plot_chart(y="word_count", y_label="Total number of words",
                  title="Total number of words vs Classes for\n" + str(len(train_labels_by_path)) + " train documents",
                  kind="bar", data=train_label_content_meta_pd, pad=30, plot_name="train_word_count", fig_num=6)
    pg.plot_chart(y="class_avg_line", y_label="Average number of lines",
                  title="Average number of lines vs Classes for\n" + str(
                      len(train_labels_by_path)) + " train documents", kind="bar", data=train_label_content_meta_pd,
                  pad=30, plot_name="train_avg_line_count", fig_num=7)
    pg.plot_chart(y="class_avg_word", y_label="Average number of words",
                  title="Average number of words vs Classes for\n" + str(
                      len(train_labels_by_path)) + " train documents", kind="bar", data=train_label_content_meta_pd,
                  pad=30, plot_name="train_avg_word_count", fig_num=8)

    val_meta_dict, val_empty_file_count = get_all_file_content_meta(file_paths, gv.data_src_path, val_paths_by_label["3"],
                                                                    val_labels_by_path)
    val_label_content_meta = get_all_label_content_meta(val_meta_dict, [val_paths_by_label])
    del val_label_content_meta["3"]
    val_label_content_meta_pd = pd.DataFrame.from_dict(val_label_content_meta, orient='index')
    val_label_content_meta_pd["class_avg_line"] = val_label_content_meta_pd["total_line_count"] / \
                                                  val_label_content_meta_pd["number_of_documents"]
    val_label_content_meta_pd["class_avg_word"] = val_label_content_meta_pd["word_count"] / val_label_content_meta_pd[
        "number_of_documents"]
    val_label_content_meta_pd = val_label_content_meta_pd.round({"class_avg_line": 0, "class_avg_word": 0})
    pg.plot_chart(y="total_line_count", y_label="Total number of lines",
                  title="Total number of lines vs Classes for " + str(len(val_labels_by_path)) + " val documents",
                  kind="bar", data=val_label_content_meta_pd, pad=30, plot_name="val_total_line_count", fig_num=9)
    pg.plot_chart(y="word_count", y_label="Total number of words",
                  title="Total number of words vs Classes for " + str(len(val_labels_by_path)) + " val documents",
                  kind="bar", data=val_label_content_meta_pd, pad=30, plot_name="val_word_count", fig_num=10)
    pg.plot_chart(y="class_avg_line", y_label="Average number of lines",
                  title="Average number of lines vs Classes for " + str(len(val_labels_by_path)) + " val documents",
                  kind="bar", data=val_label_content_meta_pd, pad=30, plot_name="val_avg_line_count", fig_num=11)
    pg.plot_chart(y="class_avg_word", y_label="Average number of words",
                  title="Average number of words vs Classes for " + str(len(val_labels_by_path)) + " val documents",
                  kind="bar", data=val_label_content_meta_pd, pad=30, plot_name="val_avg_word_count", fig_num=12)
    # plot_chart(all_label_content_meta)

    log.error(("Number of error file:", gv.error_file_count))


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

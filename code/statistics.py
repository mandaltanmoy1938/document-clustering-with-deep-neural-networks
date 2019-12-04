import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import logging as log
import matplotlib.pyplot as plt

error_file_count = 0
log.basicConfig(filename='statistics.log', level=log.DEBUG, filemode="w")


def read_file(file_name):
    global error_file_count
    lines = []
    with open(file_name, 'rt', encoding="utf-8") as f:
        try:
            lines = f.readlines()
        except Exception as e:
            error_file_count += 1
            log.warning(("Error file: ", file_name))
            log.error(e)
    return lines


def get_labels(lines, file_path):
    paths_by_label = dict(('{}'.format(k), []) for k in range(16))
    labels_by_path = dict()
    log.info(("get labels from ", file_path))
    for index, line in enumerate(lines):
        label = line.strip('\n').split(" ")
        if len(label) == 2:
            paths_by_label[label[1]].append(label[0])
            if label[1] is not "3":
                labels_by_path[label[0]] = label[1]
        else:
            log.error((str(index + 1) + " : " + str(line)))
    return paths_by_label, labels_by_path


def get_all_files_from_directory(dir_path):
    data_paths = list()
    for path, sub_dirs, files in os.walk(dir_path):
        for name in files:
            if name not in ["text_test.txt", "text_train.txt", "text_val.txt"]:
                data_paths.append(os.path.join(path, name))
    return data_paths


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


def get_all_file_content_meta(files_path, src_path, skipped_files, required_files):
    empty_file_count = 0
    log.info(("Total number of file:", len(files_path)))
    all_file_content_meta = dict(('{}'.format(file_path.replace(src_path, "").replace("\\", "/")),
                                  {"total_line_count": 0, "empty_line_count": 0, "word_count": 0, "space_count": 0,
                                   "total_character_count": 0}) for file_path in files_path)
    for file_path in files_path:
        file_path_ = file_path.replace(src_path, "").replace("\\", "/")
        if file_path_ not in skipped_files and file_path_ in required_files:
            all_file_content_meta[file_path_]["total_line_count"], \
            all_file_content_meta[file_path_]["empty_line_count"], \
            all_file_content_meta[file_path_]["word_count"], \
            all_file_content_meta[file_path_]["space_count"], \
            all_file_content_meta[file_path_]["total_character_count"] = get_file_content_meta(read_file(file_path))

            if all_file_content_meta[file_path_]["total_line_count"] == all_file_content_meta[file_path_] \
                    ["empty_line_count"]:
                del all_file_content_meta[file_path_]
                empty_file_count += 1
    return all_file_content_meta, empty_file_count


def get_all_label_content_meta(meta_dict, label_dict_list):
    label_name = {"0": "Letter", "1": "Form", "2": "Email", "3": "", "4": "Advertisement", "5": "Scientific report",
                  "6": "Scientific publication", "7": "Specification", "8": "File folder", "9": "News article",
                  "10": "Budget", "11": "Invoice", "12": "Presentation", "13": "Questionnaire", "14": "Resume",
                  "15": "Memo"}
    label_content_meta = dict(('{}'.format(k),
                               {"total_line_count": 0, "empty_line_count": 0, "word_count": 0, "space_count": 0,
                                "total_character_count": 0}) for k in range(16))
    for label_dict in label_dict_list:
        for label, paths in label_dict.items():
            label_content_meta[label]['label'] = label_name[label]
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


def plot_chart(all_label_content_meta):
    labels = ['{}'.format(k) for k in range(16)]

    total_line_counts = [v['total_line_count'] for l, v in all_label_content_meta.items()]
    empty_line_counts = [v['empty_line_count'] for l, v in all_label_content_meta.items()]
    draw_dual_grouped_bar_chart(labels, total_line_counts, empty_line_counts)

    word_counts = [v['word_count'] for l, v in all_label_content_meta.items()]
    draw_bar_chart(labels, word_counts)

    space_counts = [v['space_count'] for l, v in all_label_content_meta.items()]
    total_character_counts = [v['total_character_count'] for l, v in all_label_content_meta.items()]
    draw_dual_grouped_bar_chart(labels, total_character_counts, space_counts)


def draw_dual_grouped_bar_chart(labels, bar1, bar2):
    # set width of bar
    barWidth = 0.5
    # Set position of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, bar1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Total number of lines')
    plt.bar(r2, bar2, color='#557f2d', width=barWidth, edgecolor='white', label='Number of empty lines')

    # Add xticks on the middle of the group bars
    plt.xlabel('labels', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bar1))], labels)

    for i, v in enumerate(r1):
        if bar1[i] != 0:
            plt.text(x=r1[i] - 0.25, y=bar1[i] + 0.5, s=bar1[i], size=10, rotation='vertical')
        if bar2[i] != 0:
            plt.text(x=r2[i] - 0.25, y=bar2[i] + 0.5, s=bar2[i], size=10, rotation='vertical')

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def draw_bar_chart(labels, bar1):
    # set width of bar
    barWidth = 0.5
    # Set position of bar on X axis
    r1 = np.arange(len(labels))

    plt.bar(r1, bar1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Total number of lines')  #

    # Add xticks on the middle of the group bars
    plt.xlabel('labels', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bar1))], labels)

    for i, v in enumerate(r1):
        if bar1[i] != 0:
            plt.text(x=r1[i] - 0.25, y=bar1[i] + 0.5, s=bar1[i], size=10, rotation='vertical')

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def plot_chart(y, y_label, title, kind, data, pad, plot_name):
    sns_plot = sns.catplot(x="label", y=y, kind=kind, data=data)
    sns_plot.set_xticklabels(rotation=45, ha='right')
    sns_plot.set_axis_labels("Classes", y_label)
    for index, row in data.iterrows():
        if int(index) > 3:
            index = int(index) - 1
        sns_plot.ax.text(float(index) - 0.25, row[y], row[y], rotation=45)
    plt.title(title, pad=pad)
    sns_plot.savefig(plot_name + ".png")
    plt.show()


def main():
    src_path = "G:\\UB\\5th Sem\\Individual Project\\Data\\QS-OCR-Large\\"
    test_label_file_name = "text_test.txt"
    train_label_file_name = "text_train.txt"
    val_label_file_name = "text_val.txt"

    test_paths_by_label, test_labels_by_path = get_labels(read_file(src_path + test_label_file_name),
                                                          test_label_file_name)
    log.info(("number of handwritten files in test: ", len(test_paths_by_label["3"])))

    train_paths_by_label, train_labels_by_path = get_labels(read_file(src_path + train_label_file_name),
                                                            train_label_file_name)
    log.info(("number of handwritten files in train: ", len(train_paths_by_label["3"])))

    val_paths_by_label, val_labels_by_path = get_labels(read_file(src_path + val_label_file_name), val_label_file_name)
    log.info(("number of handwritten files in val: ", len(val_paths_by_label["3"])))

    files_path = get_all_files_from_directory(src_path)
    test_meta_dict, test_empty_file_count = get_all_file_content_meta(files_path, src_path, test_paths_by_label["3"],
                                                                      test_labels_by_path)
    test_label_content_meta = get_all_label_content_meta(test_meta_dict, [test_paths_by_label])
    del test_label_content_meta["3"]
    test_label_content_meta_pd = pd.DataFrame.from_dict(test_label_content_meta, orient='index')
    test_label_content_meta_pd["class_avg_line"] = test_label_content_meta_pd["total_line_count"] / \
                                                   test_label_content_meta_pd["number_of_documents"]
    test_label_content_meta_pd["class_avg_word"] = test_label_content_meta_pd["word_count"] / \
                                                   test_label_content_meta_pd["number_of_documents"]
    test_label_content_meta_pd = test_label_content_meta_pd.round({"class_avg_line": 0, "class_avg_word": 0})
    plot_chart(y="total_line_count", y_label="Total number of lines",
               title="Total number of lines vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
               kind="bar",
               data=test_label_content_meta_pd, pad=30, plot_name="test_total_line_count")
    plot_chart(y="word_count", y_label="Total number of words",
               title="Total number of words vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
               kind="bar",
               data=test_label_content_meta_pd, pad=30, plot_name="test_word_count")
    plot_chart(y="class_avg_line", y_label="Average number of lines",
               title="Average number of lines vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
               kind="bar",
               data=test_label_content_meta_pd, pad=30, plot_name="test_avg_line_count")
    plot_chart(y="class_avg_word", y_label="Average number of words",
               title="Average number of words vs Classes for\n" + str(len(test_labels_by_path)) + " test documents",
               kind="bar",
               data=test_label_content_meta_pd, pad=30, plot_name="test_avg_word_count")

    train_meta_dict, train_empty_file_count = get_all_file_content_meta(files_path, src_path, train_paths_by_label["3"],
                                                                        train_labels_by_path)
    train_label_content_meta = get_all_label_content_meta(train_meta_dict, [train_paths_by_label])
    del train_label_content_meta["3"]
    train_label_content_meta_pd = pd.DataFrame.from_dict(train_label_content_meta, orient='index')
    train_label_content_meta_pd["class_avg_line"] = train_label_content_meta_pd["total_line_count"] / \
                                                    train_label_content_meta_pd["number_of_documents"]
    train_label_content_meta_pd["class_avg_word"] = train_label_content_meta_pd["word_count"] / \
                                                    train_label_content_meta_pd["number_of_documents"]
    train_label_content_meta_pd = train_label_content_meta_pd.round({"class_avg_line": 0, "class_avg_word": 0})
    plot_chart(y="total_line_count", y_label="Total number of lines",
               title="Total number of lines vs Classes for\n" + str(len(train_labels_by_path)) + " train documents",
               kind="bar",
               data=train_label_content_meta_pd, pad=30, plot_name="train_total_line_count")
    plot_chart(y="word_count", y_label="Total number of words",
               title="Total number of words vs Classes for\n" + str(len(train_labels_by_path)) + " train documents",
               kind="bar",
               data=train_label_content_meta_pd, pad=30, plot_name="train_word_count")
    plot_chart(y="class_avg_line", y_label="Average number of lines",
               title="Average number of lines vs Classes for\n" + str(len(train_labels_by_path)) + " train documents",
               kind="bar",
               data=train_label_content_meta_pd, pad=30, plot_name="train_avg_line_count")
    plot_chart(y="class_avg_word", y_label="Average number of words",
               title="Average number of words vs Classes for\n" + str(len(train_labels_by_path)) + " train documents",
               kind="bar",
               data=train_label_content_meta_pd, pad=30, plot_name="train_avg_word_count")

    val_meta_dict, val_empty_file_count = get_all_file_content_meta(files_path, src_path, val_paths_by_label["3"],
                                                                    val_labels_by_path)
    val_label_content_meta = get_all_label_content_meta(val_meta_dict, [val_paths_by_label])
    del val_label_content_meta["3"]
    val_label_content_meta_pd = pd.DataFrame.from_dict(val_label_content_meta, orient='index')
    val_label_content_meta_pd["class_avg_line"] = val_label_content_meta_pd["total_line_count"] / \
                                                  val_label_content_meta_pd["number_of_documents"]
    val_label_content_meta_pd["class_avg_word"] = val_label_content_meta_pd["word_count"] / \
                                                  val_label_content_meta_pd["number_of_documents"]
    val_label_content_meta_pd = val_label_content_meta_pd.round({"class_avg_line": 0, "class_avg_word": 0})
    plot_chart(y="total_line_count", y_label="Total number of lines",
               title="Total number of lines vs Classes for " + str(len(val_labels_by_path)) + " val documents",
               kind="bar",
               data=val_label_content_meta_pd, pad=30, plot_name="val_total_line_count")
    plot_chart(y="word_count", y_label="Total number of words",
               title="Total number of words vs Classes for " + str(len(val_labels_by_path)) + " val documents",
               kind="bar",
               data=val_label_content_meta_pd, pad=30, plot_name="val_word_count")
    plot_chart(y="class_avg_line", y_label="Average number of lines",
               title="Average number of lines vs Classes for " + str(len(val_labels_by_path)) + " val documents",
               kind="bar",
               data=val_label_content_meta_pd, pad=30, plot_name="val_avg_line_count")
    plot_chart(y="class_avg_word", y_label="Average number of words",
               title="Average number of words vs Classes for " + str(len(val_labels_by_path)) + " val documents",
               kind="bar",
               data=val_label_content_meta_pd, pad=30, plot_name="val_avg_word_count")
    # plot_chart(all_label_content_meta)

    log.error(("Number of error file:", error_file_count))


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

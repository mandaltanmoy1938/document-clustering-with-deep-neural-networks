import logging as log


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

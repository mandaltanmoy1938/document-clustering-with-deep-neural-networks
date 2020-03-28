import logging as log
import seaborn as sns
import matplotlib.pyplot as plt


def plot_chart(y, y_label, title, kind, data, pad, plot_name, fig_num):
    sns_plot = sns.catplot(x="label", y=y, kind=kind, data=data)
    sns_plot.set_xticklabels(rotation=45, ha='right')
    sns_plot.set_axis_labels("Classes", y_label)
    for index, row in data.iterrows():
        if int(index) > 3:
            index = int(index) - 1
        sns_plot.ax.text(float(index) - 0.25, row[y], row[y], rotation=45)
    plt.title(title, pad=pad)
    sns_plot.savefig(plot_name + ".png")
    plt.figure(fig_num)
    plt.show()


def plot_cluster(title, data, pad, plot_name, fig_num):
    classes = data['label'].unique()
    num_classes = classes.shape[0]
    log.debug(title + "num_classes:" + str(num_classes))
    log.debug(title + "classes:")
    log.debug(classes)
    sns_plot = sns.scatterplot(
        x="x", y="y",
        hue="label",
        palette=sns.color_palette("bright", num_classes),
        data=data,
        legend="full"
    )

    plt.title(title, pad=pad)
    plt.figure(num=fig_num)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=5, fancybox=True, shadow=True)
    sns_plot.figure.savefig(plot_name + ".png")
    # plt.figure(fig_num)
    # plt.show()

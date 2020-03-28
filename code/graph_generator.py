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
    #plt.title(title, pad=pad)
    plt.figure(num=fig_num, figsize=(16, 10))
    num_classes = data['label'].unique().shape[0]
    print(num_classes)
    sns_plot = sns.scatterplot(
        x="x", y="y",
        hue="label",
        palette=sns.color_palette("bright", num_classes),
        data=data,
        legend="full"
    )

    sns_plot.savefig(plot_name + ".png")

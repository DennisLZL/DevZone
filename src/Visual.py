__author__ = 'dennis'

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from ggplot import *

def load_data(file_path, header, sep):
    raw_data = []
    labels = []
    ips = []

    with open(file_path, 'r') as f:
        for line in f:
            ip, ins = line.split(':')
            raw_data.append(dict(zip(header, ins.strip().split(', '))))
            ips.append(ip)

    data = pd.DataFrame(raw_data, index=ips).convert_objects(convert_numeric=True)

    for ip in ips:
        d = int(ip.split('.')[-1])
        labels.append([s for s, x in enumerate(sep) if x >=d][0])

    return data, labels

def pca_coord(data, num_header):
    pca = PCA(n_components=2)
    pca.fit(data[num_header])
    return pca.transform(data[num_header]), pca

if __name__ == '__main__':

    with open('../data/header.txt', 'r') as f:
        head_name = f.readline().split(',')

    # # plot ground truth of 30 devices
    # dev_file30 = '../data/dev30.txt'
    # cls_sep30 = [6, 10, 13, 18, 22, 26, 30]
    #
    # dev_data, dev_label = load_data(dev_file30, head_name, cls_sep30)
    # coord, pc = pca_coord(dev_data, head_name[0:31])
    # result = pd.DataFrame({'x': coord[:, 0], 'y': coord[:, 1], 'cluster': dev_label})
    #
    # dev30_mean = pd.read_csv('../data/dev30clsmean.txt', names=head_name[0:31])
    # coord_mean = pc.transform(dev30_mean)
    #
    # g30 = ggplot(aes(x='x', y='y', color='cluster'), data=result) + geom_point()
    # g30.draw()
    # ax = plt.gca()
    # plt.ylim((-15, 28))
    # ax.text(62, 5, 'ARI = 1.0', fontsize=15)
    #
    # for i in range(len(coord_mean)):
    #     ell = Ellipse(xy=(coord_mean[i][0], coord_mean[i][1]), width=12, height=12, color='grey', alpha=0.3)
    #     ax.add_patch(ell)

    # # plot ground truth of 100 devices
    # dev_file100 = '../data/dev100v2.txt'
    # cls_sep100 = [20, 50, 75, 100]
    # dev_data, dev_label = load_data(dev_file100, head_name, cls_sep100)
    # coord, pc = pca_coord(dev_data, head_name[0:31])
    # result = pd.DataFrame({'x': coord[:, 0], 'y': coord[:, 1], 'cluster': dev_label})
    #
    # g100 = ggplot(aes(x='x', y='y', color='cluster'), data=result) + geom_point()
    # g100.draw()

    # plot ground truth of v2 100 devices
    dev_file100v2 = '../data/dev100v2.txt'
    cls_sep100 = [20, 50, 75, 100]
    dev_data, dev_label = load_data(dev_file100v2, head_name, cls_sep100)
    coord, pc = pca_coord(dev_data, head_name[0:31])
    result = pd.DataFrame({'x': coord[:, 0], 'y': coord[:, 1], 'cluster': dev_label})

    dev100v2_mean = pd.read_csv('../data/dev100v2clsmean.txt', names=head_name[0:31])
    coord_mean = pc.transform(dev100v2_mean)

    g100v2 = ggplot(aes(x='x', y='y', color='cluster'), data=result) + geom_point()
    g100v2.draw()
    ax = plt.gca()

    for i in range(len(coord_mean)):
        ell = Ellipse(xy=(coord_mean[i][0], coord_mean[i][1]), width=12, height=12, color='grey', alpha=0.3)
        ax.add_patch(ell)
    plt.xlim((-30, 30))
    plt.ylim((-8, 8))
    ax.text(32, 0, 'ARI = 0.67', fontsize=15)

    plt.show()


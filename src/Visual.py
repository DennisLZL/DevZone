__author__ = 'dennis'

import pandas as pd
from sklearn.decomposition import PCA
from ggplot import *
import matplotlib.pyplot as plt
import numpy as np

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
    return pca.transform(data[num_header])

def ellipse_coord(x0, y0, a, b, theta):
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # coordinates
    x = np.linspace(-a, a, 100)
    y = b * np.sqrt(1 - x ** 2 / (a ** 2))
    x += x0
    y += y0
    c1 = R.dot(np.array([x, y]))
    c2 = R.dot(np.array([x, -y]))
    c = np.hstack([c1, c2])




    return pd.DataFrame({'x': x, 'ymin': ymin, 'ymax': ymax})

if __name__ == '__main__':
    res = ellipse_coord(2, 3)
    print ggplot(aes(x='x', ymin='ymin', ymax='ymax'), data=res) + geom_area(aes(fill='red', alpha=0.2))
'''
    head_name = ['f01_total_unique_receivers', 'f02_total_unique_senders', 'f03_inter_receivers_senders',
              'f04_union_receivers_senders', 'f05_ratio_receivers_to_senders', 'f06_avg_freq_sending_pkg',
              'f07_avg_freq_receiving_pkg', 'f08_std_freq_sending_pkg', 'f09_std_freq_receiving_pkg',
              'f10_min_freq_sending_pkg', 'f11_q25_freq_sending_pkg', 'f12_q50_freq_sending_pkg',
              'f13_q75_freq_sending_pkg', 'f14_max_freq_sending_pkg', 'f15_min_freq_receiving_pkg',
              'f16_q25_freq_receiving_pkg', 'f17_q50_freq_receiving_pkg', 'f18_q75_freq_receiving_pkg',
              'f19_max_freq_receiving_pkg', 'f20_ratio_freq_send_receive_pkg', 'f21_corr_freq_send_receive_pkg',
              'f22_total_unique_prot_send', 'f23_total_unique_prot_receiced', 'f24_ratio_unique_prot_send_received',
              'f25_inter_prot_send_received', 'f26_union_prot_send_received', 'f27_total_unique_info_send',
              'f28_total_unique_info_receiced', 'f29_ratio_unique_info_send_received', 'f30_inter_info_send_received',
              'f31_union_info_send_received', 'f00_mac_manufacturer', 'f32_most_freq_send_prot',
              'f33_most_freq_received_prot']

    # plot ground truth of 30 devices
    dev_file30 = '../data/dev30.txt'
    cls_sep30 = [6, 10, 13, 18, 22, 26, 30]

    dev_data, dev_label = load_data(dev_file30, head_name, cls_sep30)
    coord = pca_coord(dev_data, head_name[0:31])
    result = pd.DataFrame({'x': coord[:, 0], 'y': coord[:, 1], 'cluster': dev_label})

    g30 = ggplot(aes(x='x', y='y', color='cluster'), data=result) + geom_point()
    g30.draw()

    # plot ground truth of 100 devices
    dev_file100 = '../data/dev100.txt'
    cls_sep100 = [20, 50, 75, 100]
    dev_data, dev_label = load_data(dev_file100, head_name, cls_sep100)
    coord = pca_coord(dev_data, head_name[0:31])
    result = pd.DataFrame({'x': coord[:, 0], 'y': coord[:, 1], 'cluster': dev_label})

    g100 = ggplot(aes(x='x', y='y', color='cluster'), data=result) + geom_point()
    g100.draw()

    # plot ground truth of v2 100 devices
    dev_file100v2 = '../data/dev100v2.txt'
    cls_sep100 = [20, 50, 75, 100]
    dev_data, dev_label = load_data(dev_file100v2, head_name, cls_sep100)
    coord = pca_coord(dev_data, head_name[0:31])
    result = pd.DataFrame({'x': coord[:, 0], 'y': coord[:, 1], 'cluster': dev_label})

    g100v2 = ggplot(aes(x='x', y='y', color='cluster'), data=result) + geom_point()
    g100v2.draw()

    plt.show()
'''

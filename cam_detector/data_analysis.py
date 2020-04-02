from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from data_maker import make_data_list


plt.rcParams.update({'figure.max_open_warning': 50})


def get_max_from_lists(*lists):
    flatten = np.concatenate(lists).ravel()
    max_value = np.max(flatten)
    return max_value


def histogram(i, data_list, title, *, x_lim=[], y_lim=[]):
    f = plt.figure(i)
    plt.hist(data_list, bins=30,
                        weights=np.ones(len(data_list))/len(data_list))

    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    if x_lim:
        plt.gca().set_xlim(x_lim)

    if y_lim:
        plt.gca().set_ylim(y_lim)

    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.title(title)

    return f


def get_dict_from_data_list(data_list, name):

    data_dict = defaultdict(lambda: [])
    data_dict['name'] = name
    for vector in data_list:
        for field in vector._fields:
            data_dict[field].append(getattr(vector, field))

    return data_dict


def make_histogram(*data_dicts):
    i = 1
    for key in data_dicts[0].keys():
        if key == 'name':
            continue

        feature_lists = []
        for data_dict in data_dicts:
            feature_lists.append((data_dict[key], data_dict['name']))

        m = get_max_from_lists(*[f[0] for f in feature_lists])
        for f in feature_lists:
            histogram(i, f[0], f'{f[1]}_{key}', x_lim=[0, m])
            i += 1

    plt.show()

if __name__ == '__main__':
    with open('data_path/pos_nm_up_paths.txt', 'r') as f:
        pos_path_list = f.read().splitlines()
    pos_data_list = make_data_list(pos_path_list)
    pos_data_dict = get_dict_from_data_list(pos_data_list, 'pos')

    with open('data_path/neg_paths.txt', 'r') as f:
        neg_path_list = f.read().splitlines()
    neg_data_list = make_data_list(neg_path_list)
    neg_data_dict = get_dict_from_data_list(neg_data_list, 'neg')

    with open('data_path/neg_dummy_paths.txt', 'r') as f:
        neg_dummy_path_list = f.read().splitlines()
    neg_dummy_data_list = make_data_list(neg_dummy_path_list)
    neg_dummy_data_dict = get_dict_from_data_list(neg_dummy_data_list, 'neg_dummy')

    print(len(pos_data_list))
    print(len(neg_data_list))
    print(len(neg_dummy_data_list))

    make_histogram(pos_data_dict, neg_data_dict, neg_dummy_data_dict)

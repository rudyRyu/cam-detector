import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from data_maker import make_data_list
from logger import log

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

    argparser = argparse.ArgumentParser(description='config')
    argparser.add_argument(
        '-c',
        '--conf',
        required=True,
        help='path to a configuration file')


    args = argparser.parse_args()
    config_path = args.conf

    with open(config_path) as config_buffer:
        conf = json.loads(config_buffer.read())

    if not conf["data_analysis"]["dataset"]:
        log.info("check dataset")

    else:
        log.info("data loading ...")
        data_dict_list = []
        for name, path in conf["data_analysis"]["dataset"]:
            with open(path, 'r') as f:
                path_list = f.read().splitlines()

            data_list = make_data_list(path_list)
            data_dict = get_dict_from_data_list(data_list, name)
            data_dict_list.append(data_dict)

            log.info(f"{name}: {len(data_list)}")

        make_histogram(*data_dict_list)

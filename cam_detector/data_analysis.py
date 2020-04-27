import argparse
import json
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from data_maker import make_data_list
from utils import logger, refine_features, parse_config

log = logger(name=__name__,
             level=logging.INFO)


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

    conf = parse_config()

    if not conf["data_analysis"]["dataset"]:
        log.info("check dataset")

    else:
        log.info("data loading ...")

        min_packet_size = conf["data_analysis"]["data_options"]["min_packet_size"]
        split_num = conf["data_analysis"]["data_options"]["split_num_on_pld"]
        step_size = conf["data_analysis"]["data_options"]["step_size"]
        features_to_use = refine_features(conf["data_analysis"]["features"])

        data_dict_list = []
        for name, path in conf["data_analysis"]["dataset"]:
            with open(path, 'r') as f:
                path_list = f.read().splitlines()

            data_list = make_data_list(path_list,
                                       min_packet_size=min_packet_size,
                                       split_num_on_pld=split_num,
                                       step_size=step_size,
                                       features_to_use=features_to_use)


            data_dict = get_dict_from_data_list(data_list, name)
            data_dict_list.append(data_dict)

            log.info(f"{name}: {len(data_list)}")

        make_histogram(*data_dict_list)

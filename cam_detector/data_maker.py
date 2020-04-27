
"""
DeWiCam explores four ﬁelds (length, duration, FC and Address)

Problems
- How should I get FC from Ethernet packets?
- Duration values are too big. Did I parse it correctly?

Process
1. Sort SRC, DST addresses
2. Get 4 vectors
"""

import pickle
from collections import namedtuple
from pprint import pprint
from statistics import stdev

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, ks_2samp

from pkl_maker import read_pkl


def get_cdf1(data):
    hist, bin_edges = np.histogram(data, bins=20, density=True)
    cdf = np.cumsum(hist*np.diff(bin_edges))
    return cdf


def get_cdf2(data):
    data_size=len(data)

    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    counts = counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)
    return cdf


def get_pld(data, split_num=50):
    block_num = (len(data)+(split_num//2)) // split_num
    if block_num <= 1:
        raise ValueError('The number of blocks must be greater than 1')

    for i in range(block_num):
        data_clip = data[i*split_num:(i+1)*split_num]

        print(get_cdf1(data_clip))
        print(get_cdf2(data_clip))
        print(get_cdf3(data_clip))
        input()


def get_pld_stb(data, split_num=50):
    block_num = (len(data)+(split_num//2)) // split_num
    if block_num <= 1:
        raise ValueError('The number of blocks must be greater than 1')

    cum_s = 0
    cum_p = 0
    l1 = data[:split_num]
    for j in range(1, block_num):
        lj = data[j*split_num:(j+1)*split_num]

        cum_s += ks_2samp(l1, lj).statistic
        cum_p += ks_2samp(l1, lj).pvalue

    statistic_stb = cum_s/(block_num-1)
    pvalue_stb = cum_p/(block_num-1)
    return statistic_stb, pvalue_stb


def get_pld_stb_with_cdf(data, split_num=50):
    block_num = (len(data)+(split_num//2)) // split_num
    if block_num <= 1:
        raise ValueError('The number of blocks must be greater than 1')

    cum_s = 0
    cum_p = 0
    l1 = get_cdf1(data[:split_num])
    for j in range(1, block_num):
        lj = get_cdf1(data[j*split_num:(j+1)*split_num])

        cum_s += ks_2samp(l1, lj).statistic
        cum_p += ks_2samp(l1, lj).pvalue

    statistic_stb = cum_s/(block_num-1)
    pvalue_stb = cum_p/(block_num-1)
    return statistic_stb, pvalue_stb


def get_bandwidth_std(data):
    bandwidth_list = []
    for time_delta, length in data[1:]:
        kbps = (1./time_delta)*(length/1024*8)
        bandwidth_list.append(kbps)

    return np.std(bandwidth_list)


def get_bandwidth_avg(data):
    bandwidth_list = []
    for time_delta, length in data[1:]:
        kbps = (1./time_delta)*(length/1024*8)
        bandwidth_list.append(kbps)

    return np.mean(bandwidth_list)


def get_flow_bandwidth(data):
    bandwidth_list = []
    time_difference = data[-1][0] - data[0][0]
    length_sum = sum([d[1] for d in data])
    bandwidth_kbps = (1./time_difference)*(length_sum/1024*8)

    return bandwidth_kbps


def get_duration_std(data):
    return np.std(data)


def get_duration_avg(data):
    return np.mean(data)


def get_length_avg(data):
    return np.mean(data)


def get_vector(data, *, features_to_use=[],
                        split_num_on_pld=50) -> list:

    vector_list = []
    vector_name_list = []
    if any(['length_avg' in features_to_use,
            'pld_stat_stb' in features_to_use,
            'pld_pval_stb' in features_to_use,
            'pld_stat_stb_with_cdf' in features_to_use,
            'pld_pval_stb_with_cdf' in features_to_use]):

        length_list = [d['length'] for d in data]

        if 'length_avg' in features_to_use:
            length_avg = get_length_avg(length_list)
            vector_list.append(length_avg)
            vector_name_list.append('length_avg')

        if any(['pld_stat_stb' in features_to_use,
                'pld_pval_stb' in features_to_use]):

            pld_stat_stb, pld_pval_stb = get_pld_stb(
                                                length_list,
                                                split_num=split_num_on_pld)

            if 'pld_stat_stb' in features_to_use:
                vector_list.append(pld_stat_stb)
                vector_name_list.append('pld_stat_stb')

            if 'pld_pval_stb' in features_to_use:
                vector_list.append(pld_pval_stb)
                vector_name_list.append('pld_pval_stb')

        if any(['pld_stat_stb_with_cdf' in features_to_use,
                'pld_pval_stb_with_cdf' in features_to_use]):

            pld_stat_stb_with_cdf, pld_pval_stb_with_cdf = \
                            get_pld_stb_with_cdf(length_list,
                                                 split_num=split_num_on_pld)

            if 'pld_stat_stb_with_cdf' in features_to_use:
                vector_list.append(pld_stat_stb_with_cdf)
                vector_name_list.append('pld_stat_stb_with_cdf')

            if 'pld_pval_stb_with_cdf' in features_to_use:
                vector_list.append(pld_pval_stb_with_cdf)
                vector_name_list.append('pld_pval_stb_with_cdf')

    if any(['duration_std' in features_to_use,
            'duration_avg' in features_to_use]):

        duration_list = [d['duration'] for d in data]

        if 'duration_std' in features_to_use:
            duration_std = get_duration_std(duration_list)
            vector_list.append(duration_std)
            vector_name_list.append('duration_std')

        if 'duration_avg' in features_to_use:
            duration_avg = get_duration_avg(duration_list)
            vector_list.append(duration_avg)
            vector_name_list.append('duration_avg')

    if any(['bandwidth_std' in features_to_use,
            'bandwidth_avg' in features_to_use]):

        bandwidth_list = [(d['time_delta'], d['length']) for d in data]

        if 'bandwidth_std' in features_to_use:
            bandwidth_std = get_bandwidth_std(bandwidth_list)
            vector_list.append(bandwidth_std)
            vector_name_list.append('bandwidth_std')

        if 'bandwidth_avg' in features_to_use:
            bandwidth_avg = get_bandwidth_avg(bandwidth_list)
            vector_list.append(bandwidth_avg)
            vector_name_list.append('bandwidth_avg')

    if 'flow_bandwidth' in features_to_use:
        flow_bandwidth_list= [(d['time_relative'], d['length']) for d in data]
        flow_bandwidth = get_flow_bandwidth(flow_bandwidth_list)
        vector_list.append(flow_bandwidth)
        vector_name_list.append('flow_bandwidth')

    Vector = namedtuple('Vector', [*vector_name_list])

    vector = Vector(*vector_list)
    return vector


def make_data_list(path_list, *, min_packet_size=300,
                                 split_num_on_pld=50,
                                 step_size=300,
                                 features_to_use=None) -> list:
    data_list = []
    for path in path_list:
        with open(path, 'rb') as fp:
            flows = pickle.load(fp)

        vectors = make_data_list_from_flow(flows,
                                           min_packet_size=min_packet_size,
                                           split_num_on_pld=split_num_on_pld,
                                           step_size=step_size,
                                           features_to_use=features_to_use)

        data_list.extend(vectors)

    return data_list

def make_data_list_from_flow(flow, *, min_packet_size=300,
                                      split_num_on_pld=50,
                                      step_size=300,
                                      features_to_use=None) -> list:
    data_list = []
    for src, src_value in flow.items():
        for dst, dst_value in src_value.items():
            for fc, data in dst_value.items():
                if len(data) < min_packet_size:
                    continue

                stop_len = len(data)-min_packet_size
                for i in range(0, stop_len, step_size):
                    vector = get_vector(data[i:i+min_packet_size],
                                        split_num_on_pld=split_num_on_pld,
                                        features_to_use=features_to_use)
                    data_list.append(vector)

    return data_list


if __name__ == '__main__':
    with open('pos_paths.txt', 'r') as f:
        pos_path_list = f.read().splitlines()
    pos_data_list = make_data_list(pos_path_list)

    # with open('neg_paths.txt', 'r') as f:
    #     neg_path_list = f.read().splitlines()
    # neg_data_list = make_data_list(neg_path_list)

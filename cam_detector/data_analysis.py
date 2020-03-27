
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from data_maker import make_data_list


plt.rcParams.update({'figure.max_open_warning': 30})


def get_max_from_lists(*lists):
    flatten = np.concatenate(lists).ravel()
    max_value = np.max(flatten)
    return max_value


def histogram(i, data_list, title, *, x_lim=[], y_lim=[]):
    f = plt.figure(i)
    plt.hist(data_list, bins=30, weights=np.ones(len(data_list)) / len(data_list))  # `density=False` would make counts
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    if x_lim:
        plt.gca().set_xlim(x_lim)

    if y_lim:
        plt.gca().set_ylim(y_lim)

    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.title(title)

    return f


if __name__ == '__main__':
    with open('pos_paths.txt', 'r') as f:
        pos_path_list = f.read().splitlines()
    pos_data_list = make_data_list(pos_path_list)

    pos_duration_avg_list = []
    pos_duration_std_list = []
    pos_pld_stat_stb_list = []
    pos_pld_stat_stb_with_cdf_list = []
    pos_pld_pval_stb_list = []
    pos_pld_pval_stb_with_cdf_list = []
    pos_bandwidth_std_list = []
    pos_bandwidth_avg_list = []
    pos_length_avg_list = []

    for vector in pos_data_list:
        pos_duration_avg_list.append(vector.duration_avg)
        pos_duration_std_list.append(vector.duration_std)
        pos_pld_stat_stb_list.append(vector.pld_stat_stb)
        pos_pld_stat_stb_with_cdf_list.append(vector.pld_stat_stb_with_cdf)
        pos_pld_pval_stb_list.append(vector.pld_pval_stb)
        pos_pld_pval_stb_with_cdf_list.append(vector.pld_pval_stb_with_cdf)
        pos_bandwidth_std_list.append(vector.bandwidth_std)
        pos_bandwidth_avg_list.append(vector.bandwidth_avg)
        pos_length_avg_list.append(vector.length_avg)


    with open('neg_paths.txt', 'r') as f:
        neg_path_list = f.read().splitlines()
    neg_data_list = make_data_list(neg_path_list)

    neg_duration_avg_list = []
    neg_duration_std_list = []
    neg_pld_stat_stb_list = []
    neg_pld_stat_stb_with_cdf_list = []
    neg_pld_pval_stb_list = []
    neg_pld_pval_stb_with_cdf_list = []
    neg_bandwidth_std_list = []
    neg_bandwidth_avg_list = []
    neg_length_avg_list = []

    for vector in neg_data_list:
        neg_duration_avg_list.append(vector.duration_avg)
        neg_duration_std_list.append(vector.duration_std)
        neg_pld_stat_stb_list.append(vector.pld_stat_stb)
        neg_pld_stat_stb_with_cdf_list.append(vector.pld_stat_stb_with_cdf)
        neg_pld_pval_stb_list.append(vector.pld_pval_stb)
        neg_pld_pval_stb_with_cdf_list.append(vector.pld_pval_stb_with_cdf)
        neg_bandwidth_std_list.append(vector.bandwidth_std)
        neg_bandwidth_avg_list.append(vector.bandwidth_avg)
        neg_length_avg_list.append(vector.length_avg)


    with open('neg_dummy_paths.txt', 'r') as f:
        neg_dummy_path_list = f.read().splitlines()
    neg_dummy_data_list = make_data_list(neg_dummy_path_list)

    neg_dummy_duration_avg_list = []
    neg_dummy_duration_std_list = []
    neg_dummy_pld_stat_stb_list = []
    neg_dummy_pld_stat_stb_with_cdf_list = []
    neg_dummy_pld_pval_stb_list = []
    neg_dummy_pld_pval_stb_with_cdf_list = []
    neg_dummy_bandwidth_std_list = []
    neg_dummy_bandwidth_avg_list = []
    neg_dummy_length_avg_list = []

    for vector in neg_dummy_data_list:
        neg_dummy_duration_avg_list.append(vector.duration_avg)
        neg_dummy_duration_std_list.append(vector.duration_std)
        neg_dummy_pld_stat_stb_list.append(vector.pld_stat_stb)
        neg_dummy_pld_stat_stb_with_cdf_list.append(vector.pld_stat_stb_with_cdf)
        neg_dummy_pld_pval_stb_list.append(vector.pld_pval_stb)
        neg_dummy_pld_pval_stb_with_cdf_list.append(vector.pld_pval_stb_with_cdf)
        neg_dummy_bandwidth_std_list.append(vector.bandwidth_std)
        neg_dummy_bandwidth_avg_list.append(vector.bandwidth_avg)
        neg_dummy_length_avg_list.append(vector.length_avg)


    print(len(pos_data_list))
    print(len(neg_data_list))
    print(len(neg_dummy_data_list))

    m1 = get_max_from_lists(pos_duration_avg_list, neg_duration_avg_list, neg_dummy_duration_avg_list)
    p1 = histogram(1, pos_duration_avg_list, 'pos_duration_avg_list', x_lim=[0, m1])
    n1 = histogram(2, neg_duration_avg_list, 'neg_duration_avg_list', x_lim=[0, m1])
    d1 = histogram(3, neg_dummy_duration_avg_list, 'neg_dummy_duration_avg_list', x_lim=[0, m1])

    m2 = get_max_from_lists(pos_duration_std_list, neg_duration_std_list, neg_dummy_duration_std_list)
    p2 = histogram(4, pos_duration_std_list, 'pos_duration_std_list', x_lim=[0, m2])
    n2 = histogram(5, neg_duration_std_list, 'neg_duration_std_list', x_lim=[0, m2])
    d2 = histogram(6, neg_dummy_duration_std_list, 'neg_dummy_duration_std_list', x_lim=[0, m2])

    m3 = get_max_from_lists(pos_pld_stat_stb_list, neg_pld_stat_stb_list, neg_dummy_pld_stat_stb_list)
    p3 = histogram(7, pos_pld_stat_stb_list, 'pos_pld_stat_stb_list', x_lim=[0, m3])
    n3 = histogram(8, neg_pld_stat_stb_list, 'neg_pld_stat_stb_list', x_lim=[0, m3])
    d3 = histogram(9, neg_dummy_pld_stat_stb_list, 'neg_dummy_pld_stat_stb_list', x_lim=[0, m3])

    m4 = get_max_from_lists(pos_pld_stat_stb_with_cdf_list, neg_pld_stat_stb_with_cdf_list, neg_dummy_pld_stat_stb_with_cdf_list)
    p4 = histogram(10, pos_pld_stat_stb_with_cdf_list, 'pos_pld_stat_stb_with_cdf_list', x_lim=[0, m4])
    n4 = histogram(11, neg_pld_stat_stb_with_cdf_list, 'neg_pld_stat_stb_with_cdf_list', x_lim=[0, m4])
    d4 = histogram(12, neg_dummy_pld_stat_stb_with_cdf_list, 'neg_dummy_pld_stat_stb_with_cdf_list', x_lim=[0, m4])

    m5 = get_max_from_lists(pos_pld_pval_stb_list, neg_pld_pval_stb_list, neg_dummy_pld_pval_stb_list)
    p5 = histogram(13, pos_pld_pval_stb_list, 'pos_pld_pval_stb_list', x_lim=[0, m5])
    n5 = histogram(14, neg_pld_pval_stb_list, 'neg_pld_pval_stb_list', x_lim=[0, m5])
    d5 = histogram(15, neg_dummy_pld_pval_stb_list, 'neg_dummy_pld_pval_stb_list', x_lim=[0, m5])

    m6 = get_max_from_lists(pos_pld_pval_stb_with_cdf_list, neg_pld_pval_stb_with_cdf_list, neg_dummy_pld_pval_stb_with_cdf_list)
    p6 = histogram(16, pos_pld_pval_stb_with_cdf_list, 'pos_pld_pval_stb_with_cdf_list', x_lim=[0, m6])
    n6 = histogram(17, neg_pld_pval_stb_with_cdf_list, 'neg_pld_pval_stb_with_cdf_list', x_lim=[0, m6])
    d6 = histogram(18, neg_dummy_pld_pval_stb_with_cdf_list, 'neg_dummy_pld_pval_stb_with_cdf_list', x_lim=[0, m6])

    m7 = get_max_from_lists(pos_bandwidth_std_list, neg_bandwidth_std_list, neg_dummy_bandwidth_std_list)
    p7 = histogram(19, pos_bandwidth_std_list, 'pos_bandwidth_std_list', x_lim=[0, m7])
    n7 = histogram(20, neg_bandwidth_std_list, 'neg_bandwidth_std_list', x_lim=[0, m7])
    d7 = histogram(21, neg_dummy_bandwidth_std_list, 'neg_dummy_bandwidth_std_list', x_lim=[0, m7])

    m8 = get_max_from_lists(pos_bandwidth_avg_list, neg_bandwidth_avg_list, neg_dummy_bandwidth_avg_list)
    p8 = histogram(22, pos_bandwidth_avg_list, 'pos_bandwidth_avg_list', x_lim=[0, m8])
    n8 = histogram(23, neg_bandwidth_avg_list, 'neg_bandwidth_avg_list', x_lim=[0, m8])
    d8 = histogram(24, neg_dummy_bandwidth_avg_list, 'neg_dummy_bandwidth_avg_list', x_lim=[0, m8])

    m9 = get_max_from_lists(pos_length_avg_list, neg_length_avg_list, neg_dummy_length_avg_list)
    p9 = histogram(25, pos_length_avg_list, 'pos_length_avg_list', x_lim=[0, m9])
    n9 = histogram(26, neg_length_avg_list, 'neg_length_avg_list', x_lim=[0, m9])
    d9 = histogram(27, neg_dummy_length_avg_list, 'neg_dummy_length_avg_list', x_lim=[0, m9])


    plt.show()

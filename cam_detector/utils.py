import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, ks_2samp

def get_std(data):
    return np.std([packet[field] for packet in data])

def get_avg(data):
    return np.average([packet[field] for packet in data])

def get_cdf1(data):
    hist, bin_edges = np.histogram(data, bins=len(data), density=True)
    cdf = np.cumsum(hist*np.diff(bin_edges))
    return cdf

def get_cdf2(data):
    sorted_data = np.sort(data)
    cdf = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    return cdf

def get_cdf3(data):
    data_size=len(data)
    # Set bins edges
    data_set=sorted(set(data))
    bins=np.append(data_set, data_set[-1]+1)

    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(data, bins=bins, density=False)

    counts=counts.astype(float)/data_size

    # Find the cdf
    cdf = np.cumsum(counts)
    return cdf

def get_pld_stability(length_list, split_num=50):
    block_num = (len(length_list)+(split_num//2)) // split_num
    if block_num <= 1:
        return None

    cum_s = 0
    l1 = length_list[:split_num]
    for j in range(1, block_num):
        lj = length_list[j*split_num:(j+1)*split_num]
        cum_s += ks_2samp(l1, lj).statistic

    stb = cum_s/(block_num-1)
    return stb



if __name__ == '__main__':
    print(ecdf([3, 55, 0.5, 1.5]))
    print(ECDF([3, 55, 0.5, 1.5]))

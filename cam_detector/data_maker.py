
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

from pkl_maker import read_pkl
from scipy.stats import ks_2samp
from utils import get_pld_stability


MIN_DATA_SIZE = 300

def get_vector(data) -> list:

    print('packet num:', len(data))

    length_list = [d['length'] for d in data]
    print(get_pld_stability(length_list, split_num=50))

    print()
    input()



def make_data_list(path_list) -> list:
    data_list = []
    for path in path_list:
        with open(path, 'rb') as fp:
            flows = pickle.load(fp)

        for src, src_value in flows.items():
            for dst, dst_value in src_value.items():
                for fc, data in dst_value.items():
                    if len(data) < MIN_DATA_SIZE:
                        continue

                    for i in range(0, len(data), MIN_DATA_SIZE//2):
                        vector = get_vector(data[i:i+MIN_DATA_SIZE])


if __name__ == '__main__':
    with open('pos_paths.txt', 'r') as f:
        pos_path_list = f.read().splitlines()
    pos_data_list = make_data_list(pos_path_list)

    # with open('neg_paths.txt', 'r') as f:
    #     neg_path_list = f.read().splitlines()
    # neg_data_list = make_data_list(neg_path_list)

import argparse
import os
import pickle
from collections import defaultdict
from pprint import pprint

import pyshark


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def read_cap(path, *, save_pkl=False,
                      annotate_path='') -> dict:
    cap = pyshark.FileCapture(path)
    flows = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: [])))

    for i, packet in enumerate(cap):
        try:
            src = packet.layers[2].sa # source MAC
            dst = packet.layers[2].da # destination MAC
            fc = packet.layers[2].fc_ds # frame control
            length = len(packet)
            duration = float(packet.layers[2].duration)
            data = {
                'length': length,
                'duration': duration
            }
        except:
            continue

        flows[src][dst][fc].append(data)

    cap.close()

    flows = ddict2dict(flows)

    if save_pkl:
        pkl_name = f'{os.path.splitext(path)[0]}.pkl'
        with open(pkl_name, 'wb') as fp:
            pickle.dump(flows, fp)

        if annotate_path:
            with open(annotate_path, 'a') as f:
                f.write(pkl_name + '\n')

    return flows


def read_dir(path, *, save_pkl=True,
                      annotate_path='',
                      return_flows_list=False) -> (dict, int):
    count = 0
    if return_flows_list:
        flows_list = []

    for (path, dirs, files) in os.walk(path):
        for file_name in files:
            file_path = os.path.join(path, file_name)
            if check_path_type(file_path) != 'cap':
                continue

            flows = read_cap(file_path, save_pkl=True,
                                        annotate_path=annotate_path)
            if return_flows_list:
                flows_list.append(flows)

            count += 1

    if return_flows_list:
        return flows_list, count
    else:
        return None, count


def check_path_type(path) -> str:
    path_type = ''
    if os.path.isdir(path):
        path_type = 'dir'

    elif os.path.isfile(path):
        ext = os.path.splitext(path)[-1]
        if ext in ['.pcap', '.cap']:
            path_type = 'cap'

        elif ext in ['.txt']:
            path_type = 'txt'

    else:
        path_type = 'invalid'

    return path_type


def make_pkl(dir_or_file_path, *, save_pkl=True,
                                  annotate_path=''):
    path_type = check_path_type(dir_or_file_path)
    if path_type == 'dir':
        _, count = read_dir(dir_or_file_path, save_pkl=save_pkl,
                                              annotate_path=annotate_path,
                                              return_flows_list=False)
        return count

    elif path_type == 'cap':
        flows = read_cap(dir_or_file_path, save_pkl=save_pkl,
                                           annotate_path=annotate_path)
        return 1

    else:
        print('invalid path')
        return 0

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='pkl maker')
    argparser.add_argument(
        '-p',
        '--path',
        required=True,
        help='path to directory or file')

    args = argparser.parse_args()
    path = args.path

    count = make_pkl(path, save_pkl=True,
                           annotate_path='pos_list.txt')
    if count > 1:
        print(f'{count} pkl files have been made.')
    else:
        print(f'{count} pkl file has been made.')

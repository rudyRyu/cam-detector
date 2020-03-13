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
                      ) -> (dict, int):
    count = 0
    for (path, dirs, files) in os.walk(path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext not in ['.pcap', '.cap']:
                continue

            flows = read_cap(os.path.join(path, filename))

            count += 1

    return count


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


def make_pkl(dir_or_file_path):
    path_type = check_path_type(dir_or_file_path)
    if path_type == 'dir':
        pass

    elif path_type == 'cap':
        ext = os.path.splitext(dir_or_file_path)[-1]
        if ext not in ['.pcap', '.cap']:
            print('This is not an capture file.')
            return 0
        else:
            flows = read_cap(dir_or_file_path, save_pkl=True,
                                               annotate_path='test.txt')
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

    count = make_pkl(path)

    # if count > 1:
    #     print(f'{count} pkl files have been made.')
    # else:
    #     print(f'{count} pkl file has been made.')

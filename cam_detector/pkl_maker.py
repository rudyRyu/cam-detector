import argparse
import logging
import os
import pickle
from collections import defaultdict
from operator import itemgetter
from pprint import pprint

import pyshark

from utils import logger, parse_config

log = logger(name=__name__,
             level=logging.INFO)


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def read_pkl(path) -> dict:
    with open(path, 'rb') as fp:
        flows = pickle.load(fp)

    return flows


def remove_dummy_data(flows):

    key_len_list = []
    for src, src_value in flows.items():
        for dst, dst_value in src_value.items():
            for fc, fc_value in dst_value.items():
                key_len_list.append((src, dst, fc, len(fc_value)))

    if len(key_len_list) > 1:
        key_len_list = sorted(key_len_list, key=itemgetter(3), reverse=True)
        for t in key_len_list[1:]:
            del flows[t[0]][t[1]][t[2]]


def read_cap(path, *, save_pkl=False,
                      annotate_path='',
                      remove_dummies=True,
                      verbose=True) -> dict:

    try:
        cap = pyshark.FileCapture(path)
    except pyshark.capture.capture.TSharkCrashException as e:
        pass

    flows = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: [])))

    i = 0
    while True:
        try:
            packet = cap.next()

        except StopIteration:
            break

        except Exception as e: #pyshark.capture.capture.TSharkCrashException as e:
            log.debug('exception')
            break

        else:
            try:
                src = packet.layers[2].sa # source MAC
                dst = packet.layers[2].da # destination MAC
                fc = packet.layers[2].fc_ds # frame control
                length = len(packet)
                duration = float(packet.layers[2].duration)
                time_delta = float(packet.frame_info.time_delta)
                time_relative = float(packet.frame_info.time_relative)
                data = {
                    'length': length,
                    'duration': duration,
                    'time_delta': time_delta,
                    'time_relative': time_relative
                }

            except:
                continue

            i += 1
            flows[src][dst][fc].append(data)

    try:
        cap.close()
    except:
        log.debug('close except')

    if remove_dummies:
        remove_dummy_data(flows)

    flows = ddict2dict(flows)

    if save_pkl:
        pkl_path = f'{os.path.splitext(path)[0]}.pkl'
        with open(pkl_path, 'wb') as fp:
            pickle.dump(flows, fp)

        if verbose:
            log.info(f'{os.path.split(pkl_path)[-1]} has been created.')

        if annotate_path:
            with open(annotate_path, 'a') as f:
                f.write(pkl_path + '\n')

            if verbose:
                log.info(f'{os.path.split(pkl_path)[-1]} path has been added.\n')

    return flows


def read_dir(path, *, save_pkl=True,
                      annotate_path='',
                      remove_dummies=False,
                      return_flows_list=False) -> (dict, int):
    count = 0
    if return_flows_list:
        flows_list = []

    for (path, dirs, files) in os.walk(path):
        for file_name in files:
            file_path = os.path.join(path, file_name)
            if check_path_type(file_path) != 'cap':
                continue

            flows = read_cap(file_path, save_pkl=save_pkl,
                                        annotate_path=annotate_path,
                                        remove_dummies=remove_dummies)
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
                                  annotate_path='',
                                  remove_dummies=False):
    path_type = check_path_type(dir_or_file_path)
    if path_type == 'dir':
        _, count = read_dir(dir_or_file_path, save_pkl=save_pkl,
                                              annotate_path=annotate_path,
                                              remove_dummies=remove_dummies,
                                              return_flows_list=False)
        return count

    elif path_type == 'cap':
        flows = read_cap(dir_or_file_path, save_pkl=save_pkl,
                                           annotate_path=annotate_path,
                                           remove_dummies=remove_dummies)
        return 1

    else:
        log.info('error: invalid path')
        return 0

if __name__ == '__main__':

    conf = parse_config()

    path = conf['pkl_maker']['cap_file_or_directory']
    annotate = conf['pkl_maker']['save_annotation_path']
    save_pkl = conf['pkl_maker']['save_pkl']
    remove_dummies = conf['pkl_maker']['remove_dummies']

    log.info('Param info')
    log.info(f' - path: {path}')
    log.info(f' - annotate: {annotate}')
    log.info(f' - save_pkl: {save_pkl}')
    log.info(f' - remove_dummies: {remove_dummies}\n')

    log.info('Making pickles ...')

    count = make_pkl(path, save_pkl=save_pkl,
                           annotate_path=annotate,
                           remove_dummies=remove_dummies)

    if save_pkl:
        if count > 1:
            log.info(f'{count} pkl files have been made.')
        else:
            log.info(f'{count} pkl file has been made.')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
from pprint import pprint

import numpy as np
from keras.models import model_from_json

from classifier import preprocess, load_norm_params, model_compile
from data_maker import make_data_list_from_flow
from pkl_maker import read_cap

from utils import logger, refine_features, parse_config

log = logger(name=__name__,
             level=logging.INFO)


if __name__ == '__main__':

    conf = parse_config()

    load_model_path = conf['predict']['load_model_path']
    load_weight_path = conf['predict']['load_weight_path']
    load_norm_param_path = conf['predict']['load_norm_param_path']
    load_cap_path = conf['predict']['load_cap_path']

    log.info('Loading model ...')
    with open(load_model_path, 'r') as json_file:
        model_json =  json_file.read()

    model = model_from_json(model_json)
    model = model_compile(model)
    model.load_weights(load_weight_path)

    log.info('Loading capture file ...')
    flow = read_cap(load_cap_path, save_pkl=False,
                                   remove_dummies=True)

    min_packet_size = conf["data_maker_options"]["min_packet_size"]
    split_num = conf["data_maker_options"]["split_num_on_pld"]
    step_size = conf["data_maker_options"]["step_size"]
    features_to_use = refine_features(conf["data_maker_options"]["features"])

    data_list = make_data_list_from_flow(flow,
                                         min_packet_size=min_packet_size,
                                         split_num_on_pld=split_num,
                                         step_size=step_size,
                                         features_to_use=features_to_use)

    log.info('Loading norm params ...')
    norm_params = load_norm_params(load_norm_param_path)

    log.info('Preprocessing dataset ...')
    data_list, _ = preprocess(data_list, shuffle=False,
                                         normalize=True,
                                         is_train_data=False,
                                         norm_params=norm_params)

    log.info('Predicting ...\n')
    result = model.predict_classes(data_list)

    pos_count = np.count_nonzero(result == 1)
    neg_count = np.count_nonzero(result == 0)

    log.info('Predict result')
    log.info(f'pos_count: {pos_count}')
    log.info(f'neg_count: {neg_count}')

    if pos_count > neg_count:
        log.info('   => Camera detected')
    elif pos_count < neg_count:
        log.info('   => Non-camera detected')
    else:
        log.info('   => don\'t know')

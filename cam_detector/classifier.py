import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import pickle
from collections import defaultdict, namedtuple
from pprint import pprint

import numpy as np
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from data_maker import make_data_list
from utils import logger, refine_features, parse_config

log = logger(name=__name__,
             level=logging.INFO)

NormParam = namedtuple('NormParam', 'max, min')


def model_compile(model):
    model.compile(optimizer=Adam(lr=0.001),
                  #optimizer=SGD(lr=0.01, momentum=0.9, decay=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def build_model(input_num):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(input_num,)))
    model.add(BatchNormalization())

    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(24, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    model = model_compile(model)

    return model


def get_data(label, *, paths=[],
                       min_packet_size=300,
                       split_num_on_pld=50,
                       step_size=300,
                       features_to_use=None):

    x = []
    for path in paths:
        with open(path, 'r') as f:
            data_path_list = f.read().splitlines()

        data_list = make_data_list(data_path_list,
                                   min_packet_size=min_packet_size,
                                   split_num_on_pld=split_num,
                                   step_size=step_size,
                                   features_to_use=features_to_use)

        x.extend(data_list)

    if label in [True, 'true', 'True', 1, '1', 't', 'pos', 'positive']:
        y = [1]*len(x)
    elif label in [False, 'false', 'False', 0, '0', 'f', 'neg', 'negative']:
        y = [0]*len(x)
    else:
        raise ValueError("Check label on get_data()")

    if len(x) != len(y):
        raise ValueError("len(x) != len(y)")

    return x, y


def save_norm_params(norm_params, path):
    with open(path, 'w') as f:
        for key, norm_param in norm_params.items():
            f.write(f'{key}\t{norm_param.max}\t{norm_param.min}\n')


def load_norm_params(path):
    norm_params = {}

    with open(path, 'r') as f:
        norm_param_list = f.read().splitlines()

    for norm_param in norm_param_list:
        field, max_, min_ = norm_param.split('\t')
        norm_params[field] = NormParam(float(max_), float(min_))

    return norm_params


def preprocess(x_data, y_data=None, *, normalize,
                                       shuffle,
                                       is_train_data,
                                       norm_params):
    if normalize:
        normed_data_list = []
        for field in x_data[0]._fields:
            data_list = []
            for x in x_data:
                data_list.append(getattr(x, field))

            if is_train_data:
                max_ = max(data_list)
                min_ = min(data_list)
                norm_params[field] = NormParam(max_, min_)

                normed = []
                for data in data_list:
                    normed.append((data-min_)/(max_-min_))

            else:
                max_ = norm_params[field].max
                min_ = norm_params[field].min

                normed = []
                for data in data_list:
                    normed.append((data-min_)/(max_-min_))

            normed_data_list.append(normed)

        x_data = np.array(normed_data_list, dtype=np.float32).transpose()

    else:
        x_data = np.array(x_data, dtype=np.float32)

    if y_data is not None:
        y_data = np.array(y_data, dtype=np.float32)

    if shuffle:
        indices = np.arange(x_data.shape[0])
        np.random.shuffle(indices)

        x_data = x_data[indices]
        if y_data is not None:
            y_data = y_data[indices]

    return x_data, y_data


if __name__ == '__main__':

    conf = parse_config()

    log.info('Load dataset ...')

    min_packet_size = conf["data_maker_options"]["min_packet_size"]
    split_num = conf["data_maker_options"]["split_num_on_pld"]
    step_size = conf["data_maker_options"]["step_size"]
    features_to_use = refine_features(conf["data_maker_options"]["features"])

    # Get Train data
    x_pos, y_pos = get_data(1, paths=conf['train']['dataset']['train']['pos'],
                               min_packet_size=min_packet_size,
                               split_num_on_pld=split_num,
                               step_size=step_size,
                               features_to_use=features_to_use)

    x_neg, y_neg = get_data(0, paths=conf['train']['dataset']['train']['neg'],
                               min_packet_size=min_packet_size,
                               split_num_on_pld=split_num,
                               step_size=step_size,
                               features_to_use=features_to_use)

    x = x_pos + x_neg
    y = y_pos + y_neg

    # Split data into train and valid
    train_ratio = conf['train']['dataset']['ratio']['train']
    valid_ratio = conf['train']['dataset']['ratio']['valid']
    test_ratio = conf['train']['dataset']['ratio']['test']

    split_ratio = (valid_ratio+test_ratio)/(train_ratio+valid_ratio+test_ratio)
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=split_ratio)

    x_test = []
    y_test = []

    if test_ratio != 0:
        split_ratio = test_ratio/(valid_ratio+test_ratio)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                        test_size=split_ratio)

    # Get Test data if the paths exist
    x_pos_test, y_pos_test = get_data(1,
                                paths=conf['train']['dataset']['test']['pos'],
                                min_packet_size=min_packet_size,
                                split_num_on_pld=split_num,
                                step_size=step_size,
                                features_to_use=features_to_use)

    x_neg_test, y_neg_test = get_data(0,
                                paths=conf['train']['dataset']['test']['neg'],
                                min_packet_size=min_packet_size,
                                split_num_on_pld=split_num,
                                step_size=step_size,
                                features_to_use=features_to_use)

    x_test_temp = x_pos_test + x_neg_test
    y_test_temp = y_pos_test + y_neg_test

    if x_test_temp and y_test_temp:
        x_test = x_test_temp
        y_test = y_test_temp

    norm_params = {}

    x_train, y_train = preprocess(x_train, y_train, shuffle=True,
                                                    normalize=True,
                                                    is_train_data=True,
                                                    norm_params=norm_params)

    x_val, y_val = preprocess(x_val, y_val, shuffle=False,
                                            normalize=True,
                                            is_train_data=False,
                                            norm_params=norm_params)

    if len(y_test) > 0:
        x_test, y_test = preprocess(x_test, y_test, shuffle=False,
                                                    normalize=True,
                                                    is_train_data=False,
                                                    norm_params=norm_params)

    if conf['train']['save_norm_param_path']:
        save_norm_params(norm_params, conf['train']['save_norm_param_path'])

    log.info('Build model ...')
    model = build_model(len(features_to_use))

    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       verbose=1,
                       patience=70)

    save = ModelCheckpoint(conf['train']['save_weight_path'],
                           monitor='val_accuracy',
                           mode='max',
                           verbose=0,
                           save_best_only=True)

    epochs = conf['train']['fit_options']['max_epoch']
    batch_size = conf['train']['fit_options']['batch_size']
    verbose = conf['train']['fit_options']['verbose']

    log.info('Train model ...\n')
    history = model.fit(x_train, y_train, epochs=epochs,
                                          batch_size=batch_size,
                                          validation_data=(x_val, y_val),
                                          verbose=verbose,
                                          callbacks=[es, save],
                                          shuffle=True)

    log.info("Save model ...\n")
    with open(conf['train']['save_model_path'], 'w') as json_file:
        json_file.write(model.to_json())

    if len(y_test) > 0:
        log.info('Evaluate model ...\n')
        results = model.evaluate(x_test, y_test, batch_size=128, verbose=0)

    log.info('----------------- Summary -----------------')
    log.info('Data info')
    di_headers = ['Dataset', 'Total', 'Pos', 'Neg']
    di_bodies = [('Train', len(y_train), np.count_nonzero(y_train == 1),
                                         np.count_nonzero(y_train == 0)),
                 ('Valid', len(y_val), np.count_nonzero(y_val == 1),
                                       np.count_nonzero(y_val == 0)),
                 ('Test', len(y_test), np.count_nonzero(y_test == 1),
                                       np.count_nonzero(y_test == 0)),
                 ('All', len(y), y.count(1), y.count(0))]

    print(tabulate(di_bodies, headers=di_headers, tablefmt='grid'))
    print()

    log.info('Train result')
    re_headers = ['Dataset', 'Num', 'Accuracy']
    re_bodies = [('Train', len(y_train), history.history['accuracy'][-1]),
                 ('Valid', len(y_val), history.history['val_accuracy'][-1]),
                 ('Test', len(y_test), results[1] if len(y_test) > 0 else '-')]

    print(tabulate(re_bodies, headers=re_headers, tablefmt='grid'))

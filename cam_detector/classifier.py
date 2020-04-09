import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import defaultdict, namedtuple
from pprint import pprint

import numpy as np
from keras.callbacks.callbacks import EarlyStopping
from keras.models import Sequential
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop, Nadam, Adamax
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from data_maker import make_data_list
from logger import log


norm_params = {}

def build_model():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(10,)))
    model.add(BatchNormalization())

    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(24, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.001),
                  #optimizer=SGD(lr=0.01, momentum=0.9, decay=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def get_data():
    with open('data_path/pos_nm_up_paths.txt', 'r') as f:
        pos_path_list = f.read().splitlines()
    x_pos = make_data_list(pos_path_list)
    y_pos = [1]*len(x_pos)

    with open('data_path/neg_paths.txt', 'r') as f:
        neg_path_list = f.read().splitlines()
    x_neg = make_data_list(neg_path_list)

    with open('data_path/neg_dummy_paths.txt', 'r') as f:
        neg_dummy_path_list = f.read().splitlines()
    x_neg_dummy = make_data_list(neg_dummy_path_list)
    x_neg.extend(x_neg_dummy)
    y_neg = [0]*len(x_neg)

    x_data = x_pos + x_neg
    y_data = y_pos + y_neg

    if len(x_data) != len(y_data):
        raise ValueError

    return x_data, y_data


def preprocess(x_data, y_data, *, normalize, shuffle, is_train_data):
    if normalize:
        normed_data_list = []
        NormParam = namedtuple('NormParam', 'max, min')
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

    y_data = np.array(y_data, dtype=np.float32)

    if shuffle:
        indices = np.arange(x_data.shape[0])
        np.random.shuffle(indices)

        x_data = x_data[indices]
        y_data = y_data[indices]

    return x_data, y_data


if __name__ == '__main__':

    log.info('Load dataset ...')
    x, y = get_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                               test_size=0.2)

    x_train, y_train = preprocess(x_train, y_train, shuffle=True,
                                                    normalize=True,
                                                    is_train_data=True)

    x_val, y_val = preprocess(x_val, y_val, shuffle=False,
                                            normalize=True,
                                            is_train_data=False)

    x_test, y_test = preprocess(x_test, y_test, shuffle=False,
                                                normalize=True,
                                                is_train_data=False)

    log.info('Build model ...')
    model = build_model()

    es = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       verbose=1,
                       patience=70)

    log.info('Train model ...')
    print()
    history = model.fit(x_train, y_train, epochs=300,
                                          batch_size=128,
                                          validation_data=(x_val, y_val),
                                          verbose=1,
                                          callbacks=[es],
                                          shuffle=True)

    print()
    log.info('Evaluate model ...')
    results = model.evaluate(x_test, y_test, batch_size=128, verbose=0)

    print()
    log.info('----------------- Summary -----------------')
    log.info('Data info')
    di_headers = ['Dataset', 'Total', 'Pos', 'Neg']
    di_bodies = [('Train', len(y_train), np.count_nonzero(y_train == 1), np.count_nonzero(y_train == 0)),
                 ('Valid', len(y_val), np.count_nonzero(y_val == 1), np.count_nonzero(y_val == 0)),
                 ('Test', len(y_test), np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 0)),
                 ('All', len(y), y.count(1), y.count(0))]

    print(tabulate(di_bodies, headers=di_headers, tablefmt='grid'))
    print()

    log.info('Result')
    re_headers = ['Dataset', 'Accuracy']
    re_bodies = [('Train', history.history['accuracy'][-1]),
                 ('Valid', history.history['val_accuracy'][-1]),
                 ('Test', results[1])]

    print(tabulate(re_bodies, headers=re_headers, tablefmt='grid'))





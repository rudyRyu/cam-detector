
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split

from data_maker import make_data_list


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
            print(field)
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

    x, y = get_data()










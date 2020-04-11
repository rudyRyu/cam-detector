import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pprint import pprint

import numpy as np

from classifier import build_model, preprocess, load_norm_params
from data_maker import make_data_list_from_flow
from pkl_maker import read_cap


if __name__ == '__main__':
    model = build_model()
    model.load_weights('saved_models/weights.hdf5')
    cap_path = '/Users/rav/Desktop/Development/Virtualenv/CameraDetector/project/cam-detector/data/20200327/camera/L100MN_camera_up/EasyN_2_1C-BF-CE-3D-5E-E5_up.pcap'

    flow = read_cap(cap_path, save_pkl=False,
                              remove_dummies=True)

    data_list = make_data_list_from_flow(flow)

    norm_params = load_norm_params('norm_params.txt')

    data_list, _ = preprocess(data_list, shuffle=False,
                                         normalize=True,
                                         is_train_data=False,
                                         norm_params=norm_params)

    result = model.predict_classes(data_list)

    pos_count = np.count_nonzero(result == 1)
    neg_count = np.count_nonzero(result == 0)

    print('pos_count:', pos_count)
    print('neg_count:', neg_count)

    if pos_count > neg_count:
        print('   => Camera detected')
    elif pos_count < neg_count:
        print('   => Non-camera detected')
    else:
        print('   => don\'t know')

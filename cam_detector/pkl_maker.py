import os
import pickle
from collections import defaultdict
from pprint import pprint

import pyshark

from utils import ddict2dict


for (path, dirs, files) in os.walk('data/non-camera'):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext not in ['.pcap', '.cap']:
            continue

        cap = pyshark.FileCapture(os.path.join(path, filename))
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
                print('go')
                continue

            flows[src][dst][fc].append(data)

        cap.close()

        flows = ddict2dict(flows)
        pkl_name = f'{path}/{os.path.splitext(filename)[0]}.pkl'
        with open(pkl_name, 'wb') as fp:
            pickle.dump(flows, fp)
        print(pkl_name, 'done')


# Installation

### Prerequisites
```
Python 3.7+
```

### Install
```
pip3 install -r requirements.txt
```

# Run
### 1. Convert pcap file to pickle (pcap -> pkl)
```
// option
[-p PATH] Packet file path (file or directory)
[-a PATH] File path that saves pkl paths (Append)
[-s] Whether to save pickle files or not
[-r] Delete dummy packets (only one pair of src and dst is left)

// command
$ python3 pkl_maker.py -p [pcap file or directory] -a [where to save pkl paths] [-s] [-r]

// example
$ python3 pkl_maker.py -p ../data/old/etc -a pos_test.txt -s -r
$ python3 pkl_maker.py -p ../data/dummy_test -a neg_test.txt -s
$ python3 pkl_maker.py -p ../data/dummy/test_file.pcap -a neg_test.txt -s
```

### 2. Data analysis
```
// command
$ python3 data_analysis.py

// Set data path in main func in data_analysis.py
if __name__ == '__main__':
    with open('pos_test.txt', 'r') as f:
      ...

    with open('neg_test.txt', 'r') as f:
      ...
```

### 3. Train
```
// command
$ python3 classifier.py

// Set data path in get_data func in classifier.py
def get_data():
    with open('pos_test.txt', 'r') as f:
      ...

    with open('neg_test.txt', 'r') as f:
      ...

```

### 4. Prediction
```
// command
$ python predict.py

// Set weight path, cap_path, param_path in main func in predict.py
if __name__ == '__main__':
	...

    model.load_weights('saved_models/weights.hdf5')
    cap_path = '...../camera.pcap'

    ...

    norm_params = load_norm_params('norm_params.txt')
```

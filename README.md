# This is an implementation of a wireless camera detector.

- Reference: [On Detecting Hidden Wireless Cameras: A Traffic Pattern-based Approach
](https://ieeexplore.ieee.org/document/8648293)

# Installation

### Prerequisites
```
Python 3.7+
```

### Install
```
pip3 install -r requirements.txt
```

# How to Use
### 1. Convert pcap file to pickle (pcap -> pkl)

 - Configuration (config.json)
```
{
      "pkl_maker": {
          // Packet file path (file or directory)
          "cap_file_or_directory": "data/camera/hd/",

          // Whether to save pickle files or not
          "save_pkl": true,

          // File path that saves pkl paths (Append)
          "save_annotation_path": "data_path/example.txt",

          // Delete dummy packets (only one pair of src and dst will be left)
          "remove_dummies": True
      }
}
```

- Run
```
$ python3 pkl_maker.py -c config.json
```

### 2. Data analysis
 - Configuration (config.json)
 ```
{
      "data_maker_options":{
          // Features to use
          "features":["length_avg",
                      "pld_stat_stb",
                      "pld_pval_stb",
                      "pld_stat_stb_with_cdf",
                      "pld_pval_stb_with_cdf",
                      "duration_std",
                      "duration_avg"
                      ],

          // The number of packets that compose one data
          "min_packet_size": 300,

          // split size on pld features
          "split_num_on_pld": 50,

          "step_size": 300
      },

      "data_analysis": {
          "dataset":[
              // [data name, data path]
              ["pos", "data_path/pos_nm_up_paths.txt"],
              ["neg", "data_path/neg_paths.txt"],
              ["neg_dummy", "data_path/neg_dummy_paths.txt"]
          ]
      }
}
 ```

- Run
```
$ python3 data_analysis.py -c config.json
```

### 3. Train
- Configuration (config.json)
```
{
      "data_maker_options":{
          // same as above
          ...
      },

      "train": {
          "save_model_path": "saved_models/model.json",
          "save_weight_path": "saved_models/weight.hdf5",
          "save_norm_param_path": "saved_models/norm_params.txt",

          "dataset":{
              "train":{
                  "pos": ["data_path/pos_nm_up_paths.txt"],
                  "neg": ["data_path/neg_paths.txt",
                          "data_path/neg_dummy_paths.txt"]
              },

              "test":{
                  "pos": [],
                  "neg": []
              },

              // Split dataset into Train/Validation/Test
              // If you use your own Test dataset, set test ratio to 0
              "split_ratio":{
                  "train": 0.64,
                  "valid": 0.16,
                  "test": 0.2
              }
          },

          // Train options
          "fit_options": {
              "max_epoch": 300,
              "batch_size": 128,
              "verbose": 1
          }
      },
}
```

- Run
```
$ python3 classifier.py -c config.json
```

### 4. Predict
- Configuration (config.json)
```
{
      "data_maker_options":{
          // same as above
          ...
      },

      "predict": {
          "load_model_path": "saved_models/model.json",
          "load_weight_path": "saved_models/weight.hdf5",
          "load_norm_param_path": "saved_models/norm_params.txt",

          "load_cap_path": "data/xwfscan.pcap"
      }
}
```

- Run
```
$ python3 predict.py -c config.json
```

### * Features list
```
"length_avg",
"pld_stat_stb",
"pld_pval_stb",
"pld_stat_stb_with_cdf",
"pld_pval_stb_with_cdf",
"duration_std",
"duration_avg",
"bandwidth_std",
"bandwidth_avg",
"flow_bandwidth"
```

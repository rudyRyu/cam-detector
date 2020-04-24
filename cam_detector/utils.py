import argparse
import json
import logging

def logger(*, name='',
              level=logging.INFO,
              formatter=''):
    if name:
        log = logging.getLogger(name)
    else:
        log = logging.getLogger(__name__)

    handler = logging.StreamHandler()
    # handler.setFormatter(logging.Formatter(
    #                         '%(asctime)-15s [%(levelname)s] %(message)s',
    #                         '%Y-%m-%d %H:%M:%S'))

    if formatter:
        handler.setFormatter(logging.Formatter(formatter))
    else:
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))

    log.addHandler(handler)
    log.setLevel(level)

    return log

log = logger()

def parse_config():
    argparser = argparse.ArgumentParser(description='config')
    argparser.add_argument(
        '-c',
        '--conf',
        required=True,
        help='path to a configuration file')


    args = argparser.parse_args()
    config_path = args.conf

    with open(config_path) as config_buffer:
        conf = json.loads(config_buffer.read())

    return conf


def refine_features(features):

    all_features = ["length_avg",
                    "pld_stat_stb",
                    "pld_pval_stb",
                    "pld_stat_stb_with_cdf",
                    "pld_pval_stb_with_cdf",
                    "duration_std",
                    "duration_avg",
                    "bandwidth_std",
                    "bandwidth_avg",
                    "flow_bandwidth"]

    refined_features = list(set(all_features) & set(features))

    if len(features) != len(refined_features):
        log.info("Feature list has been refined.")
        log.info("----------------------------")
        log.info("Refined feature list")
        for feature in refined_features:
            log.info(f" - {feature}")
        log.info("----------------------------")


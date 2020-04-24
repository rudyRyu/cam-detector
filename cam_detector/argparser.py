import argparse
import json

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

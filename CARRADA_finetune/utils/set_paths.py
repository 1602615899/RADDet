"""Script to set the path to CARRADA in the config.ini file"""
import argparse
import os
from configurable import Configurable
from models.CARRADA_finetune.utils import MVRSS_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--carrada', default='/mnt/truenas_datasets/Datasets_mmRadar/',
                        help='Path to the CARRADA dataset.')
    parser.add_argument('--logs', default='/mnt/truenas_users/ChenHongliang/TransRadar/mvrss/carrada_logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    config_path = MVRSS_HOME / 'config_files' / 'config.ini'
    if not os.path.isfile(config_path):
        f = open(config_path,"w")
        f.write("""[data]
warehouse = 
logs =""")
        f.close()
    configurable = Configurable(config_path)
    configurable.set('data', 'warehouse', args.carrada)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)

import os
import time
import subprocess
import argparse
from loguru import logger
from scripts import *
from hydra import compose, initialize
from loguru import logger

from util.preprocess import *
from util.helper_functions import set_manual_seed, create_directories, save_run_info

set_manual_seed()

def main(config_files):
    
    train_set, val_set = pd.DataFrame(), pd.DataFrame()
    train_test_set, dl19_test_set, dl20_test_set = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    initialize(config_path="config")
    for config_file in config_files:
        
        cfg = compose(config_name=f"{config_file}")

        create_directories(
            cfg.directories
        )
        
        logger.info(f"Loading train_set for {config_file}")
        if train_set.empty:
            start_time = time.time()
            train_set = load_dataset(cfg.train_config, "TRAIN")
        logger.info(f"Loaded train_set for {config_file} in {time.time() - start_time} seconds")

        logger.info(f"Loading val_set for {config_file}")

        print(train_set.info())
        print(train_set.head(2))    

        train_start_time = time.time()
        train_model(cfg, train_set, train_set)
        train_end_time = time.time()
        train_run_time = train_end_time - train_start_time
        logger.info("original df")

        train_run_time = 0
        # ------------cv--------------
        logger.info(f"Loading  train_test_set for {config_file}")
        train_test_set = train_set

        qid_list = []
        qid_list_path = cfg.train_config.qid_train_list_path

        with open(qid_list_path, 'r') as file:
            for line in file:
                number = int(line.strip())
                qid_list.append(str(number))

        train_test_set = train_test_set[train_test_set['qid'].isin(qid_list)]
        print(train_test_set["qid"].unique())
        print(train_test_set)

        if train_test_set.empty:
            start_time = time.time()
            train_test_set = load_dataset(cfg.eval_config, "TRAIN_TEST")
        logger.info(f"Loaded train_test_set for {config_file} in {time.time() - start_time} seconds")
        
        eval_start_time = time.time()
        eval_model(cfg.eval_config, train_test_set, mode="TRAIN")
        eval_end_time = time.time()
        eval_run_time = eval_end_time - eval_start_time

        save_run_info(cfg.train_config, train_run_time, eval_run_time)
        # # # --------------
        logger.info(f"Loading DL19_test_set for {config_file}")
        if dl19_test_set.empty:
            start_time = time.time()
            dl19_test_set = load_dataset(cfg.eval_config, "DL19")
        logger.info(f"Loaded DL19_test_set for {config_file} in {time.time() - start_time} seconds")
        
        eval_start_time = time.time()
        eval_model(cfg.eval_config, dl19_test_set, mode='DL19')
        eval_end_time = time.time()
        eval_run_time = eval_end_time - eval_start_time

        save_run_info(cfg.train_config, train_run_time, eval_run_time)
        # # # --------------
        logger.info(f"Loading DL20_test_set for {config_file}")
        if dl20_test_set.empty:
            start_time = time.time()
            dl20_test_set = load_dataset(cfg.eval_config, "DL20")
        logger.info(f"Loaded dl20_test_set for {config_file} in {time.time() - start_time} seconds")
        
        eval_start_time = time.time()
        eval_model(cfg.eval_config, dl20_test_set, mode='DL20')
        eval_end_time = time.time()
        eval_run_time = eval_end_time - eval_start_time

        save_run_info(cfg.train_config, train_run_time, eval_run_time)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('config_files', metavar='N', type=str, nargs='+',
                        help='config files to be processed')
    args = parser.parse_args()

    main(args.config_files)
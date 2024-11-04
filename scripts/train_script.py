import argparse
import time
import random
import os
from tqdm import tqdm
from loguru import logger
from hydra import compose, initialize
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from model.dqn import DQNAgent
from model.mdp import BasicBuffer
from util.preprocess import *
from util.helper_functions import set_manual_seed
from util.plot_results import save_and_plot_results

set_manual_seed()

def train_model(cfg, train_set=None, valid_set=None):

    logger.info(f"Training in {cfg.run_mode} mode")

    train_cfg = cfg.train_config

    writer = SummaryWriter(log_dir=f'{train_cfg.output_folder}/tensorboard')

    train_set = train_set if train_set is not None and not train_set.empty else load_dataset(train_cfg)

    train_buffer = BasicBuffer(train_cfg.run_params.buffer_qid * 15)
    train_buffer.push_batch(train_cfg.qid_train_list_path, train_set, train_cfg.reward_params,  train_cfg.run_params.buffer_qid)

    agent = DQNAgent(dataset=train_set, buffer=train_buffer, config_dict = train_cfg.model_config_dict)
    
    training_metrics = {'train_loss': [], 'expectedQ': []}

    train_loss_list, curr_Q_list, expected_Q_list, reward_batches = [], [], [], []

    for epoch in tqdm(range(train_cfg.run_params.epochs)):

        if epoch % 100000 == 0:
            model_temp_path = "/".join(train_cfg.model_path.split("/")[:-1])
            model_temp_path = f"{model_temp_path}/model_{epoch}.pth"
            torch.save(agent.model.state_dict(), model_temp_path)

        loss, curr_Q, expected_Q, reward_batch, done_batch = agent.update(1, normalized=train_cfg.run_params.normalized)

        training_metrics['train_loss'].append(loss)
        training_metrics['expectedQ'].append(expected_Q)

        writer.add_scalars('Q-Values', {'Current': curr_Q.item(), 'Expected': expected_Q.item()}, epoch)

        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Reward/train', expected_Q, epoch)


    for metric_name, data in training_metrics.items():
        data = [float(x) for x in data]
        save_and_plot_results(data, train_cfg.window, train_cfg.output_folder, train_cfg.plot_folder, metric_name)

    writer.close() 
    
def main():

    parser = argparse.ArgumentParser(description="Running train_script")
    parser.add_argument("--conf", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.conf:
        config_file = args.conf
        logger.info(f"Config file name: {config_file}")
    else:
        logger.info(
            "Please provide the name of the config file using the --conf argument. \nExample: --conf config.yaml"
        )
        return

    initialize(config_path="config")
    cfg = compose(config_name=f"{config_file}")

    create_directories(cfg.output_paths)

    random.seed(cfg.constants.seed)

    start_time = time.time()
    train_model(cfg)
    end_time = time.time()
    train_run_time = end_time - start_time
    save_run_info(cfg.train_config, train_run_time, train_run_time)

    logger.info("Finished Training Successfully.")

if __name__ == "__main__":
    main()

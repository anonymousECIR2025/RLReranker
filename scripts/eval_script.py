import argparse
import os
import time
from loguru import logger
from hydra import compose, initialize
import torch
from torch.utils.tensorboard import SummaryWriter

from util import *
from model.dqn import DQN, DQNAgent
from util.helper_functions import set_manual_seed

set_manual_seed()

def eval_model(eval_cfg, test_set=None, mode="TRAIN"):

    logger.info(f"Testing in {eval_cfg.run_mode} mode")
    writer = SummaryWriter(log_dir=f'{eval_cfg.output_folder}/tensorboard')

    modes = {
        'DL19': ("dl19", eval_cfg.dl19_eval_trec_path, eval_cfg.dl19_eval_output_path, eval_cfg.dl19_qrel_file ),
        'DL20': ("dl20", eval_cfg.dl20_eval_trec_path, eval_cfg.dl20_eval_output_path, eval_cfg.dl20_qrel_file ),
    }

    for mode_value, settings in modes.items():
        if mode == mode_value:
            trec_mode, trec_path, output_path, qrel_file = settings
            break

    logger.info(f"EVALUATING IN {trec_mode} MODE")
    if test_set is None or test_set.empty:
        test_set = load_dataset(eval_cfg)

    features = eval_cfg.model_config_dict.features
    if features:
        len_features = len(features)
    else:
        len_features = 0

    input_dim = eval_cfg.model_config_dict.input_dim + len_features

    output_dim = eval_cfg.model_config_dict.output_dim
    model_size = eval_cfg.model_config_dict.model_size
    model = DQN(input_dim, output_dim, model_size)
    normalized = eval_cfg.run_params.normalized
    #----------------------------------------------
    NDCG10_input = calculate_ndcg(qrel_file, test_set)

    models_dir = os.path.dirname(eval_cfg.pretrained_model_path)
    model_files = sorted(
        [f for f in os.listdir(models_dir) if f.endswith('100000.pth') or f.endswith('200000.pth')],

        key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else float('inf')
    )

    NDCG_list, mse_list, formatted_bias_values_list = [], [], []

    for model_file in model_files:

        logger.info(f"Evaluating {model_file}")
        model_path = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(model_path))
        current_trec_path = f"{trec_path}_{model_file}.txt"
        
        agent = DQNAgent(dataset=None, buffer=None, config_dict = eval_cfg.model_config_dict, pre_trained_model=model)

        ndcg, mse, formatted_bias_values  = calculate_metrics(agent, test_set, qrel_file, features, normalized, current_trec_path)
        print(f"==============={model_file}\n===============")
        NDCG_list.append(ndcg)
        mse_list.append(mse)
        formatted_bias_values_list.append(formatted_bias_values)

        log_performance(model_file, NDCG_list, mse_list, formatted_bias_values_list, eval_cfg.plot_folder, trec_mode, output_path)

        writer.add_scalar(f'MSE_{trec_mode}', mse, len(mse_list))
        print("---------------------------------")

    
    writer.close()

    mse_list_path = f"{eval_cfg.output_folder}/{trec_mode}_mse_list.txt"
    with open(mse_list_path, 'w') as file:
        for item in mse_list:
            file.write(f"{item}\n")



def calculate_metrics(agent, test_set, qrel_file, features, normalized, trec_path):

    mse_loss = write_trec_results(agent, test_set, features, normalized, trec_path)
    NDCG10_output = calculate_ndcg(qrel_file, trec_path)

    return NDCG10_output, mse_loss, formatted_bias_values

def log_performance(model_file, NDCG_list, mse_list, formatted_bias_values_list, plot_folder, trec_mode, output_path):

    with open(output_path, "a+") as f:
        f.write(f"================{model_file}================\n")
        f.write(f"NDCG@10:\t{NDCG_list[-1]}\n")
        f.write(f"MSE:\t{mse_list[-1]}\n")
        f.write(f"=========================================================\n")

    plot_metric(NDCG_list, f"{plot_folder}/{trec_mode}_NDCG.png", label=f"NDCG values on {trec_mode} set")
    plot_metric(mse_list, f"{plot_folder}/{trec_mode}_mse.png", label=f"MSE loss values on {trec_mode} set")


def main():

    parser = argparse.ArgumentParser(description="Running eval_script")
    parser.add_argument("--conf", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.conf:
        config_file = args.conf
        logger.info(f"Config file name: {config_file}")
    else:
        logger.info(
            "Please provide the name of the config file using the --conf argument. \nExample: --conf rank.yaml"
        )

    initialize(config_path="config")
    cfg = compose(config_name=f"{config_file}")

    start_time = time.time()
    eval_model(cfg.eval_config)
    end_time = time.time()

    eval_run_time = end_time - start_time
    save_run_info(cfg.train_config, 0, eval_run_time)

    logger.info("Finished Evaluating Model Successfully in {eval_run_time} seconds.")

if __name__ == "__main__":
    main()

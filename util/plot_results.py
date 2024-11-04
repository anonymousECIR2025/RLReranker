import torch
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt


def save_and_plot_results(numbers, window, results_folder, plot_folder, stage):

    with open(f"{results_folder}/losses/{stage}.txt", 'w+') as f:
        f.write(str(numbers))

    plot_loss(numbers, f"{plot_folder}/{stage}.png", label=f"{stage}")
    plot_loss(pd.Series(numbers).rolling(window=window).mean().tolist(), f"{plot_folder}/{stage}_MA.png", label=f"{stage} Moving Average")
    
    print("results_folder", results_folder)

def plot_MA_log10(numbers: List, window: int, plot_name: str, label = ""):

    plt.figure(figsize=(10, 6))

    moving_avg = np.convolve(np.log10(numbers), np.ones(window) / window, mode='valid')
    plt.plot(moving_avg)
    plt.grid(True)
    plt.legend()
    plt.title(label)
    plt.savefig(plot_name)
    plt.close()  

def plot_loss(numbers: List, plot_name: str, label = ""):

    if torch.is_tensor(numbers):
        numbers = numbers.detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(numbers)
    plt.grid(True)
    plt.legend()
    plt.title(label)
    plt.savefig(plot_name)
    plt.close()  


def plot_metric(metric_list, plot_name, label):

    plt.figure(figsize=(10, 6))
    
    x_values = list(range(len(metric_list)))
    plt.plot(x_values, metric_list, marker='o')  # 'o' is for circle markers

    plt.title(label)
    plt.xlabel('Epoch/1000')
    plt.ylabel('metric value')
    plt.savefig(plot_name)
    plt.close()  


def plot_metric_comparison(metric1, label1, metric2, label2, title, plot_name):

    plt.figure(figsize=(20, 14))
    plt.scatter(metric1, metric2, color='blue') 

    labels = list(range(len(metric2)))
    for i, label in enumerate(labels):
        plt.text(metric1[i], metric2[i], label)

    plt.title(title)
    plt.xlabel(label1)
    plt.ylabel(label2)

    plt.savefig(plot_name, format='png', dpi=300)
    plt.clf()

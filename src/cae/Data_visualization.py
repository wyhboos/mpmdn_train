import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
import cv2
import matplotlib.pyplot as plt
def vis_loss(save_fig_dir, train_loss, test_loss, local_range=None):
    train_loss_file = save_fig_dir + "_train.png"
    train_loss_local_file = save_fig_dir + "_train_local.png"
    all_loss_file = save_fig_dir + "_all.png"
    all_loss_local_file = save_fig_dir + "_all_local.png"

    # plot the whole loss
    fig = plt.figure(figsize=(10, 10))
    plt.plot(train_loss, label="Train loss")
    plt.savefig(train_loss_file)
    plt.plot(test_loss, label="test loss")
    plt.legend()
    plt.savefig(all_loss_file)

    # plot local loss
    if local_range is not None:
        l = len(train_loss)
        fig = plt.figure(figsize=(10, 10))
        plt.plot(train_loss[l - local_range:], label="Train loss")
        plt.savefig(train_loss_local_file)
        plt.plot(test_loss[l - local_range:], label="test loss")
        plt.legend()
        plt.savefig(all_loss_local_file)
    plt.close('all')


def plot_env_scatter(save_fig_path, env_cloud):
    env_cloud = env_cloud.reshape(-1, 2)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(env_cloud[:, 0], env_cloud[:, 1], s=0.5)
    plt.savefig(save_fig_path+'.png')
    plt.close('all')
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math
import os
import torch
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def org_to_img(x, y, pic_shape, ppm):
    aixs1 = pic_shape[0] - 1 - y * ppm
    aixs2 = x * ppm
    return int(aixs1 + 0.5), int(aixs2 + 0.5)


def img_to_org(aixs1, aixs2, pic_shape, ppm):
    x = aixs2 / ppm
    y = pic_shape[0] - 1 - aixs1 / ppm
    return x, y


def plot_nearby(pic, aixs1, aixs2, r, color):
    if color == 'red':
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if 0 < aixs1 + i < pic.shape[0] and 0 < aixs2 + j < pic.shape[1]:
                    pic[aixs1 + i, aixs2 + j, 0] = 0
                    pic[aixs1 + i, aixs2 + j, 1] = 0
    if color == 'blue':
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if 0 < aixs1 + i < pic.shape[0] and 0 < aixs2 + j < pic.shape[1]:
                    pic[aixs1 + i, aixs2 + j, 1] = 0
                    pic[aixs1 + i, aixs2 + j, 2] = 0
    if color == 'black':
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if 0 < aixs1 + i < pic.shape[0] and 0 < aixs2 + j < pic.shape[1]:
                    pic[aixs1 + i, aixs2 + j, 0] = 0
                    pic[aixs1 + i, aixs2 + j, 1] = 0
                    pic[aixs1 + i, aixs2 + j, 2] = 0
    if color == 'green':
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if 0 < aixs1 + i < pic.shape[0] and 0 < aixs2 + j < pic.shape[1]:
                    pic[aixs1 + i, aixs2 + j, 0] = 0
                    pic[aixs1 + i, aixs2 + j, 2] = 0
    if color == 'shallow_green':
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if 0 < aixs1 + i < pic.shape[0] and 0 < aixs2 + j < pic.shape[1]:
                    pic[aixs1 + i, aixs2 + j, 0] = 0
                    pic[aixs1 + i, aixs2 + j, 1] = 255
                    pic[aixs1 + i, aixs2 + j, 2] = 127

    if color == 'yellow':
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if 0 < aixs1 + i < pic.shape[0] and 0 < aixs2 + j < pic.shape[1]:
                    pic[aixs1 + i, aixs2 + j, 0] = 0
                    pic[aixs1 + i, aixs2 + j, 1] = 255
                    pic[aixs1 + i, aixs2 + j, 2] = 255

    return pic


# plot line
def plot_line(pic, plot_aixs, aixs1_range, aixs2_range, r, color):
    if plot_aixs == 1:
        aixs1_start = min(aixs1_range)
        aixs1_end = max(aixs1_range)
        for x in range(aixs1_start, aixs1_end + 1):
            pic = plot_nearby(pic=pic, aixs1=x, aixs2=aixs2_range[0], r=r, color=color)
    else:
        aixs2_start = min(aixs2_range)
        aixs2_end = max(aixs2_range)
        for y in range(aixs2_start, aixs2_end + 2):
            pic = plot_nearby(pic=pic, aixs1=aixs1_range[0], aixs2=y, r=r, color=color)
    return pic


def sample(alpha, sigma, mean):
    alpha = np.array(alpha)
    sigma = np.array(sigma)
    mean = np.array(mean)
    prior = np.random.random()
    k = alpha.shape[0]
    d = len(mean[0])
    for i in range(k):
        if i == 0:
            pro_l = 0
        else:
            pro_l = np.sum(alpha[0: i])
        if i == alpha.shape[0] - 1:
            pro_r = 1
        else:
            pro_r = np.sum(alpha[0: i + 1])
        if pro_l <= prior <= pro_r:
            break
    mean_i = mean[i]

    sigma_i = np.diag([sigma[i] for j in range(d)])
    sample = np.random.multivariate_normal(mean=mean_i, cov=sigma_i)

    return sample


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


def plot_samples_absolute_for_global(fig, sp_cnt, sample_distribution, ppm, r):
    # sample and plot
    alpha = sample_distribution[0]
    sigma = sample_distribution[1]
    mean = sample_distribution[2]
    in_cnt = 0  # record points located inside the fig
    for i in range(sp_cnt):
        sp = sample(alpha, sigma, mean)
        x = sp[0] + 20
        y = sp[1] + 20
        aixs1, aixs2 = org_to_img(x, y, fig.shape, ppm)
        if 0 < aixs1 < fig.shape[0] and 0 < aixs2 < fig.shape[1]:
            in_cnt += 1
            fig = plot_nearby(fig, aixs1, aixs2, r=r, color="shallow_green")
    return fig, in_cnt / sp_cnt


def vis_origin_fig_global_Cloud_input(Cloud_env, cur_loc, goal_loc, next_loc, pixel_per_meter, pre_next_loc=None):
    l = 35 * pixel_per_meter
    fig = 255 * np.ones((int(l * 1.2), int(l * 1.2), 3), dtype=np.int8)
    # 整体向右边上边位移1米，方便打印
    # plot obstacles in env
    env_info = list(np.array(Cloud_env).reshape(-1, 2))
    for point in env_info:
        point_x = point[0] + 20
        point_y = point[1] + 20
        point_aixs1, point_aixs2 = org_to_img(point_x, point_y, fig.shape, pixel_per_meter)
        fig = plot_nearby(fig, point_aixs1, point_aixs2, r=2, color='black')

    # plot current location
    cur_x = cur_loc[0] + 20
    cur_y = cur_loc[1] + 20
    cur_aixs1, cur_aixs2 = org_to_img(cur_x, cur_y, fig.shape, pixel_per_meter)
    fig = plot_nearby(fig, cur_aixs1, cur_aixs2, r=5, color='blue')

    # plot goal
    goal_x = goal_loc[0] + 20
    goal_y = goal_loc[1] + 20
    goal_aixs1, goal_aixs2 = org_to_img(goal_x, goal_y, fig.shape, pixel_per_meter)
    fig = plot_nearby(fig, goal_aixs1, goal_aixs2, r=5, color='red')

    # plot next position
    next_x = next_loc[0] + 20
    next_y = next_loc[1] + 20
    next_aixs1, next_aixs2 = org_to_img(next_x, next_y, fig.shape, pixel_per_meter)
    fig = plot_nearby(fig, next_aixs1, next_aixs2, r=5, color='green')

    # plot pre next position
    if pre_next_loc is not None:
        pre_next_x = pre_next_loc[0] + 20
        pre_next_y = pre_next_loc[1] + 20
        pre_next_aixs1, pre_next_aixs2 = org_to_img(pre_next_x, pre_next_y, fig.shape, pixel_per_meter)
        fig = plot_nearby(fig, pre_next_aixs1, pre_next_aixs2, r=5, color='yellow')

    return fig


def plot_output_info_for_global(save_fig_file, pixel_per_meter, Cloud_env,
                                cur_loc, goal_loc, next_loc, sample_cnt=None, sample_distribution=None,
                                tensorboard_writer=None, fig_cat=None, epoch=None):
    # plot cur goal next location
    fig = vis_origin_fig_global_Cloud_input(Cloud_env, cur_loc, goal_loc, next_loc, pixel_per_meter)
    # plot samples
    if sample_cnt is not None:
        fig, inside_rate = plot_samples_absolute_for_global(fig=fig, sp_cnt=sample_cnt,
                                                            sample_distribution=sample_distribution,
                                                            ppm=pixel_per_meter, r=1)
        save_fig_file += "_inside_" + str(int(inside_rate * 100))
    # save fig
    cv2.imwrite(save_fig_file + '.png', fig)

    # write img to tensorboard
    if tensorboard_writer is not None:
        # fig = np.ones(fig.shape) - fig/255
        fig = fig / 255
        fig_r = fig[:, :, 2:3]
        fig_g = fig[:, :, 1:2]
        fig_b = fig[:, :, 0:1]
        fig = np.concatenate((fig_r, fig_g, fig_b), axis=2)
        tensorboard_writer.add_image(fig_cat, fig, epoch, dataformats='HWC')

def plot_output_info_for_global_MPN_Reg(save_fig_file, pixel_per_meter, Cloud_env,
                                cur_loc, goal_loc, next_loc, pre_next_loc, tensorboard_writer=None, fig_cat=None, epoch=None):
    # plot cur goal next location
    fig = vis_origin_fig_global_Cloud_input(Cloud_env, cur_loc, goal_loc, next_loc, pixel_per_meter, pre_next_loc=pre_next_loc)
    # plot samples
    # save fig
    cv2.imwrite(save_fig_file + '.png', fig)

    # write img to tensorboard
    if tensorboard_writer is not None:
        # fig = np.ones(fig.shape) - fig/255
        fig = fig / 255
        fig_r = fig[:, :, 2:3]
        fig_g = fig[:, :, 1:2]
        fig_b = fig[:, :, 0:1]
        fig = np.concatenate((fig_r, fig_g, fig_b), axis=2)
        tensorboard_writer.add_image(fig_cat, fig, epoch, dataformats='HWC')

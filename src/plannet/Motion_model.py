import random

import numpy as np
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

def hook_test(arg):
    def hook(gard):
        print("the arg is:", arg)
        print("new hook is coming!")
        print(gard)
    return hook


class LMPN_MDN_0(nn.Module):
    """
    This model is designed for Mixture density network
    """

    def __init__(self, lidar_input_size, output_size, mixture_num):
        super(LMPN_MDN_0, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        self.lidar_input_size = lidar_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer1 = nn.Linear(lidar_input_size + 2, 1024)
        self.layer1_activate = nn.PReLU()
        self.layer1_dropout = nn.Dropout(0.0)

        self.layer2 = nn.Linear(1024, 1024)
        self.layer2_activate = nn.PReLU()
        self.layer2_dropout = nn.Dropout(0.0)

        self.layer3 = nn.Linear(1024, 1024)
        self.layer3_activate = nn.PReLU()
        self.layer3_dropout = nn.Dropout(0.0)

        self.layer4 = nn.Linear(1024, 1024)
        self.layer4_activate = nn.PReLU()
        self.layer4_dropout = nn.Dropout(0.0)

        self.layer5 = nn.Linear(1024, 1024)
        self.layer5_activate = nn.PReLU()
        self.layer5_dropout = nn.Dropout(0.0)

        self.layer6 = nn.Linear(1024, 32)
        self.layer6_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_lidar, x_pos):
        x = torch.cat([x_lidar, x_pos], dim=1)
        out = self.layer1(x)
        out = self.layer1_activate(out)
        out = self.layer1_dropout(out)

        out = self.layer2(out)
        out = self.layer2_activate(out)
        out = self.layer2_dropout(out)

        out = self.layer3(out)
        out = self.layer3_activate(out)
        out = self.layer3_dropout(out)

        out = self.layer4(out)
        out = self.layer4_activate(out)
        out = self.layer4_dropout(out)

        out = self.layer5(out)
        out = self.layer5_activate(out)
        out = self.layer5_dropout(out)

        out = self.layer6(out)
        out = self.layer6_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-15)/(1+self.mixture_num*1e-15)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-15

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss

class LMPN_MDN_1(nn.Module):
    """
    This model is designed for Mixture density network
    """

    def __init__(self, lidar_input_size, output_size, mixture_num):
        super(LMPN_MDN_1, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        self.lidar_input_size = lidar_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer1 = nn.Linear(lidar_input_size + 2, 1024)
        self.layer1_activate = nn.PReLU()
        self.layer1_dropout = nn.Dropout(0.0)

        self.layer2 = nn.Linear(1024, 1024)
        self.layer2_activate = nn.PReLU()
        self.layer2_dropout = nn.Dropout(0.0)

        self.layer3 = nn.Linear(1024, 1024)
        self.layer3_activate = nn.PReLU()
        self.layer3_dropout = nn.Dropout(0.0)

        self.layer4 = nn.Linear(1024, 1024)
        self.layer4_activate = nn.PReLU()
        self.layer4_dropout = nn.Dropout(0.0)

        self.layer5 = nn.Linear(1024, 32)
        self.layer5_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_lidar, x_pos):
        x = torch.cat([x_lidar, x_pos], dim=1)
        out = self.layer1(x)
        out = self.layer1_activate(out)
        out = self.layer1_dropout(out)

        out = self.layer2(out)
        out = self.layer2_activate(out)
        out = self.layer2_dropout(out)

        out = self.layer3(out)
        out = self.layer3_activate(out)
        out = self.layer3_dropout(out)

        out = self.layer4(out)
        out = self.layer4_activate(out)
        out = self.layer4_dropout(out)

        out = self.layer5(out)
        out = self.layer5_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-15)/(1+self.mixture_num*1e-15)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-15

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss

class LMPN_MDN_2(nn.Module):
    """
    This model is designed for Mixture density network
    """

    def __init__(self, lidar_input_size, output_size, mixture_num):
        super(LMPN_MDN_2, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        self.lidar_input_size = lidar_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer1 = nn.Linear(lidar_input_size + 2, 1024)
        self.layer1_activate = nn.PReLU()
        self.layer1_dropout = nn.Dropout(0.01)

        self.layer2 = nn.Linear(1024, 32)
        self.layer2_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_lidar, x_pos):
        x = torch.cat([x_lidar, x_pos], dim=1)
        out = self.layer1(x)
        out = self.layer1_activate(out)
        out = self.layer1_dropout(out)

        out = self.layer2(out)
        out = self.layer2_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-15)/(1+self.mixture_num*1e-15)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-15

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss

class LMPN_MDN_3(nn.Module):
    """
    This model is designed for Mixture density network
    """

    def __init__(self, lidar_input_size, output_size, mixture_num):
        super(LMPN_MDN_3, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        self.lidar_input_size = lidar_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer1 = nn.Linear(lidar_input_size + 2, 1024)
        self.layer1_activate = nn.PReLU()
        self.layer1_dropout = nn.Dropout(0.01)

        self.layer2 = nn.Linear(1024, 1024)
        self.layer2_activate = nn.PReLU()
        self.layer2_dropout = nn.Dropout(0.01)

        self.layer3 = nn.Linear(1024, 32)
        self.layer3_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_lidar, x_pos):
        x = torch.cat([x_lidar, x_pos], dim=1)
        out = self.layer1(x)
        out = self.layer1_activate(out)
        out = self.layer1_dropout(out)

        out = self.layer2(out)
        out = self.layer2_activate(out)
        out = self.layer2_dropout(out)

        out = self.layer3(out)
        out = self.layer3_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-15)/(1+self.mixture_num*1e-15)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-15

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss

class LMPN_MDN_4(nn.Module):
    """
    This model is designed for Mixture density network
    """

    def __init__(self, lidar_input_size, output_size, mixture_num):
        super(LMPN_MDN_4, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        self.lidar_input_size = lidar_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer1 = nn.Linear(lidar_input_size + 2, 1024)
        self.layer1_activate = nn.PReLU()
        self.layer1_dropout = nn.Dropout(0.01)

        self.layer2 = nn.Linear(1024, 1024)
        self.layer2_activate = nn.PReLU()
        self.layer2_dropout = nn.Dropout(0.01)

        self.layer3 = nn.Linear(1024, 1024)
        self.layer3_activate = nn.PReLU()
        self.layer3_dropout = nn.Dropout(0.01)

        self.layer4 = nn.Linear(1024, 32)
        self.layer4_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_lidar, x_pos):
        x = torch.cat([x_lidar, x_pos], dim=1)
        out = self.layer1(x)
        out = self.layer1_activate(out)
        out = self.layer1_dropout(out)

        out = self.layer2(out)
        out = self.layer2_activate(out)
        out = self.layer2_dropout(out)

        out = self.layer3(out)
        out = self.layer3_activate(out)
        out = self.layer3_dropout(out)

        out = self.layer4(out)
        out = self.layer4_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-15)/(1+self.mixture_num*1e-15)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-15

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss


class GMPN_REC_MDN_1(nn.Module):
    """
    This model is designed for Mixture density network with Global info input
    """

    def __init__(self, env_input_size, output_size, mixture_num):
        super(GMPN_REC_MDN_1, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        self.env_input_size = env_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer1 = nn.Linear(env_input_size + 4, 1024)
        self.layer1_activate = nn.PReLU()
        self.layer1_dropout = nn.Dropout(0.0)

        self.layer2 = nn.Linear(1024, 1024)
        self.layer2_activate = nn.PReLU()
        self.layer2_dropout = nn.Dropout(0.0)

        self.layer3 = nn.Linear(1024, 1024)
        self.layer3_activate = nn.PReLU()
        self.layer3_dropout = nn.Dropout(0.0)

        self.layer4 = nn.Linear(1024, 1024)
        self.layer4_activate = nn.PReLU()
        self.layer4_dropout = nn.Dropout(0.0)

        self.layer5 = nn.Linear(1024, 32)
        self.layer5_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_env, x_cur_pos, x_goal_pos):
        x = torch.cat([x_env, x_cur_pos, x_goal_pos], dim=1)
        out = self.layer1(x)
        out = self.layer1_activate(out)
        out = self.layer1_dropout(out)

        out = self.layer2(out)
        out = self.layer2_activate(out)
        out = self.layer2_dropout(out)

        out = self.layer3(out)
        out = self.layer3_activate(out)
        out = self.layer3_dropout(out)

        out = self.layer4(out)
        out = self.layer4_activate(out)
        out = self.layer4_dropout(out)

        out = self.layer5(out)
        out = self.layer5_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-15)/(1+self.mixture_num*1e-15)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-15

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss


class GMPN_EDGE_CLOUD_MDN_1(nn.Module):
    """
    This model is designed for Mixture density network with Global info input
    """

    def __init__(self, env_input_size, output_size, mixture_num):
        super(GMPN_EDGE_CLOUD_MDN_1, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        self.env_input_size = env_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer1 = nn.Linear(env_input_size, 1024)
        self.layer1_activate = nn.PReLU()
        self.layer1_dropout = nn.Dropout(0.0)

        self.layer2 = nn.Linear(1024, 256)
        self.layer2_activate = nn.PReLU()
        self.layer2_dropout = nn.Dropout(0.0)

        self.layer3 = nn.Linear(256, 128)
        self.layer3_activate = nn.PReLU()
        self.layer3_dropout = nn.Dropout(0.0)

        self.layer4 = nn.Linear(128, 32)
        self.layer4_activate = nn.PReLU()
        self.layer4_dropout = nn.Dropout(0.0)

        self.layer5 = nn.Linear(36, 1280)
        self.layer5_activate = nn.PReLU()

        self.layer6 = nn.Linear(1280, 1024)
        self.layer6_activate = nn.PReLU()

        self.layer7 = nn.Linear(1024, 896)
        self.layer7_activate = nn.PReLU()

        self.layer8 = nn.Linear(896, 768)
        self.layer8_activate = nn.PReLU()

        self.layer9 = nn.Linear(768, 512)
        self.layer9_activate = nn.PReLU()

        self.layer10 = nn.Linear(512, 32)
        self.layer10_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_env, x_cur_pos, x_goal_pos):

        x = x_env
        out = self.layer1(x)
        out = self.layer1_activate(out)
        # out = self.layer1_dropout(out)

        out = self.layer2(out)
        out = self.layer2_activate(out)
        # out = self.layer2_dropout(out)

        out = self.layer3(out)
        out = self.layer3_activate(out)
        # out = self.layer3_dropout(out)

        out = self.layer4(out)
        out = self.layer4_activate(out)

        out = torch.cat([out, x_cur_pos, x_goal_pos], dim=1)
        # out = self.layer4_dropout(out)

        out = self.layer5(out)
        out = self.layer5_activate(out)

        out = self.layer6(out)
        out = self.layer6_activate(out)

        out = self.layer7(out)
        out = self.layer7_activate(out)

        out = self.layer8(out)
        out = self.layer8_activate(out)

        out = self.layer9(out)
        out = self.layer9_activate(out)

        out = self.layer10(out)
        out = self.layer10_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-10)/(1+self.mixture_num*1e-10)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-10

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss


class GMPN_S2D_CLOUD_MDN_Pnet(nn.Module):
    """
    This model is designed for Mixture density network with Global info input
    """

    def __init__(self, input_size, output_size, mixture_num):
        super(GMPN_S2D_CLOUD_MDN_Pnet, self).__init__()

        self.grads = {}
        self.mixture_num = mixture_num
        # self.env_input_size = env_input_size
        self.output_size = output_size
        self.mixture_num = mixture_num

        self.layer5 = nn.Linear(input_size, 1024)
        self.layer5_activate = nn.PReLU()

        self.layer6 = nn.Linear(1024, 1024)
        self.layer6_activate = nn.PReLU()

        self.layer7 = nn.Linear(1024, 1024)
        self.layer7_activate = nn.PReLU()

        self.layer8 = nn.Linear(1024, 512)
        self.layer8_activate = nn.PReLU()

        self.layer9 = nn.Linear(512, 256)
        self.layer9_activate = nn.PReLU()

        self.layer10 = nn.Linear(256, 32)
        self.layer10_activate = nn.PReLU()

        self.layer_alpha = nn.Linear(32, mixture_num)
        self.layer_alpha_softmax = nn.Softmax(dim=1)

        self.layer_sigma = nn.Linear(32, mixture_num)
        self.layer_sigma_activate = nn.ELU(alpha=1.0)


        self.layer_mean = nn.Linear(32, mixture_num * output_size)

    def forward(self, x_env, x_cur_pos, x_goal_pos):

        out = torch.cat([x_env, x_cur_pos, x_goal_pos], dim=1)
        # out = self.layer4_dropout(out)

        out = self.layer5(out)
        out = self.layer5_activate(out)

        out = self.layer6(out)
        out = self.layer6_activate(out)

        out = self.layer7(out)
        out = self.layer7_activate(out)

        out = self.layer8(out)
        out = self.layer8_activate(out)

        out = self.layer9(out)
        out = self.layer9_activate(out)

        out = self.layer10(out)
        out = self.layer10_activate(out)

        alpha = self.layer_alpha(out)
        alpha_softmax = (self.layer_alpha_softmax(alpha) + 1e-10)/(1+self.mixture_num*1e-10)
        sigma = self.layer_sigma(out)

        sigma = self.layer_sigma_activate(sigma) + 1 + 1e-10

        mean = self.layer_mean(out)
        mean = mean.view(-1, self.mixture_num, self.output_size)
        return alpha_softmax, sigma, mean



    def compute_Gaussian_probability(self, alpha, sigma, mean, output_size, target):
        """
        Compute Gaussian distribution probability for a batch of data!
        input:
            alpha: prior probability of each gaussian, size: (batch size x mixture_num)
            sigma: variance, size: (batch size x mixture_num)
            mean: mean of gaussian, size: (batch size x mixture_num x output_size)
            target: label data, size: (batch size x output_size)
        output:
            prob_each_sample: each sample probability of the target, size: (batch size)
        """
        const = 1 / math.pow(2 * math.pi, (0.5 * output_size))
        target = torch.unsqueeze(target, dim=1).expand_as(
            mean)  # target size is shaped as (batch size x mixture_num x output_size)
        prob_each_gaussian = alpha * const * torch.pow(sigma, -output_size) * \
                             torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                 -2))  # size (batch size x mixture_num)
        prob_each_gaussian_no_prior = const * torch.pow(sigma, -output_size) * \
                                      torch.exp(-0.5 * torch.sum((target - mean) ** 2, dim=2) * torch.pow(sigma,
                                                                                                          -2))  # size (batch size x mixture_num)
        log_prob = torch.log(alpha) + math.log(const) - output_size * torch.log(sigma) - 0.5 * torch.sum(
            (target - mean) ** 2, dim=2) * torch.pow(sigma, -2)
        prob_each_sample = torch.sum(prob_each_gaussian, dim=1)  # size (batch size)

        return prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob

    def MDN_loss(self, alpha, sigma, mean, output_size, target):
        """
        Compute negative log maximum likelihood as loss function!
        attention: we compute it by batch!
        """
        prob_each_sample, prob_each_gaussian, prob_each_gaussian_no_prior, log_prob = self.compute_Gaussian_probability(
            alpha, sigma, mean, output_size, target)

        log_sum_exp = -torch.logsumexp(log_prob, dim=1)

        NLL_loss = torch.mean(log_sum_exp)

        lambda_sigma = 0.000
        sigma_loss = torch.sum(torch.pow(sigma, -1))
        sigma_loss = lambda_sigma * sigma_loss

        loss = NLL_loss + sigma_loss

        return loss, NLL_loss, sigma_loss

class S2D_MDN_Pnet(nn.Module):
    def __init__(self, input_size, output_size):
        super(S2D_MDN_Pnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1280), nn.PReLU(), nn.Dropout(),
            nn.Linear(1280, 1024), nn.PReLU(), nn.Dropout(),
            nn.Linear(1024, 896), nn.PReLU(), nn.Dropout(),
            nn.Linear(896, 768), nn.PReLU(), nn.Dropout(),
            nn.Linear(768, 512), nn.PReLU(), nn.Dropout(),
            nn.Linear(512, 384), nn.PReLU(), nn.Dropout(),
            nn.Linear(384, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 256), nn.PReLU(), nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.PReLU(), nn.Dropout(),
            nn.Linear(64, 32), nn.PReLU(),
            nn.Linear(32, output_size))

    def forward(self, x_env, x_cur_pos, x_goal_pos):
        x = torch.cat([x_env, x_cur_pos, x_goal_pos], dim=1)
        out = self.fc(x)
        return out

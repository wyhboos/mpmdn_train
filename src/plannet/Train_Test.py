import time

from torch.utils.data import DataLoader
from Data_loader import GMPNDataset, GMPNDataset_S2D_RB, GMPNDataset_S2D_TL
from Motion_model import GMPN_REC_MDN_1, GMPN_EDGE_CLOUD_MDN_1, GMPN_S2D_CLOUD_MDN_Pnet, S2D_MDN_Pnet
from Data_visualization import plot_output_info_for_global, vis_loss, plot_output_info_for_global_MPN_Reg
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# For tensorboard
from torch.utils.tensorboard import SummaryWriter


def created_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def Train_loop_mdn(model, optimizer, train_dataset, batch_size, device):
    model.train()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_pbar = tqdm.tqdm(train_loader, position=0, leave=True)
    train_loss = 0
    for x_env, x_cur_pos, x_goal_pos, y, index in train_pbar:
        optimizer.zero_grad()
        x_env, x_cur_pos, x_goal_pos, y = x_env.to(device), x_cur_pos.to(device), x_goal_pos.to(device), y.to(
            device)
        alpha, sigma, mean = model(x_env, x_cur_pos, x_goal_pos)
        loss, NLL_loss, sigma_loss = model.MDN_loss(alpha, sigma, mean, output_size=2, target=y)
        train_loss += loss.data
        loss.backward()
        # total_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()
    return train_loss


def Train_loop_mpn(model, optimizer, train_dataset, batch_size, device, criterion):
    model.train()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_pbar = tqdm.tqdm(train_loader, position=0, leave=True)
    train_loss = 0
    for x_env, x_cur_pos, x_goal_pos, y, index in train_pbar:
        optimizer.zero_grad()
        x_env, x_cur_pos, x_goal_pos, y = x_env.to(device), x_cur_pos.to(device), x_goal_pos.to(device), y.to(
            device)
        pre_next = model(x_env, x_cur_pos, x_goal_pos)
        loss = criterion(pre_next, y)
        train_loss += loss.data
        loss.backward()
        # total_norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()
    return train_loss


def test_loop_mdn(model, test_dataset, batch_size, device):
    model.eval()
    if device == 'cuda':
        model.cuda()
    else:
        model.cuda()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loss = 0
    with torch.no_grad():
        for x_env, x_cur_pos, x_goal_pos, y, index in test_loader:
            x_env, x_cur_pos, x_goal_pos, y = x_env.to(device), x_cur_pos.to(device), x_goal_pos.to(device), y.to(
                device)
            alpha, sigma, mean = model(x_env, x_cur_pos, x_goal_pos)
            loss, NLL_loss, sigma_loss = model.MDN_loss(alpha, sigma, mean, output_size=2, target=y)
            test_loss += loss.data
    return test_loss


def test_loop_mpn(model, test_dataset, batch_size, device, criterion):
    model.eval()
    if device == 'cuda':
        model.cuda()
    else:
        model.cuda()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loss = 0
    with torch.no_grad():
        for x_env, x_cur_pos, x_goal_pos, y, index in test_loader:
            x_env, x_cur_pos, x_goal_pos, y = x_env.to(device), x_cur_pos.to(device), x_goal_pos.to(device), y.to(
                device)
            pre_next = model(x_env, x_cur_pos, x_goal_pos)
            loss = criterion(pre_next, y)
            test_loss += loss.data
    return test_loss


def vis_output_for_Cloud_input(obs_cloud, model, vis_dataset, save_fig_path, device, epoch=None,
                               loss_gate=None, epoch_gate=None, tensorboard_writer=None, fig_cat=None):
    model.eval()
    vis_batch_size = min(len(vis_dataset), 512)
    vis_dataloader = DataLoader(vis_dataset, batch_size=vis_batch_size, shuffle=False)
    save_large_loss_fig_path = save_fig_path + 'large_loss/'
    created_dir(save_large_loss_fig_path)
    with torch.no_grad():
        vis_fig_cnt = 0
        for x_env, x_cur_pos, x_goal_pos, y, index in vis_dataloader:
            x_env, x_cur_pos, x_goal_pos, y = x_env.to(device), x_cur_pos.to(device), x_goal_pos.to(device), y.to(
                device)
            alpha, sigma, mean = model(x_env, x_cur_pos, x_goal_pos)
            alpha_ = alpha.cpu().detach().numpy()
            sigma_ = sigma.cpu().detach().numpy()
            mean_ = mean.cpu().detach().numpy()
            x_env_ = x_env.cpu().detach().numpy()
            x_cur_pos_ = x_cur_pos.cpu().detach().numpy()
            x_goal_pos_ = x_goal_pos.cpu().detach().numpy()
            y_ = y.cpu().detach().numpy()
            index_ = index.detach().numpy()
            for i in range(vis_batch_size):
                vis_loss, vis_NLL_loss, vis_sigma_loss = model.MDN_loss(alpha[i:i + 1], sigma[i:i + 1],
                                                                        mean[i:i + 1], output_size=2, target=y[i:i + 1])
                vis_loss_ = vis_loss.cpu().detach().numpy()
                index_i = int(index_[i])
                env_i = obs_cloud[index_i]
                save_fig_path_i = save_fig_path + "fig_cnt" + str(vis_fig_cnt)
                if fig_cat is not None:
                    fig_cat_i = fig_cat + '_cnt' + str(vis_fig_cnt)
                if epoch is not None:
                    save_fig_path_i += "_epoch_" + str(epoch)
                save_fig_path_i += "_loss_" + str((int(vis_loss_ * 1000)) / 1000)
                # save large loss pic
                if loss_gate is not None and epoch > epoch_gate and vis_loss_ > loss_gate:
                    save_large_loss_fig_path_i = save_large_loss_fig_path + "_epoch_" + \
                                                 str(epoch) + "_loss_" + str((int(vis_loss_ * 1000)) / 1000)
                    plot_output_info_for_global(save_fig_file=save_large_loss_fig_path_i, pixel_per_meter=20,
                                                Cloud_env=env_i,
                                                cur_loc=x_cur_pos_[i], goal_loc=x_goal_pos_[i], next_loc=y_[i],
                                                sample_cnt=1000, sample_distribution=[alpha_[i], sigma_[i], mean_[i]])

                plot_output_info_for_global(save_fig_file=save_fig_path_i, pixel_per_meter=20, Cloud_env=env_i,
                                            cur_loc=x_cur_pos_[i], goal_loc=x_goal_pos_[i], next_loc=y_[i],
                                            sample_cnt=1000, sample_distribution=[alpha_[i], sigma_[i], mean_[i]],
                                            tensorboard_writer=tensorboard_writer, fig_cat=fig_cat_i, epoch=epoch)

                vis_fig_cnt += 1


def vis_output_for_Cloud_input_Reg(obs_cloud, model, vis_dataset, save_fig_path, device, criterion, epoch=None,
                                   loss_gate=None, epoch_gate=None, tensorboard_writer=None, fig_cat=None):
    model.eval()
    vis_batch_size = min(len(vis_dataset), 512)
    vis_dataloader = DataLoader(vis_dataset, batch_size=vis_batch_size, shuffle=False)
    save_large_loss_fig_path = save_fig_path + 'large_loss/'
    created_dir(save_large_loss_fig_path)
    with torch.no_grad():
        vis_fig_cnt = 0
        for x_env, x_cur_pos, x_goal_pos, y, index in vis_dataloader:
            x_env, x_cur_pos, x_goal_pos, y = x_env.to(device), x_cur_pos.to(device), x_goal_pos.to(device), y.to(
                device)
            pre_next = model(x_env, x_cur_pos, x_goal_pos)
            pre_next_ = pre_next.cpu().detach().numpy()
            x_env_ = x_env.cpu().detach().numpy()
            x_cur_pos_ = x_cur_pos.cpu().detach().numpy()
            x_goal_pos_ = x_goal_pos.cpu().detach().numpy()
            y_ = y.cpu().detach().numpy()
            index_ = index.detach().numpy()
            for i in range(vis_batch_size):
                vis_loss = criterion(pre_next[i:i + 1], y[i:i + 1])
                vis_loss_ = vis_loss.cpu().detach().numpy()
                index_i = int(index_[i])
                env_i = obs_cloud[index_i]
                save_fig_path_i = save_fig_path + "fig_cnt" + str(vis_fig_cnt)
                if fig_cat is not None:
                    fig_cat_i = fig_cat + '_cnt' + str(vis_fig_cnt)
                if epoch is not None:
                    save_fig_path_i += "_epoch_" + str(epoch)
                save_fig_path_i += "_loss_" + str((int(vis_loss_ * 1000)) / 1000)
                # save large loss pic
                if loss_gate is not None and epoch > epoch_gate and vis_loss_ > loss_gate:
                    save_large_loss_fig_path_i = save_large_loss_fig_path + "_epoch_" + \
                                                 str(epoch) + "_loss_" + str((int(vis_loss_ * 1000)) / 1000)

                    plot_output_info_for_global_MPN_Reg(save_fig_file=save_large_loss_fig_path_i, pixel_per_meter=20,
                                                        Cloud_env=env_i,
                                                        cur_loc=x_cur_pos_[i], goal_loc=x_goal_pos_[i], next_loc=y_[i],
                                                        pre_next_loc=pre_next_[i],
                                                        tensorboard_writer=None, fig_cat=None, epoch=epoch)

                plot_output_info_for_global_MPN_Reg(save_fig_file=save_fig_path_i, pixel_per_meter=20,
                                                    Cloud_env=env_i,
                                                    cur_loc=x_cur_pos_[i], goal_loc=x_goal_pos_[i], next_loc=y_[i],
                                                    pre_next_loc=pre_next_[i],
                                                    tensorboard_writer=tensorboard_writer, fig_cat=fig_cat_i,
                                                    epoch=epoch)

                vis_fig_cnt += 1


def Train_Eval_global_Cloud_input_main():
    # parm set
    device = 'cuda'

    mixture_num = 20
    lr = 3 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 5000

    # train_data_load_file = "../../../output/data/paths/paths_raw_data/train_data/" \
    #                        "train_data_shrunk_0.125redundant_remove_global_100k_train.npy"
    # test_data_load_file = "../../../output/data/paths/paths_raw_data/train_data/" \
    #                        "train_data_shrunk_0.125redundant_remove_global_100k_test.npy"

    # train_data_load_file = "../../../output/data/S2D/MPN_S2D_lzrm_train.npy"
    # train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_lzrm_train_env_test.npy"
    # new_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_lzrm_new_env_test.npy"

    train_data_load_file = "../../../output/data/S2D/MPN_S2D_train_578k.npy"
    train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_train_env_test_82k.npy"
    new_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_new_env_test_74k.npy"

    model_name = "GMPN_S2D_CLOUD_MDN_4"
    model_dir = '../../../output/model/' + model_name + '/'
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/GMPN_S2D_CLOUD_MDN_6/checkpoint_save/checkpoint_epoch_340.pt'

    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)

    train_env_test_vis_fig_save_dir = model_dir + "train_env_test_vis_fig/"
    created_dir(train_env_test_vis_fig_save_dir)

    new_env_test_vis_fig_save_dir = model_dir + "new_env_test_vis_fig/"
    created_dir(new_env_test_vis_fig_save_dir)

    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)

    obs_cloud = np.load("../../../output/data/S2D/obs_cloud_30000.npy")
    obs_cloud = obs_cloud[:100, :]

    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = '../../../output/visualizations/tensorboard/' + model_name + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    train_batch_size = 512
    train_env_test_batch_size = 8192
    new_env_test_batch_size = 8192
    env_info_length = 28
    train_data_vis_cnt = 30
    train_env_test_data_vis_cnt = 30
    new_env_test_data_vis_cnt = 30

    checkpoint_save_interval = 10
    vis_fig_save_interval = 10

    # load dataset
    print("Start load dataset!")
    train_dataset = GMPNDataset(data_file=train_data_load_file, env_info_length=env_info_length, data_len=None)
    train_env_test_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
                                         data_len=None)
    new_env_test_dataset = GMPNDataset(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                       data_len=None)

    train_vis_dataset = GMPNDataset(data_file=train_data_load_file, env_info_length=env_info_length,
                                    data_len=train_data_vis_cnt)
    train_env_test_vis_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
                                             data_len=train_env_test_data_vis_cnt)
    new_env_test_vis_dataset = GMPNDataset(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                           data_len=new_env_test_data_vis_cnt)
    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model = GMPN_S2D_CLOUD_MDN_Pnet(env_input_size=env_info_length, output_size=2, mixture_num=mixture_num)
    model = model.float()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    train_env_test_loss_all = []
    new_env_test_loss_all = []
    train_data_size = len(train_dataset)
    train_env_test_data_size = len(train_env_test_dataset)
    new_env_test_data_size = len(new_env_test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        model.cuda()
        train_loss_i = Train_loop_global_Cloud_input(model=model, optimizer=optimizer, train_dataset=train_dataset,
                                                     batch_size=train_batch_size, device=device)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)
        # Eval loop
        train_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=train_env_test_dataset,
                                                             batch_size=train_env_test_batch_size, device=device)
        train_env_test_loss_i_mean = float(
            train_env_test_loss_i.data) * train_env_test_batch_size / train_env_test_data_size
        print('train_env_test loss is,', train_env_test_loss_i_mean)
        train_env_test_loss_all.append(train_env_test_loss_i_mean)

        new_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=new_env_test_dataset,
                                                           batch_size=new_env_test_batch_size, device=device)
        new_env_test_loss_i_mean = float(new_env_test_loss_i.data) * new_env_test_batch_size / new_env_test_data_size
        print('new_env_test loss is,', new_env_test_loss_i_mean)
        new_env_test_loss_all.append(new_env_test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            'train_env_test': train_env_test_loss_i_mean,
            'new_env_test': new_env_test_loss_i_mean
        }, epoch)

        # Save checkpoint and loss
        if epoch % checkpoint_save_interval == 0:
            # save loss file
            # loss_save_file_train_i = loss_save_dir + "train_loss_epoch_" + str(epoch)
            # loss_save_file_test_i = loss_save_dir + "train_loss_epoch_" + str(epoch)
            # np.save(loss_save_file_train_i, np.array(train_loss_all))
            # np.save(loss_save_file_test_i, np.array(test_loss_all))
            # save checkpoint
            checkpoint_save_file_i = checkpoint_save_dir + "checkpoint_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_i_mean
            }, checkpoint_save_file_i)
            # vis loss
            # vis_loss_dir_i = vis_loss_dir + "epoch_" + str(epoch)
            # vis_loss(save_fig_dir=vis_loss_dir_i, train_loss=train_loss_all,
            #          test_loss=test_loss_all, local_range=vis_fig_save_interval - 1)
        # vis dataset
        if epoch % vis_fig_save_interval == 0:
            # vis train dataset
            vis_output_for_Cloud_input(obs_cloud=obs_cloud, model=model, vis_dataset=train_vis_dataset,
                                       save_fig_path=train_vis_fig_save_dir, device=device,
                                       epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                       fig_cat='img/train')
            # vis test dataset
            vis_output_for_Cloud_input(obs_cloud=obs_cloud, model=model, vis_dataset=train_env_test_vis_dataset,
                                       save_fig_path=train_env_test_vis_fig_save_dir, device=device,
                                       epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                       fig_cat='img/train_env_test')

            vis_output_for_Cloud_input(obs_cloud=obs_cloud, model=model, vis_dataset=new_env_test_vis_dataset,
                                       save_fig_path=new_env_test_vis_fig_save_dir, device=device,
                                       epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                       fig_cat='img/new_env_test')


# 三种地图比较
def Train_Eval_global_Cloud_input_MPN_Reg_main():
    # parm set
    device = 'cuda'

    mixture_num = 20
    lr = 3 * 1e-4
    # lr = 10 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 2000

    # train_data_load_file = "../../../output/data/paths/paths_raw_data/train_data/" \
    #                        "train_data_shrunk_0.125redundant_remove_global_100k_train.npy"
    # test_data_load_file = "../../../output/data/paths/paths_raw_data/train_data/" \
    #                        "train_data_shrunk_0.125redundant_remove_global_100k_test.npy"

    # train_data_load_file = "../../../output/data/S2D/MPN_S2D_train.npy"
    # test_data_load_file = "../../../output/data/S2D/MPN_S2D_test.npy"
    # train_data_load_file = "../../../output/data/S2D/MPN_S2D_lzrm_train.npy"
    # train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_lzrm_train_env_test.npy"
    # new_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_lzrm_new_env_test.npy"

    train_data_load_file = "../../../output/data/S2D/MPN_S2D_train_578k.npy"
    train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_train_env_test_82k.npy"
    new_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_new_env_test_74k.npy"

    # model_name = "GMPN_S2D_CLOUD_MPN_Reg_lzrm_3"
    model_name = "GMPN_S2D_CLOUD_MPN_Reg_13"
    model_dir = '../../../output/model/' + model_name + '/'
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/GMPN_S2D_CLOUD_MPN_Reg_lzrm_1/checkpoint_save/checkpoint_epoch_500.pt'

    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)

    train_env_test_vis_fig_save_dir = model_dir + "train_env_test_vis_fig/"
    created_dir(train_env_test_vis_fig_save_dir)

    new_env_test_vis_fig_save_dir = model_dir + "new_env_test_vis_fig/"
    created_dir(new_env_test_vis_fig_save_dir)

    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)

    obs_cloud = np.load("../../../output/data/S2D/obs_cloud_30000.npy")
    obs_cloud = obs_cloud[:100, :]

    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = '../../../output/visualizations/tensorboard/' + model_name + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    train_batch_size = 512
    train_env_test_batch_size = 8192
    new_env_test_batch_size = 8192
    env_info_length = 28
    train_data_vis_cnt = 30
    train_env_test_data_vis_cnt = 30
    new_env_test_data_vis_cnt = 30

    checkpoint_save_interval = 10
    vis_fig_save_interval = 10

    # load dataset
    print("Start load dataset!")
    train_dataset = GMPNDataset(data_file=train_data_load_file, env_info_length=env_info_length, data_len=None)
    train_env_test_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
                                         data_len=None)
    new_env_test_dataset = GMPNDataset(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                       data_len=None)

    train_vis_dataset = GMPNDataset(data_file=train_data_load_file, env_info_length=env_info_length,
                                    data_len=train_data_vis_cnt)
    train_env_test_vis_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
                                             data_len=train_env_test_data_vis_cnt)
    new_env_test_vis_dataset = GMPNDataset(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                           data_len=new_env_test_data_vis_cnt)
    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model = S2D_MDN_Pnet(input_size=32, output_size=2)
    model = model.float()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    train_env_test_loss_all = []
    new_env_test_loss_all = []
    train_data_size = len(train_dataset)
    train_env_test_data_size = len(train_env_test_dataset)
    new_env_test_data_size = len(new_env_test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        model.cuda()
        train_loss_i = Train_loop_global_Cloud_input(model=model, optimizer=optimizer, train_dataset=train_dataset,
                                                     batch_size=train_batch_size, device=device, criterion=criterion)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)

        # Eval loop
        train_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=train_env_test_dataset,
                                                             batch_size=train_env_test_batch_size, device=device,
                                                             criterion=criterion)
        train_env_test_loss_i_mean = float(
            train_env_test_loss_i.data) * train_env_test_batch_size / train_env_test_data_size
        print('train_env_test loss is,', train_env_test_loss_i_mean)
        train_env_test_loss_all.append(train_env_test_loss_i_mean)

        new_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=new_env_test_dataset,
                                                           batch_size=new_env_test_batch_size, device=device,
                                                           criterion=criterion)
        new_env_test_loss_i_mean = float(new_env_test_loss_i.data) * new_env_test_batch_size / new_env_test_data_size
        print('new_env_test loss is,', new_env_test_loss_i_mean)
        new_env_test_loss_all.append(new_env_test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            'train_env_test': train_env_test_loss_i_mean,
            'new_env_test': new_env_test_loss_i_mean
        }, epoch)

        # Save checkpoint and loss
        if epoch % checkpoint_save_interval == 0:
            # save loss file
            # loss_save_file_train_i = loss_save_dir + "train_loss_epoch_" + str(epoch)
            # loss_save_file_test_i = loss_save_dir + "train_loss_epoch_" + str(epoch)
            # np.save(loss_save_file_train_i, np.array(train_loss_all))
            # np.save(loss_save_file_test_i, np.array(test_loss_all))
            # save checkpoint
            checkpoint_save_file_i = checkpoint_save_dir + "checkpoint_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_i_mean
            }, checkpoint_save_file_i)
            # vis loss
            # vis_loss_dir_i = vis_loss_dir + "epoch_" + str(epoch)
            # vis_loss(save_fig_dir=vis_loss_dir_i, train_loss=train_loss_all,
            #          test_loss=test_loss_all, local_range=vis_fig_save_interval - 1)
        # vis dataset
        if epoch % vis_fig_save_interval == 0:
            # vis train dataset
            vis_output_for_Cloud_input_Reg(obs_cloud=obs_cloud, model=model, vis_dataset=train_vis_dataset,
                                           save_fig_path=train_vis_fig_save_dir, device=device,
                                           epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                           fig_cat='img/train', criterion=criterion)
            # vis test dataset
            vis_output_for_Cloud_input_Reg(obs_cloud=obs_cloud, model=model, vis_dataset=train_env_test_vis_dataset,
                                           save_fig_path=train_env_test_vis_fig_save_dir, device=device,
                                           epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                           fig_cat='img/train_env_test', criterion=criterion)

            vis_output_for_Cloud_input_Reg(obs_cloud=obs_cloud, model=model, vis_dataset=new_env_test_vis_dataset,
                                           save_fig_path=new_env_test_vis_fig_save_dir, device=device,
                                           epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                           fig_cat='img/new_env_test', criterion=criterion)


# 只有两种地图比较
def Train_Eval_global_Cloud_input_MPN_Reg_main_old():
    # parm set
    device = 'cuda'

    mixture_num = 20
    lr = 1 * 1e-4
    # lr = 10 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 2000

    # train_data_load_file = "../../../output/data/paths/paths_raw_data/train_data/" \
    #                        "train_data_shrunk_0.125redundant_remove_global_100k_train.npy"
    # test_data_load_file = "../../../output/data/paths/paths_raw_data/train_data/" \
    #                        "train_data_shrunk_0.125redundant_remove_global_100k_test.npy"

    # train_data_load_file = "../../../output/data/S2D/MPN_S2D_train.npy"
    # test_data_load_file = "../../../output/data/S2D/MPN_S2D_test.npy"
    train_data_load_file = "../../../output/data/S2D/MPN_S2D_train.npy"
    train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_test.npy"
    # new_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_lzrm_new_env_test.npy"

    model_name = "GMPN_S2D_CLOUD_MPN_Reg_12"
    model_dir = '../../../output/model/' + model_name + '/'
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/GMPN_S2D_CLOUD_MPN_Reg_lzrm_1/checkpoint_save/checkpoint_epoch_500.pt'

    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)

    train_env_test_vis_fig_save_dir = model_dir + "train_env_test_vis_fig/"
    created_dir(train_env_test_vis_fig_save_dir)

    new_env_test_vis_fig_save_dir = model_dir + "new_env_test_vis_fig/"
    created_dir(new_env_test_vis_fig_save_dir)

    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)

    obs_cloud = np.load("../../../output/data/S2D/obs_cloud_30000.npy")
    obs_cloud = obs_cloud[:100, :]

    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = '../../../output/visualizations/tensorboard/' + model_name + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    train_batch_size = 128
    train_env_test_batch_size = 8192
    new_env_test_batch_size = 8192
    env_info_length = 28
    train_data_vis_cnt = 30
    train_env_test_data_vis_cnt = 30
    new_env_test_data_vis_cnt = 30

    checkpoint_save_interval = 10
    vis_fig_save_interval = 10

    # load dataset
    print("Start load dataset!")
    train_dataset = GMPNDataset(data_file=train_data_load_file, env_info_length=env_info_length, data_len=None)
    train_env_test_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
                                         data_len=None)
    # new_env_test_dataset = GMPNDataset(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
    #                                         data_len=None)

    train_vis_dataset = GMPNDataset(data_file=train_data_load_file, env_info_length=env_info_length,
                                    data_len=train_data_vis_cnt)
    train_env_test_vis_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
                                             data_len=train_env_test_data_vis_cnt)
    # new_env_test_vis_dataset = GMPNDataset(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
    #                                        data_len=new_env_test_data_vis_cnt)
    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model = S2D_MDN_Pnet(input_size=32, output_size=2)
    model = model.float()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    train_env_test_loss_all = []
    new_env_test_loss_all = []
    train_data_size = len(train_dataset)
    train_env_test_data_size = len(train_env_test_dataset)
    # new_env_test_data_size = len(new_env_test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        model.cuda()
        train_loss_i = Train_loop_global_Cloud_input(model=model, optimizer=optimizer, train_dataset=train_dataset,
                                                     batch_size=train_batch_size, device=device, criterion=criterion)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)

        # Eval loop
        train_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=train_env_test_dataset,
                                                             batch_size=train_env_test_batch_size, device=device,
                                                             criterion=criterion)
        train_env_test_loss_i_mean = float(
            train_env_test_loss_i.data) * train_env_test_batch_size / train_env_test_data_size
        print('train_env_test loss is,', train_env_test_loss_i_mean)
        train_env_test_loss_all.append(train_env_test_loss_i_mean)

        # new_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=new_env_test_dataset,
        #                                                    batch_size=new_env_test_batch_size, device=device,
        #                                                    criterion=criterion)
        # new_env_test_loss_i_mean = float(new_env_test_loss_i.data) * new_env_test_batch_size / new_env_test_data_size
        # print('new_env_test loss is,', new_env_test_loss_i_mean)
        # new_env_test_loss_all.append(new_env_test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            'train_env_test': train_env_test_loss_i_mean,
            # 'new_env_test': new_env_test_loss_i_mean
        }, epoch)

        # Save checkpoint and loss
        if epoch % checkpoint_save_interval == 0:
            # save loss file
            # loss_save_file_train_i = loss_save_dir + "train_loss_epoch_" + str(epoch)
            # loss_save_file_test_i = loss_save_dir + "train_loss_epoch_" + str(epoch)
            # np.save(loss_save_file_train_i, np.array(train_loss_all))
            # np.save(loss_save_file_test_i, np.array(test_loss_all))
            # save checkpoint
            checkpoint_save_file_i = checkpoint_save_dir + "checkpoint_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_i_mean
            }, checkpoint_save_file_i)
            # vis loss
            # vis_loss_dir_i = vis_loss_dir + "epoch_" + str(epoch)
            # vis_loss(save_fig_dir=vis_loss_dir_i, train_loss=train_loss_all,
            #          test_loss=test_loss_all, local_range=vis_fig_save_interval - 1)
        # vis dataset
        if epoch % vis_fig_save_interval == 0:
            # vis train dataset
            vis_output_for_Cloud_input_Reg(obs_cloud=obs_cloud, model=model, vis_dataset=train_vis_dataset,
                                           save_fig_path=train_vis_fig_save_dir, device=device,
                                           epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                           fig_cat='img/train', criterion=criterion)
            # vis test dataset
            vis_output_for_Cloud_input_Reg(obs_cloud=obs_cloud, model=model, vis_dataset=train_env_test_vis_dataset,
                                           save_fig_path=train_env_test_vis_fig_save_dir, device=device,
                                           epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
                                           fig_cat='img/train_env_test', criterion=criterion)

            # vis_output_for_Cloud_input_Reg(obs_cloud=obs_cloud, model=model, vis_dataset=new_env_test_vis_dataset,
            #                                save_fig_path=new_env_test_vis_fig_save_dir, device=device,
            #                                epoch=epoch, loss_gate=10, epoch_gate=2000, tensorboard_writer=writer,
            #                                fig_cat='img/new_env_test', criterion=criterion)


def Train_Eval_Cloud_input_S2D_RB_MDN_main():
    # parm set
    device = 'cuda'

    mixture_num = 20
    lr = 3 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 5000

    train_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Rigidbody_train.npy"
    # train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_train_env_test_82k.npy"
    new_env_test_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Rigidbody_train.npy"

    model_name = "MDN_S2D_RB_1"
    model_dir = "../../data/model/" + model_name + '/'
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/GMPN_S2D_CLOUD_MDN_6/checkpoint_save/checkpoint_epoch_340.pt'

    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)

    train_env_test_vis_fig_save_dir = model_dir + "train_env_test_vis_fig/"
    created_dir(train_env_test_vis_fig_save_dir)

    new_env_test_vis_fig_save_dir = model_dir + "new_env_test_vis_fig/"
    created_dir(new_env_test_vis_fig_save_dir)

    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)

    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = model_dir + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    train_batch_size = 512
    train_env_test_batch_size = 8192
    new_env_test_batch_size = 8192
    env_info_length = 28
    train_data_vis_cnt = 30
    train_env_test_data_vis_cnt = 30
    new_env_test_data_vis_cnt = 30

    checkpoint_save_interval = 20
    vis_fig_save_interval = 10

    # load dataset
    print("Start load dataset!")
    train_dataset = GMPNDataset_S2D_RB(data_file=train_data_load_file, env_info_length=env_info_length, data_len=None)
    # train_env_test_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
    #                                      data_len=None)
    new_env_test_dataset = GMPNDataset_S2D_RB(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                              data_len=None)

    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model = GMPN_S2D_CLOUD_MDN_Pnet(input_size=34, output_size=3, mixture_num=mixture_num)
    model = model.float()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    train_env_test_loss_all = []
    new_env_test_loss_all = []
    train_data_size = len(train_dataset)
    # train_env_test_data_size = len(train_env_test_dataset)
    new_env_test_data_size = len(new_env_test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        model.cuda()
        train_loss_i = Train_loop_mdn(model=model, optimizer=optimizer, train_dataset=train_dataset,
                                      batch_size=train_batch_size, device=device)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)
        # Eval loop
        # train_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=train_env_test_dataset,
        #                                                    batch_size=train_env_test_batch_size, device=device)
        # train_env_test_loss_i_mean = float(train_env_test_loss_i.data) * train_env_test_batch_size / train_env_test_data_size
        # print('train_env_test loss is,', train_env_test_loss_i_mean)
        # train_env_test_loss_all.append(train_env_test_loss_i_mean)

        new_env_test_loss_i = test_loop_mdn(model=model, test_dataset=new_env_test_dataset,
                                            batch_size=new_env_test_batch_size, device=device)
        new_env_test_loss_i_mean = float(new_env_test_loss_i.data) * new_env_test_batch_size / new_env_test_data_size
        print('new_env_test loss is,', new_env_test_loss_i_mean)
        new_env_test_loss_all.append(new_env_test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            # 'train_env_test':train_env_test_loss_i_mean,
            'new_env_test': new_env_test_loss_i_mean
        }, epoch)

        # Save checkpoint and loss
        if epoch % checkpoint_save_interval == 0:
            # save checkpoint
            checkpoint_save_file_i = checkpoint_save_dir + "checkpoint_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_i_mean
            }, checkpoint_save_file_i)


def Train_Eval_Cloud_input_S2D_TL_MDN_main():
    # parm set
    device = 'cuda'

    mixture_num = 20
    lr = 3 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 5000

    train_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Two_Link_train.npy"
    # train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_train_env_test_82k.npy"
    new_env_test_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Two_Link_test.npy"
    model_name = "MDN_S2D_TL_1"
    model_dir = "../../data/model/" + model_name + '/'
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/GMPN_S2D_CLOUD_MDN_6/checkpoint_save/checkpoint_epoch_340.pt'

    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)

    train_env_test_vis_fig_save_dir = model_dir + "train_env_test_vis_fig/"
    created_dir(train_env_test_vis_fig_save_dir)

    new_env_test_vis_fig_save_dir = model_dir + "new_env_test_vis_fig/"
    created_dir(new_env_test_vis_fig_save_dir)

    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)


    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = model_dir + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    train_batch_size = 512
    train_env_test_batch_size = 8192
    new_env_test_batch_size = 8192
    env_info_length = 28
    train_data_vis_cnt = 30
    train_env_test_data_vis_cnt = 30
    new_env_test_data_vis_cnt = 30

    checkpoint_save_interval = 20
    vis_fig_save_interval = 10

    # load dataset
    print("Start load dataset!")
    train_dataset = GMPNDataset_S2D_TL(data_file=train_data_load_file, env_info_length=env_info_length, data_len=None)
    # train_env_test_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
    #                                      data_len=None)
    new_env_test_dataset = GMPNDataset_S2D_TL(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                              data_len=None)

    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model = GMPN_S2D_CLOUD_MDN_Pnet(input_size=36, output_size=4, mixture_num=mixture_num)
    model = model.float()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    train_env_test_loss_all = []
    new_env_test_loss_all = []
    train_data_size = len(train_dataset)
    # train_env_test_data_size = len(train_env_test_dataset)
    new_env_test_data_size = len(new_env_test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        model.cuda()
        train_loss_i = Train_loop_mdn(model=model, optimizer=optimizer, train_dataset=train_dataset,
                                      batch_size=train_batch_size, device=device)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)
        # Eval loop
        # train_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=train_env_test_dataset,
        #                                                    batch_size=train_env_test_batch_size, device=device)
        # train_env_test_loss_i_mean = float(train_env_test_loss_i.data) * train_env_test_batch_size / train_env_test_data_size
        # print('train_env_test loss is,', train_env_test_loss_i_mean)
        # train_env_test_loss_all.append(train_env_test_loss_i_mean)

        new_env_test_loss_i = test_loop_mdn(model=model, test_dataset=new_env_test_dataset,
                                            batch_size=new_env_test_batch_size, device=device)
        new_env_test_loss_i_mean = float(new_env_test_loss_i.data) * new_env_test_batch_size / new_env_test_data_size
        print('new_env_test loss is,', new_env_test_loss_i_mean)
        new_env_test_loss_all.append(new_env_test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            # 'train_env_test':train_env_test_loss_i_mean,
            'new_env_test': new_env_test_loss_i_mean
        }, epoch)

        # Save checkpoint and loss
        if epoch % checkpoint_save_interval == 0:
            # save checkpoint
            checkpoint_save_file_i = checkpoint_save_dir + "checkpoint_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_i_mean
            }, checkpoint_save_file_i)


def Train_Eval_Cloud_input_S2D_RB_MPN_main():
    # parm set
    device = 'cuda'
    lr = 3 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 5000

    train_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Rigidbody_train.npy"
    # train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_train_env_test_82k.npy"
    new_env_test_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Rigidbody_train.npy"

    model_name = "MPN_S2D_RB_1"
    model_dir = "../../data/model/" + model_name + '/'
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/GMPN_S2D_CLOUD_MDN_6/checkpoint_save/checkpoint_epoch_340.pt'

    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)

    train_env_test_vis_fig_save_dir = model_dir + "train_env_test_vis_fig/"
    created_dir(train_env_test_vis_fig_save_dir)

    new_env_test_vis_fig_save_dir = model_dir + "new_env_test_vis_fig/"
    created_dir(new_env_test_vis_fig_save_dir)

    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)

    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = model_dir + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    train_batch_size = 512
    train_env_test_batch_size = 8192
    new_env_test_batch_size = 8192
    env_info_length = 28
    train_data_vis_cnt = 30
    train_env_test_data_vis_cnt = 30
    new_env_test_data_vis_cnt = 30

    checkpoint_save_interval = 20
    vis_fig_save_interval = 10

    # load dataset
    print("Start load dataset!")
    train_dataset = GMPNDataset_S2D_RB(data_file=train_data_load_file, env_info_length=env_info_length, data_len=None)
    # train_env_test_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
    #                                      data_len=None)
    new_env_test_dataset = GMPNDataset_S2D_RB(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                              data_len=None)

    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model = S2D_MDN_Pnet(input_size=34, output_size=3)
    model = model.float()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss(reduction='mean')
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    train_env_test_loss_all = []
    new_env_test_loss_all = []
    train_data_size = len(train_dataset)
    # train_env_test_data_size = len(train_env_test_dataset)
    new_env_test_data_size = len(new_env_test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        model.cuda()
        train_loss_i = Train_loop_mpn(model=model, optimizer=optimizer, train_dataset=train_dataset,
                                      batch_size=train_batch_size, device=device, criterion=criterion)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)
        # Eval loop
        # train_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=train_env_test_dataset,
        #                                                    batch_size=train_env_test_batch_size, device=device)
        # train_env_test_loss_i_mean = float(train_env_test_loss_i.data) * train_env_test_batch_size / train_env_test_data_size
        # print('train_env_test loss is,', train_env_test_loss_i_mean)
        # train_env_test_loss_all.append(train_env_test_loss_i_mean)

        new_env_test_loss_i = test_loop_mpn(model=model, test_dataset=new_env_test_dataset,
                                            batch_size=new_env_test_batch_size, device=device,
                                            criterion=criterion)
        new_env_test_loss_i_mean = float(new_env_test_loss_i.data) * new_env_test_batch_size / new_env_test_data_size
        print('new_env_test loss is,', new_env_test_loss_i_mean)
        new_env_test_loss_all.append(new_env_test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            # 'train_env_test':train_env_test_loss_i_mean,
            'new_env_test': new_env_test_loss_i_mean
        }, epoch)

        # Save checkpoint and loss
        if epoch % checkpoint_save_interval == 0:
            checkpoint_save_file_i = checkpoint_save_dir + "checkpoint_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_i_mean
            }, checkpoint_save_file_i)


def Train_Eval_Cloud_input_S2D_TL_MPN_main():
    # parm set
    device = 'cuda'
    lr = 3 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 5000

    train_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Two_Link_train.npy"
    # train_env_test_data_load_file = "../../../output/data/S2D/MPN_S2D_train_env_test_82k.npy"
    new_env_test_data_load_file = "../../data/train/s2d/1000env_400pt/S2D_Two_Link_test.npy"

    model_name = "MPN_S2D_TL_1"
    model_dir = "../../data/model/" + model_name + '/'
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/GMPN_S2D_CLOUD_MDN_6/checkpoint_save/checkpoint_epoch_340.pt'

    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)

    train_env_test_vis_fig_save_dir = model_dir + "train_env_test_vis_fig/"
    created_dir(train_env_test_vis_fig_save_dir)

    new_env_test_vis_fig_save_dir = model_dir + "new_env_test_vis_fig/"
    created_dir(new_env_test_vis_fig_save_dir)

    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)

    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = model_dir + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    train_batch_size = 512
    train_env_test_batch_size = 8192
    new_env_test_batch_size = 8192
    env_info_length = 28
    train_data_vis_cnt = 30
    train_env_test_data_vis_cnt = 30
    new_env_test_data_vis_cnt = 30

    checkpoint_save_interval = 20
    vis_fig_save_interval = 10

    # load dataset
    print("Start load dataset!")
    train_dataset = GMPNDataset_S2D_TL(data_file=train_data_load_file, env_info_length=env_info_length, data_len=None)
    # train_env_test_dataset = GMPNDataset(data_file=train_env_test_data_load_file, env_info_length=env_info_length,
    #                                      data_len=None)
    new_env_test_dataset = GMPNDataset_S2D_TL(data_file=new_env_test_data_load_file, env_info_length=env_info_length,
                                              data_len=None)
    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model = S2D_MDN_Pnet(input_size=36, output_size=4)
    model = model.float()
    if device == 'cuda':
        model.cuda()
    else:
        model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss(reduction='mean')
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    train_env_test_loss_all = []
    new_env_test_loss_all = []
    train_data_size = len(train_dataset)
    # train_env_test_data_size = len(train_env_test_dataset)
    new_env_test_data_size = len(new_env_test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        model.cuda()
        train_loss_i = Train_loop_mpn(model=model, optimizer=optimizer, train_dataset=train_dataset,
                                      batch_size=train_batch_size, device=device, criterion=criterion)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)
        # Eval loop
        # train_env_test_loss_i = test_loop_global_Cloud_input(model=model, test_dataset=train_env_test_dataset,
        #                                                    batch_size=train_env_test_batch_size, device=device)
        # train_env_test_loss_i_mean = float(train_env_test_loss_i.data) * train_env_test_batch_size / train_env_test_data_size
        # print('train_env_test loss is,', train_env_test_loss_i_mean)
        # train_env_test_loss_all.append(train_env_test_loss_i_mean)

        new_env_test_loss_i = test_loop_mpn(model=model, test_dataset=new_env_test_dataset,
                                            batch_size=new_env_test_batch_size, device=device,
                                            criterion=criterion)
        new_env_test_loss_i_mean = float(new_env_test_loss_i.data) * new_env_test_batch_size / new_env_test_data_size
        print('new_env_test loss is,', new_env_test_loss_i_mean)
        new_env_test_loss_all.append(new_env_test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            # 'train_env_test':train_env_test_loss_i_mean,
            'new_env_test': new_env_test_loss_i_mean
        }, epoch)

        # Save checkpoint and loss
        if epoch % checkpoint_save_interval == 0:
            checkpoint_save_file_i = checkpoint_save_dir + "checkpoint_epoch_" + str(epoch) + ".pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss_i_mean
            }, checkpoint_save_file_i)


if __name__ == '__main__':
    # Train_Eval_global_Cloud_input_main()
    # Train_Eval_global_Cloud_input_MPN_Reg_main_old()
    # Train_Eval_global_Cloud_input_MPN_Reg_main()
    # Train_Eval_Cloud_input_S2D_RB_MDN_main()
    # Train_Eval_Cloud_input_S2D_TL_MDN_main()
    # Train_Eval_Cloud_input_S2D_RB_MPN_main()
    # Train_Eval_Cloud_input_S2D_TL_MPN_main()

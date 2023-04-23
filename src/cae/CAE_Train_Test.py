import random

from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
from Data_loader import CAE_cloud_Dataset
from CAE_model import Encoder, Decoder, Encoder_S2D, Decoder_S2D
from Data_visualization import vis_loss, plot_env_scatter
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

mse_loss = torch.nn.MSELoss()
lam = 1e-3


def loss_function(W, x, recons_x, h):
    mse = mse_loss(recons_x, x)
    """
	W is shape of N_hidden x N. So, we do not need to transpose it as opposed to http://wiseodd.github.io/techblog/2016/12/05/contractive-autoencoder/
	"""
    dh = h * (1 - h)  # N_batch x N_hidden
    contractive_loss = torch.sum(Variable(W) ** 2, dim=1).sum().mul_(lam)
    return mse + contractive_loss


def created_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def Train_loop_CAE(model_encoder, model_decoder, optimizer, criterion, train_dataset, batch_size, device):
    if device == 'cuda':
        model_encoder.cuda()
        model_decoder.cuda()
    else:
        model_encoder.cpu()
        model_decoder.cpu()
    model_encoder.train()
    model_decoder.train()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loss = 0
    for x_cloud in train_loader:
        x_cloud = x_cloud.to(device)
        encoder_out = model_encoder(x_cloud)
        decoder_out = model_decoder(encoder_out)

        # loss = criterion(decoder_out, x_cloud)
        W = model_encoder.state_dict()['encoder.6.weight']
        loss = loss_function(W, x_cloud, decoder_out, encoder_out)

        train_loss += loss.data
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return train_loss


def Test_loop_CAE(model_encoder, model_decoder, criterion, test_dataset, batch_size, device):
    if device == 'cuda':
        model_encoder.cuda()
        model_decoder.cuda()
    else:
        model_encoder.cpu()
        model_decoder.cpu()
    model_encoder.eval()
    model_decoder.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loss = 0
    for x_cloud in test_loader:
        x_cloud = x_cloud.to(device)
        encoder_out = model_encoder(x_cloud)
        decoder_out = model_decoder(encoder_out)
        loss = criterion(decoder_out, x_cloud)
        test_loss += loss
    return test_loss


def vis_output_for_edge_cloud_input(model_encoder, model_decoder, criterion, vis_dataset, save_fig_path,
                                    device, epoch=None, loss_gate=None, epoch_gate=None):
    model_encoder.eval()
    model_decoder.eval()
    vis_batch_size = min(len(vis_dataset), 128)
    vis_dataloader = DataLoader(vis_dataset, batch_size=vis_batch_size, shuffle=False)
    save_large_loss_fig_path = save_fig_path + 'large_loss/'
    created_dir(save_large_loss_fig_path)

    origin_save_flag = [False for i in range(len(vis_dataset))]
    with torch.no_grad():
        vis_fig_cnt = 0
        for x_cloud in vis_dataloader:
            x_cloud = x_cloud.to(device)
            encoder_out = model_encoder(x_cloud)
            decoder_out = model_decoder(encoder_out)

            x_cloud_ = x_cloud.cpu().detach().numpy()
            decoder_out_ = decoder_out.cpu().detach().numpy()
            for i in range(vis_batch_size):
                vis_loss = criterion(decoder_out[i:i + 1, :], x_cloud[i:i + 1, :])
                vis_loss_ = vis_loss.cpu().detach().numpy()

                save_fig_path_i = save_fig_path + "fig_cnt" + str(vis_fig_cnt)
                if epoch is not None:
                    save_fig_path_i += "_epoch_" + str(epoch)
                save_fig_path_i += "_loss_" + str((int(vis_loss_ * 1000)) / 1000)

                # save large loss pic
                if loss_gate is not None and epoch > epoch_gate and vis_loss_ > loss_gate:
                    save_large_loss_fig_path_i = save_large_loss_fig_path + "_epoch_" + \
                                                 str(epoch) + "_loss_" + str((int(vis_loss_ * 1000)) / 1000)
                    plot_env_scatter(save_fig_path=save_large_loss_fig_path_i, env_cloud=decoder_out_[i])

                # save output fig
                plot_env_scatter(save_fig_path=save_fig_path_i, env_cloud=decoder_out_[i])

                # save origin fig
                if not origin_save_flag[vis_fig_cnt]:
                    save_fig_path_i_origin = save_fig_path + "fig_cnt" + str(vis_fig_cnt) + '_origin'
                    plot_env_scatter(save_fig_path=save_fig_path_i_origin, env_cloud=x_cloud_[i])
                    origin_save_flag[vis_fig_cnt] = True
                vis_fig_cnt += 1


def Train_Eval_CAE_main():
    # parm set
    device = 'cuda'

    encoder_input_size = 2800
    encoder_output_size = 28
    lr = 3 * 1e-4
    weight_decay = 0
    epoch_start = 0
    epoch_end = 50000

    train_data_load_file = "../../../output/data/envs_for_global_lidar_info/cloud_edge_1400_train_MPN_test.npy"
    train_data_load_file = "../../../output/data/S2D/obs_cloud_30000.npy"
    # test_data_load_file = "../../../output/data/envs_for_global_lidar_info/cloud_edge_750_test_0.2k.npy"

    model_name = "Autoencoder_CAE_edge_sca_1400_MPN_S2D_1"
    model_dir = '../../../output/model/Autoencoder/' + model_name + "/"
    load_checkpoint_flag = False
    checkpoint_load_file = '../../../output/model/Autoencoder/Autoencoder_CAE_edge_sca_1400_MPN_2/checkpoint_save/checkpoint_epoch_9000.pt'
    created_dir(model_dir)
    train_vis_fig_save_dir = model_dir + "train_vis_fig/"
    created_dir(train_vis_fig_save_dir)
    test_vis_fig_save_dir = model_dir + "test_vis_fig/"
    created_dir(test_vis_fig_save_dir)
    vis_loss_dir = model_dir + 'vis_loss/'
    created_dir(vis_loss_dir)
    checkpoint_save_dir = model_dir + 'checkpoint_save/'
    created_dir(checkpoint_save_dir)
    loss_save_dir = model_dir + 'loss_save/'
    created_dir(loss_save_dir)

    train_batch_size = 128
    test_batch_size = 64
    train_data_vis_cnt = 20
    test_data_vis_cnt = 1

    checkpoint_save_interval = 100
    vis_fig_save_interval = 50

    # For tensorboard vis, the dir can not be too long!
    tensorboard_dir = '../../../output/visualizations/tensorboard/' + model_name + '/exp1'
    writer = SummaryWriter(tensorboard_dir)

    # load dataset
    print("Start load dataset!")
    train_dataset = CAE_cloud_Dataset(data_file=train_data_load_file, data_len=29900)
    # test_dataset = CAE_cloud_Dataset(data_file=test_data_load_file, data_clx=encoder_input_size, data_len=200)

    train_vis_dataset = CAE_cloud_Dataset(data_file=train_data_load_file, data_len=train_data_vis_cnt)
    # test_vis_dataset = CAE_cloud_Dataset(data_file=test_data_load_file, data_clx=encoder_input_size,
    #                                      data_len=test_data_vis_cnt)
    print('Load dataset suc!')

    # load or create model and optimizer (checkpoint)
    model_encoder = Encoder_S2D(input_size=encoder_input_size, output_size=encoder_output_size)
    model_decoder = Decoder_S2D(input_size=encoder_output_size, output_size=encoder_input_size)
    model_encoder = model_encoder.float()
    model_decoder = model_decoder.float()
    if device == 'cuda':
        model_encoder.cuda()
        model_decoder.cuda()
    else:
        model_encoder.cpu()
        model_decoder.cpu()
    criterion = torch.nn.MSELoss(reduction="mean")
    # optimizer = torch.optim.Adam(
    #     [{'params': model_encoder.parameters(), 'lr': lr},
    #      {'params': model_decoder.parameters(), 'lr': lr}])

    optimizer = torch.optim.Adagrad(
        [{'params': model_encoder.parameters(), 'lr': lr},
         {'params': model_decoder.parameters(), 'lr': lr}])
    print('Create model and optimizer suc!')
    if load_checkpoint_flag:
        checkpoint = torch.load(checkpoint_load_file)
        model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
        model_decoder.load_state_dict(checkpoint['model_decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 好像也会存下来model.parameters()的cuda状态
        epoch_start = checkpoint['epoch']
        print("Load checkpoint suc!")

    train_loss_all = []
    test_loss_all = []
    train_data_size = len(train_dataset)
    # test_data_size = len(test_dataset)
    for epoch in range(epoch_start + 1, epoch_end + 1):
        print("---------Epoch--", epoch, "-------")
        # Train loop
        train_loss_i = Train_loop_CAE(model_encoder=model_encoder, model_decoder=model_decoder,
                                      optimizer=optimizer, criterion=criterion, train_dataset=train_dataset,
                                      batch_size=train_batch_size, device=device)
        train_loss_i_mean = float(train_loss_i.data) * train_batch_size / train_data_size
        print('Train loss is,', train_loss_i_mean)
        train_loss_all.append(train_loss_i_mean)

        # Eval loop
        # test_loss_i = Test_loop_CAE(model_encoder=model_encoder, model_decoder=model_decoder,
        #                             criterion=criterion, test_dataset=test_dataset,
        #                             batch_size=test_batch_size, device=device)
        # test_loss_i_mean = float(test_loss_i.data) * test_batch_size / test_data_size
        # print('Test loss is,', test_loss_i_mean)
        # test_loss_all.append(test_loss_i_mean)

        writer.add_scalars('loss', {
            'train': train_loss_i_mean,
            # 'test': test_loss_i_mean,
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
                'model_encoder_state_dict': model_encoder.state_dict(),
                'model_decoder_state_dict': model_decoder.state_dict(),
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
            vis_output_for_edge_cloud_input(model_encoder=model_encoder, model_decoder=model_decoder,
                                            criterion=criterion,
                                            vis_dataset=train_vis_dataset, save_fig_path=train_vis_fig_save_dir,
                                            device=device, epoch=epoch, loss_gate=10, epoch_gate=1000)
            # vis test dataset
            # vis_output_for_edge_cloud_input(model_encoder=model_encoder, model_decoder=model_decoder,
            #                                 criterion=criterion,
            #                                 vis_dataset=test_vis_dataset, save_fig_path=test_vis_fig_save_dir,
            #                                 device=device, epoch=epoch, loss_gate=10, epoch_gate=1000)

if __name__ == '__main__':
    pass

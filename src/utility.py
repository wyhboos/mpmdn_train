import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import struct
import numpy as np
import argparse
from plannet.Motion_model import S2D_MDN_Pnet, GMPN_S2D_CLOUD_MDN_Pnet
import torch
from cae.CAE_model import Encoder_S2D

def get_vertex_3D(obs):
    min_ = []
    max_ = []
    size_ = []
    mean_ = []
    obs_ = obs.reshape(-1, 3)
    obs_rec = []
    for i in range(10):
        obs_i_x = obs_[200 * i:200 * (i + 1), 0]
        obs_i_y = obs_[200 * i:200 * (i + 1), 1]
        obs_i_z = obs_[200 * i:200 * (i + 1), 2]
        min_x = np.min(obs_i_x)
        min_y = np.min(obs_i_y)
        min_z = np.min(obs_i_z)

        max_x = np.max(obs_i_x)
        max_y = np.max(obs_i_y)
        max_z = np.max(obs_i_z)
        min_.append([min_x, min_y, min_z])
        max_.append([max_x, max_y, max_z])
        size_.append([int(max_x - min_x + 0.5), int(max_y - min_y + 0.5), int(max_z - min_z + 0.5)])
        mean_.append([[(max_x + min_x) * 0.5, (max_y + min_y) * 0.5, (max_z + min_z) * 0.5]])
        obs_rec.append([(max_x + min_x) * 0.5 - 0.5 * int(max_x - min_x + 0.5),
                        (max_y + min_y) * 0.5 - 0.5 * int(max_y - min_y + 0.5),
                        (max_z + min_z) * 0.5 - 0.5 * int(max_z - min_z + 0.5),
                        (max_x + min_x) * 0.5 + 0.5 * int(max_x - min_x + 0.5),
                        (max_y + min_y) * 0.5 + 0.5 * int(max_y - min_y + 0.5),
                        (max_z + min_z) * 0.5 + 0.5 * int(max_z - min_z + 0.5),
                        ])

    # print("min_", min_)
    # print("max_", max_)
    # print("size_", size_)
    # print("mean_", mean_)
    # print("obs_rec", obs_rec)
    obs_rec = np.array(obs_rec).reshape(-1, 2, 3)
    # print(obs_rec.shape)
    return obs_rec

def load_3D_cloud_save():
    obs_path = "I:/Work/MPNdata/r-3d/dataset2/obs_cloud/"
    save_file = "../data/train/c3d/c3d_obs_cloud_50000.npy"
    obs_cloud_3d = []
    for i in range(50000):
        if i % 100 == 0:
            print(i)
        obs_file = obs_path + "obc" + str(i) + ".dat"
        temp = np.fromfile(obs_file)
        obs = np.array(temp).astype(np.float32)
        # print(obs.shape)
        obs_cloud_3d.append(obs)
        # get_vertex_3D(obs)
    obs_cloud_3d = np.array(obs_cloud_3d)
    print(obs_cloud_3d.shape)
    np.save(save_file, obs_cloud_3d)

def save_3D_obs_rec():
    obs_rec = []
    obs_rec_save_file = "../data/train/c3d/c3d_obs_rec_50000.npy"
    obs_cloud_file = "../data/train/c3d/c3d_obs_cloud_50000.npy"
    obs_cloud = np.load(obs_cloud_file)
    for i in range(50000):
        if i % 100 == 0:
            print(i)
        obs_cloud_i = obs_cloud[i:]
        rec_i = get_vertex_3D(obs_cloud_i)
        obs_rec.append(rec_i)
    obs_rec = np.array(obs_rec)
    print(obs_rec.shape)
    np.save(obs_rec_save_file, obs_rec)

def load_3D_point_path_save():
    path_path = "I:/Work/MPNdata/r-3d/dataset2/"
    save_file = "../data/train/c3d/c3d_path_e110_p5000.npy"
    path_all = []
    for i in range(110):
        print(i)
        path_env = []
        path_path_i = path_path + "e" + str(i) + "/"
        for j in range(5000):
            path_path_i_j = path_path_i + "path" + str(j) + ".dat"
            p = np.fromfile(path_path_i_j)
            p = np.array(p).astype(np.float32).reshape(-1, 3)
            # print(p.shape)
            path_env.append(p)
        path_all.append(path_env)
    np.save(save_file, np.array(path_all, dtype="object"))

# def load_3D_obs_rec_save():
#     obs_location_file = "I:/Work/MPNdata/r-3d/dataset2/obs.dat"
#     loc = np.fromfile(obs_location_file)
#     print(loc.shape)
#     print(loc.reshape(-1, 3))
#
#     obs_order_file = "I:/Work/MPNdata/r-3d/dataset2/obs_perm2.dat"
#     order = np.fromfile(obs_order_file, dtype=int)
#     print(order.shape)
#     print(order[:100])
def divide_cloud_to_train_test():
    cloud_file = "../data/train/s2d/obs_cloud_30000_random.npy"
    train_data_save_file = "../data/train/s2d/obs_cloud_30000_rd_train.npy"
    test_data_save_file = "../data/train/s2d/obs_cloud_30000_rd_test.npy"
    cloud_all = np.load(cloud_file)
    cloud_train = cloud_all[:24000, :]
    cloud_test = cloud_all[24000:, :]

    print(cloud_all.shape)
    np.save(train_data_save_file, cloud_train)
    np.save(test_data_save_file, cloud_test)

def get_libtorch_model():

    # print("encoder")
    # encoder = Encoder_S2D(input_size=2800, output_size=28)
    # checkpoint_load_file = '../../../output/model/Autoencoder/Autoencoder_CAE_edge_sca_1400_MPN_S2D_1/checkpoint_save/checkpoint_epoch_500.pt'
    # checkpoint = torch.load(checkpoint_load_file)
    # encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
    # sm_encoder = torch.jit.script(encoder)
    # i_encoder = torch.rand(1, 2800)
    # o = sm_encoder(i_encoder)
    # sm_encoder.save("../../../output/model/Encoder_S2D.pt")

    # print("MDN")
    # mdn = GMPN_S2D_CLOUD_MDN_Pnet(input_size=42, output_size=7, mixture_num=20)

    # checkpoint_load_file = "../data/model/models/MDN_ARM_1_checkpoint_epoch_1300.pt"
    # checkpoint = torch.load(checkpoint_load_file)
    # mdn.load_state_dict(checkpoint['model_state_dict'])
    # sm_mdn = torch.jit.script(mdn)
    # # i_mdn_x_e = torch.rand(1, 28)
    # # i_mdn_x_c = torch.rand(1, 2)
    # # i_mdn_x_g = torch.rand(1, 2)
    # # o_mdn = sm_mdn(i_mdn_x_e, i_mdn_x_c, i_mdn_x_g)
    # # for i in o_mdn:
    # #     print(i.shape)
    # sm_mdn.save("../data/model/models/MDN_ARM_1_ckp_1300_libtorch.pt")
    #
    # mdn = GMPN_S2D_CLOUD_MDN_Pnet(input_size=42, output_size=7, mixture_num=20)

    # checkpoint_load_file = "../data/model/models/MDN_ARM_2_checkpoint_epoch_1000.pt"
    # checkpoint = torch.load(checkpoint_load_file)
    # mdn.load_state_dict(checkpoint['model_state_dict'])
    # sm_mdn = torch.jit.script(mdn)
    # # i_mdn_x_e = torch.rand(1, 28)
    # # i_mdn_x_c = torch.rand(1, 2)
    # # i_mdn_x_g = torch.rand(1, 2)
    # # o_mdn = sm_mdn(i_mdn_x_e, i_mdn_x_c, i_mdn_x_g)
    # # for i in o_mdn:
    # #     print(i.shape)
    # sm_mdn.save("../data/model/models/MDN_ARM_2_ckp_1000_libtorch.pt")

    # mdn = GMPN_S2D_CLOUD_MDN_Pnet(input_size=36, output_size=4, mixture_num=20)
    #
    # checkpoint_load_file = "../data/model/models/MDN_S2D_TL_2_checkpoint_epoch_1000.pt"
    # checkpoint = torch.load(checkpoint_load_file)
    # mdn.load_state_dict(checkpoint['model_state_dict'])
    # sm_mdn = torch.jit.script(mdn)
    # # i_mdn_x_e = torch.rand(1, 28)
    # # i_mdn_x_c = torch.rand(1, 2)
    # # i_mdn_x_g = torch.rand(1, 2)
    # # o_mdn = sm_mdn(i_mdn_x_e, i_mdn_x_c, i_mdn_x_g)
    # # for i in o_mdn:
    # #     print(i.shape)
    # sm_mdn.save("../data/model/models/MDN_S2D_TL_2_ckp_1000_libtorch.pt")

    # print("mpn")
    # mpn = S2D_MDN_Pnet(42, 7)
    # checkpoint_load_file = "../data/model/models/MPN_ARM_1_checkpoint_epoch_500.pt"
    # checkpoint = torch.load(checkpoint_load_file)
    # mpn.load_state_dict(checkpoint['model_state_dict'])
    # sm_mpn = torch.jit.script(mpn)
    # # # i_mpn_x_e = torch.rand(1, 28)
    # # # i_mpn_x_c = torch.rand(1, 2)
    # # # i_mpn_x_g = torch.rand(1, 2)
    # # # o_mpn = sm_mpn(i_mpn_x_e, i_mpn_x_c, i_mpn_x_g)
    # # # for i in o_mpn:
    # # #     print(i.shape)
    # sm_mpn.save("../data/model/models/MPN_ARM_1_ckp_500_libtorch.pt")

    mpn = S2D_MDN_Pnet(42, 7)
    checkpoint_load_file = "../data/model/models/MPN_ARM_3_checkpoint_epoch_100.pt"
    checkpoint = torch.load(checkpoint_load_file)
    mpn.load_state_dict(checkpoint['model_state_dict'])
    sm_mpn = torch.jit.script(mpn)
    # # i_mpn_x_e = torch.rand(1, 28)
    # # i_mpn_x_c = torch.rand(1, 2)
    # # i_mpn_x_g = torch.rand(1, 2)
    # # o_mpn = sm_mpn(i_mpn_x_e, i_mpn_x_c, i_mpn_x_g)
    # # for i in o_mpn:
    # #     print(i.shape)
    sm_mpn.save("../data/model/models/MPN_ARM_3_ckp_100_libtorch.pt")


    # mpn = S2D_MDN_Pnet(36, 4)
    # checkpoint_load_file = "../data/model/models/MPN_S2D_TL_2_checkpoint_epoch_380.pt"
    # checkpoint = torch.load(checkpoint_load_file)
    # mpn.load_state_dict(checkpoint['model_state_dict'])
    # sm_mpn = torch.jit.script(mpn)
    # # # i_mpn_x_e = torch.rand(1, 28)
    # # # i_mpn_x_c = torch.rand(1, 2)
    # # # i_mpn_x_g = torch.rand(1, 2)
    # # # o_mpn = sm_mpn(i_mpn_x_e, i_mpn_x_c, i_mpn_x_g)
    # # # for i in o_mpn:
    # # #     print(i.shape)
    # sm_mpn.save("../data/model/models/MPN_S2D_TL_2_ckp_380.pt")

def divide_s2d_cloud():
    cloud_file = "../data/train/s2d/obs_cloud_30000.npy"
    cloud_data = np.load(cloud_file)
    print(cloud_data.shape)
    cloud_data_2000 = cloud_data[:2000, :]
    np.save("../data/train/s2d/obs_cloud_2000.npy", cloud_data_2000)

def divide_cloud():
    cloud_file = "../data/train/c3d/c3d_obs_cloud_50000.npy"
    cloud_small_save_file = "../data/train/c3d/c3d_obs_cloud_2000.npy"
    cloud_all = np.load(cloud_file)
    cloud_small = cloud_all[:2000, :]

    print(cloud_all.shape)
    np.save(cloud_small_save_file, cloud_small)

def make_cloud_random():
    cloud_file = "../data/train/s2d/obs_cloud_30000.npy"
    cloud_data = np.load(cloud_file)
    print(cloud_data.shape)
    random_cloud_all = []
    for i in range(cloud_data.shape[0]):
        print(i)
        cloud_i = cloud_data[i, :]
        cloud_i = cloud_i.reshape(1400, 2)
        np.random.shuffle(cloud_i)
        cloud_i = cloud_i.reshape(2800)
        random_cloud_all.append(cloud_i)
    random_cloud_all = np.array(random_cloud_all)
    print(random_cloud_all.shape)
    np.save("../data/train/s2d/obs_cloud_30000_random.npy", random_cloud_all)

def S2D_get_Joint_Train_data():
    cloud_file = np.load("../data/train/s2d/obs_cloud_30000_random.npy")
    # paths = np.load("../../data/train/s2d/1000env_400pt/S2D_Three_Link_Path_all.npy", allow_pickle=True)
    paths = np.load("../data/train/s2d/1000env_400pt/S2D_Three_Link_Path_all.npy", allow_pickle=True)
    train_data = []
    train_data_env = []
    train_data_current = []
    train_data_target = []
    train_data_next = []
    train_env_index = []

    test_data = []
    test_data_env = []
    test_data_current = []
    test_data_target = []
    test_data_next = []
    test_env_index = []
    for i in range(1000):
        # print("i", i)
        env_latent_i = cloud_file[i, :]
        env_latent_i = env_latent_i.reshape(2800)
        env_i_paths = paths[i]
        for j in range(len(env_i_paths)):
            env_i_path_j = env_i_paths[j]
            # l = env_i_path_j.shape[0]
            l = len(env_i_path_j)
            print("i,j,l", i, j, l)
            for k in range(l - 1):
                target = env_i_path_j[l - 1]
                current = env_i_path_j[k]
                next = env_i_path_j[k + 1]
                if i < 900:
                    train_data_env.append(env_latent_i)
                    train_data_current.append(current)
                    train_data_target.append(target)
                    train_data_next.append(next)
                    train_env_index.append(i)
                else:
                    test_data_env.append(env_latent_i)
                    test_data_current.append(current)
                    test_data_target.append(target)
                    test_data_next.append(next)
                    test_env_index.append(i)

    train_data_env = np.array(train_data_env, dtype=np.float32)
    train_data_current = np.array(train_data_current, dtype=np.float32)
    train_data_target = np.array(train_data_target, dtype=np.float32)
    train_data_next = np.array(train_data_next, dtype=np.float32)
    train_env_index = np.array(train_env_index, dtype=np.float32)
    train_env_index = train_env_index.reshape(-1, 1)

    train_data = np.concatenate((train_data_current, train_data_target, train_data_next, train_env_index), axis=1)
    print(train_data.shape)

    test_data_env = np.array(test_data_env, dtype=np.float32)
    test_data_current = np.array(test_data_current, dtype=np.float32)
    test_data_target = np.array(test_data_target, dtype=np.float32)
    test_data_next = np.array(test_data_next, dtype=np.float32)
    test_env_index = np.array(test_env_index, dtype=np.float32)
    test_env_index = test_env_index.reshape(-1, 1)
    test_data = np.concatenate(
        (test_data_current, test_data_target, test_data_next, test_env_index), axis=1)
    print(test_data.shape)
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    np.save("../data/train/s2d/1000env_400pt/S2D_Three_Link_Joint_train.npy", train_data)
    np.save("../data/train/s2d/1000env_400pt/S2D_Three_Link_Joint_test.npy", test_data)

def change_cloud_dim():
    cloud_file = np.load("../data/train/s2d/obs_cloud_30000_random.npy")
    print(cloud_file[0:1, 0:14])
    cloud_file_dim = cloud_file.reshape(-1, 1400, 2)
    cloud_file_dim_tran = np.transpose(cloud_file_dim, (0, 2, 1))
    print(cloud_file_dim_tran.shape)
    print(cloud_file_dim[0:1, :7, 0])
    print(cloud_file_dim[0:1, :7, 1])

    print(cloud_file_dim_tran[0:1, 0, :7])
    print(cloud_file_dim_tran[0:1, 1, :7])
    np.save("../data/train/s2d/obs_cloud_30000_2_1400_rd.npy", cloud_file_dim_tran)

if __name__ == '__main__':
    # change_cloud_dim()
    S2D_get_Joint_Train_data()
    # divide_cloud_to_train_test()
    # make_cloud_random()
    # divide_cloud()
    # divide_s2d_cloud()
    # get_libtorch_model()

    # divide_cloud_to_train_test()

    # save_3D_obs_rec()
    # load_3D_cloud_save()
    # load_3D_point_path_save()
    # load_3D_obs_rec_save()

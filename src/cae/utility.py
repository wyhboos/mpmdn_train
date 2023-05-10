import numpy as np

from CAE_Train_Test import *
def encode_obs_main_ompl():
    paths = np.load("../../../output/data/S2D/S2D_OMPL/S2D_Data/S2D_Two_Link_Path_all.npy", allow_pickle=True)
    envs = np.load("../../../output/data/S2D/obs_cloud_30000.npy", allow_pickle=True)
    model_encoder = Encoder_S2D(input_size=2800, output_size=28)
    model_encoder.cuda()
    checkpoint_load_file = '../../../output/model/Autoencoder/Autoencoder_CAE_edge_sca_1400_MPN_S2D_1/checkpoint_save/checkpoint_epoch_500.pt'
    checkpoint = torch.load(checkpoint_load_file)
    model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
    envs = envs[:100, :]
    print(envs[:1, :20])
    env_cloud = envs.reshape(100, 2800)
    env_cloud_t = torch.tensor(env_cloud, dtype=torch.float32)
    env_cloud_t = env_cloud_t.to('cuda')
    env_latent = model_encoder(env_cloud_t)
    env_latent = env_latent.cpu().detach().numpy()
    print(env_latent[:3, :])

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
    for i in range(100):
        # print("i", i)
        env_latent_i = env_latent[i, :]
        env_latent_i = env_latent_i.reshape(28)
        env_i_paths = paths[i]
        for j in range(len(env_i_paths)):
            print("i,j", i, j)
            env_i_path_j = env_i_paths[j]
            # l = env_i_path_j.shape[0]
            l = len(env_i_path_j)
            for k in range(l - 1):
                target = env_i_path_j[l - 1]
                current = env_i_path_j[k]
                next = env_i_path_j[k + 1]
                if i < 90:
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

    train_data = np.concatenate((train_data_env, train_data_current, train_data_target, train_data_next, train_env_index), axis=1)
    print(train_data.shape)

    test_data_env = np.array(test_data_env, dtype=np.float32)
    test_data_current = np.array(test_data_current, dtype=np.float32)
    test_data_target = np.array(test_data_target, dtype=np.float32)
    test_data_next = np.array(test_data_next, dtype=np.float32)
    test_env_index = np.array(test_env_index, dtype=np.float32)
    test_env_index = test_env_index.reshape(-1, 1)
    test_data = np.concatenate(
        (test_data_env, test_data_current, test_data_target, test_data_next, test_env_index), axis=1)
    print(test_data.shape)
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    # np.save("../../../output/data/S2D/S2D_OMPL/S2D_Data/S2D_Two_Link_train.npy", train_data)
    # np.save("../../../output/data/S2D/S2D_OMPL/S2D_Data/S2D_Two_Link_test.npy", test_data)

# cat 10 part of paths
def cat_paths():
    file = "../../data/train/s2d/1000env_400pt/S2D_Three_Link_Path_"
    path_r = []
    for i in range(10):
        file_i = np.load(file+str(i)+".npy", allow_pickle=True)
        path_r += list(file_i)
    np.save("../../data/train/s2d/1000env_400pt/S2D_Three_Link_Path_all", np.array(path_r))

    # file = "../../data/train/s2d/1000env_400pt/S2D_Two_Link_Path_"
    # path_r = []
    # for i in range(10):
    #     file_i = np.load(file+str(i)+".npy", allow_pickle=True)
    #     path_r += list(file_i)
    # np.save("../../data/train/s2d/1000env_400pt/S2D_Two_Link_Path_all", np.array(path_r))

def encode_s2d_cloud_to_latent():
    envs = np.load("../../data/train/s2d/obs_cloud_30000.npy", allow_pickle=True)
    model_encoder = Encoder_S2D(input_size=2800, output_size=28)
    model_encoder.cuda()
    checkpoint_load_file = "../../data/model/cae_s2d/checkpoint_epoch_500.pt"
    checkpoint = torch.load(checkpoint_load_file)
    model_encoder.load_state_dict(checkpoint['model_encoder_state_dict'])
    env_cloud = envs.reshape(30000, 2800)
    env_cloud_t = torch.tensor(env_cloud, dtype=torch.float32)
    env_cloud_t = env_cloud_t.to('cuda')
    env_latent = model_encoder(env_cloud_t)
    env_latent = env_latent.cpu().detach().numpy()
    print(env_latent.shape)
    np.save("../../data/train/s2d/s2d_env_latent_30000", env_latent)

def obtain_train_eval_data():
    """
    Arm data only consists of one env with 50000 paths
    :return:
    """
    # paths = np.load("../../data/train/s2d/1000env_400pt/S2D_Three_Link_Path_all.npy", allow_pickle=True)
    paths = np.load("../../data/train/panda_arm/pathpart_49999.npy", allow_pickle=True)
    paths = [paths]

    # paths = np.load("../../data/train/s2d/1000env_400pt/S2D_Three_Link_Path_all.npy", allow_pickle=True)
    env_latent = np.load("../../data/train/s2d/s2d_env_latent_30000.npy", allow_pickle=True)
    env_latent = np.ones((1, 28))
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
    for i in range(1):
        # print("i", i)
        env_latent_i = env_latent[i, :]
        env_latent_i = env_latent_i.reshape(28)
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
                if i < 40000:
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

    train_data = np.concatenate((train_data_env, train_data_current, train_data_target, train_data_next, train_env_index), axis=1)
    print(train_data.shape)

    test_data_env = np.array(test_data_env, dtype=np.float32)
    test_data_current = np.array(test_data_current, dtype=np.float32)
    test_data_target = np.array(test_data_target, dtype=np.float32)
    test_data_next = np.array(test_data_next, dtype=np.float32)
    test_env_index = np.array(test_env_index, dtype=np.float32)
    test_env_index = test_env_index.reshape(-1, 1)
    test_data = np.concatenate(
        (test_data_env, test_data_current, test_data_target, test_data_next, test_env_index), axis=1)
    print(test_data.shape)
    np.random.shuffle(test_data)
    np.random.shuffle(train_data)
    np.save("../../data/train/panda_arm/arm_train.npy", train_data)
    np.save("../../data/train/panda_arm/arm_test.npy", test_data)

def divide_arm_data():
    train = np.load("../../data/train/panda_arm/arm_train.npy")
    test = np.load("../../data/train/panda_arm/arm_test.npy")
    l_train = train.shape[0]
    l_test = test.shape[0]
    np.save("../../data/train/panda_arm/arm_train_s.npy", train[:int(0.1 * l_train), :])
    np.save("../../data/train/panda_arm/arm_test_s.npy", train[:int(0.1 * l_test), :])

if __name__ == '__main__':
    # cat_paths()
    # encode_s2d_cloud_to_latent()
    # obtain_train_eval_data()
    divide_arm_data()
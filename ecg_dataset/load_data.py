import numpy as np
import pickle
# from COCA.ecg_dataset.load_data_chaosuan import load_chaosuan_data
from VF_fake.ecg_dataset.load_data_cpsc2021 import load_data_cpsc2021
# from COCA.ecg_dataset.load_data_icential11k import load_icentia11k_data
from VF_fake.ecg_dataset.load_data_ptbxl import load_ptbxl_data
from VF_fake.ecg_dataset.load_data_cudb import load_cudb_data
from VF_fake.ecg_dataset.load_data_mitbih import load_mitbih_data
from VF_fake.ecg_dataset.load_vfdb_data import load_vfdb_data


def load_cpsc(opt):
    global train_dataset, val_dataset, test_dataset
    # train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
    #     '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000)

    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
        'E:\研究生\data\CPSC\\', 1000)
    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label

def load_icentia11k(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
        '/data/icentia11k/', opt.seed)

    # if NOT_ALL
    train_data = np.expand_dims(train_data, 1)
    val_normal_data = np.expand_dims(val_normal_data, 1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, 1)
    test_normal_data = np.expand_dims(test_normal_data, 1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, 1)

    # train_data = np.stack((train_data,) * 2, axis=1)
    # val_normal_data = np.stack((val_normal_data,) * 2, axis=1)
    # val_abnormal_data = np.stack((val_abnormal_data,) * 2, axis=1)
    # test_normal_data = np.stack((test_normal_data,) * 2, axis=1)
    # test_abnormal_data = np.stack((test_abnormal_data,) * 2, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)


    return train_data, train_label, val_data, val_label, test_data, test_label



def load_ptbxl(opt):
    global train_dataset, val_dataset, test_dataset
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_ptbxl_data(opt)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label



def load_chaosuan(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, train_data_abnormal = load_chaosuan_data(opt)

    # if NOT_ALL
    train_data = np.expand_dims(train_data, 1)
    val_normal_data = np.expand_dims(val_normal_data, 1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, 1)
    test_normal_data = np.expand_dims(test_normal_data, 1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, 1)
    train_data_abnormal = np.expand_dims(train_data_abnormal, 1)

    # train_data = np.stack((train_data,) * 2, axis=1)
    # val_normal_data = np.stack((val_normal_data,) * 2, axis=1)
    # val_abnormal_data = np.stack((val_abnormal_data,) * 2, axis=1)
    # test_normal_data = np.stack((test_normal_data,) * 2, axis=1)
    # test_abnormal_data = np.stack((test_abnormal_data,) * 2, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                             val_normal_data,
                                                                                             val_abnormal_data,
                                                                                             test_normal_data,
                                                                                             test_abnormal_data,
                                                                                             opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label, train_data_abnormal

def load_vfdb(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_vfdb_data()
    train_data = np.expand_dims(train_data, axis=1)
    val_normal_data = np.expand_dims(val_normal_data, axis=1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    test_normal_data = np.expand_dims(test_normal_data, axis=1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                         val_normal_data,
                                                                                         val_abnormal_data,
                                                                                         test_normal_data,
                                                                                         test_abnormal_data,
                                                                                         opt.seed)

    return train_data,train_label, val_data, val_label, test_data, test_label

def load_noise(opt):
    normal_class = 0
    if opt.dataset == 'BUTQDB':
        # with open('E:\研究生\data\BUTQDB\\ClassA.pickle', 'rb') as handle1:
        with open('/home/jinjiahao/data/BUTQDB/ClassA.pickle', 'rb') as handle1:
            data_X_N = pickle.load(handle1)
            data_X_N = data_X_N[:18414000].reshape(-1, 1000)
            data_Y_N = np.zeros(len(data_X_N))

        # with open('E:\研究生\data\BUTQDB\\ClassB.pickle', 'rb') as handle2:
        with open('/home/jinjiahao/data/BUTQDB/ClassB.pickle', 'rb') as handle2:
            data_X_A_1 = pickle.load(handle2)
            data_X_A_1 = data_X_A_1[:11923000].reshape(-1, 1000)

        # with open('E:\研究生\data\BUTQDB\\ClassC.pickle', 'rb') as handle2:
        with open('/home/jinjiahao/data/BUTQDB/ClassC.pickle', 'rb') as handle2:
            data_X_A_2 = pickle.load(handle2)
            data_X_A_2 = data_X_A_2[:5444000].reshape(-1, 1000)
            data_X_A = np.concatenate((data_X_A_1, data_X_A_2))
            data_Y_A = np.ones(len(data_X_A))

        data_X = np.concatenate((data_X_N, data_X_A))  # (18414,1000)  (17367,1000)
        data_Y = np.concatenate((data_Y_N, data_Y_A))



    elif opt.dataset == 'CinC2011':
        with open('/home/huangxunhua/ECG_Quality/CICC2011/Cinc2011_Accept.pickle', 'rb') as handle1:

            data_X_N = pickle.load(handle1)
            data_Y_N = np.zeros(len(data_X_N))

        with open('/home/huangxunhua/ECG_Quality/CICC2011/Cinc2011_UnAccept.pickle', 'rb') as handle2:
            data_X_A = pickle.load(handle2)
            data_Y_A = np.ones(len(data_X_A))

        data_X = np.concatenate((data_X_N, data_X_A))  # (773,12,1000)  (225,12,1000)
        data_Y = np.concatenate((data_Y_N, data_Y_A))
    elif opt.dataset == 'Icentia11K':
        with open('/home/huangxunhua/ECG_Quality/Icentia11K/Icentia11K_Clean.pickle', 'rb') as handle1:
            data_X_N = pickle.load(handle1)
            data_Y_N = np.zeros(len(data_X_N))

        with open('/home/huangxunhua/ECG_Quality/Icentia11K/Icentia11K_Noisy.pickle', 'rb') as handle2:
            data_X_A = pickle.load(handle2)
            data_Y_A = np.ones(len(data_X_A))

        data_X = np.concatenate((data_X_N, data_X_A))  # (773,12,1000)  (225,12,1000)
        data_Y = np.concatenate((data_Y_N, data_Y_A))
    elif opt.dataset == 'CinC2017':
        with open('/home/huangxunhua/ECG_Quality/CICC2017/Cinc2017_Accept.pickle', 'rb') as handle1:

            data_X_N = pickle.load(handle1)
            data_Y_N = np.zeros(len(data_X_N))

        with open('/home/huangxunhua/ECG_Quality/CICC2017/Cinc2017_UnAccept.pickle', 'rb') as handle2:
            data_X_A = pickle.load(handle2)
            data_Y_A = np.ones(len(data_X_A))

        data_X = np.concatenate((data_X_N, data_X_A))  # (5116,3001) (319,3001)
        data_Y = np.concatenate((data_Y_N, data_Y_A))
    elif opt.dataset == 'EEG_Physiobank':
        with open(
                '/home/changhuihui/learn_project/Deep-SVDD-Noisy3/Deep-SVDD-Noisy3/EEG_signal/EEG_03data/EEG_clean.pkl',
                'rb') as handle1:
            data_X_N = pickle.load(handle1)
            data_Y_N = np.zeros(len(data_X_N))
        with open(
                '/home/changhuihui/learn_project/Deep-SVDD-Noisy3/Deep-SVDD-Noisy3/EEG_signal/EEG_03data/EEG_noise.pkl',
                'rb') as handle2:
            data_X_A = pickle.load(handle2)
            data_Y_A = np.ones(len(data_X_A))
        data_X = np.concatenate((data_X_N, data_X_A))
        data_Y = np.concatenate((data_Y_N, data_Y_A))
    elif opt.dataset == 'EEGSemi-simulated':
        with open(
                '/home/changhuihui/learn_project/Deep-SVDD-Noisy3/Deep-SVDD-Noisy3/EEG_signal/EEG_02data/EEG_clean.pkl',
                'rb') as handle1:
            data_X_N = pickle.load(handle1)
            data_Y_N = np.zeros(len(data_X_N))
        with open(
                '/home/changhuihui/learn_project/Deep-SVDD-Noisy3/Deep-SVDD-Noisy3/EEG_signal/EEG_02data/EEG_noise.pkl',
                'rb') as handle2:
            data_X_A = pickle.load(handle2)
            data_Y_A = np.ones(len(data_X_A))

        data_X = np.concatenate((data_X_N, data_X_A))  # (773,12,1000)  (225,12,1000)
        data_Y = np.concatenate((data_Y_N, data_Y_A))
    elif opt.dataset == 'BUTPPG':

        with open('/home/changhuihui/learn_project/Deep-SVDD-Noisy3/Deep-SVDD-Noisy3/PPG_Signal/good_class.pkl',
                  'rb') as handle1:

            data_X_N = pickle.load(handle1)
            data_Y_N = np.zeros(len(data_X_N))

        with open('/home/changhuihui/learn_project/Deep-SVDD-Noisy3/Deep-SVDD-Noisy3/PPG_Signal/bad_class.pkl',
                  'rb') as handle2:
            data_X_A = pickle.load(handle2)
            data_Y_A = np.ones(len(data_X_A))

        data_X = np.concatenate((data_X_N, data_X_A))  # (773,12,1000)  (225,12,1000)
        data_Y = np.concatenate((data_Y_N, data_Y_A))

    elif opt.dataset == 'ZZU_Noisy':

        clean_root_dir = '/repository/dataset/ZZUnoise_chh/ZZU_Noisy/正常'
        noisy_root_dir = '/repository/dataset/ZZUnoise_chh/ZZU_Noisy/干扰信号'

        print('Loading ZZU Noisy All Channel Data.....')
        x_clean, y_clean, x_noisy, y_noisy = load_ZZU_noisy_all_channel(clean_root_dir, noisy_root_dir)

        data_X_N = x_clean
        data_Y_N = y_clean
        data_X_A = x_noisy
        data_Y_A = y_noisy

        data_X = np.concatenate((data_X_N, data_X_A))  # (5116,3001) (319,3001)
        data_Y = np.concatenate((data_Y_N, data_Y_A))

    label_idxs = np.unique(data_Y)

    class_stat = {}
    for idx in label_idxs:
        class_stat[idx] = len(np.where(data_Y == idx)[0])

    if normal_class >= len(label_idxs):
        normal_idx = normal_class % len(label_idxs)
    else:
        normal_idx = normal_class

    if opt.dataset == 'EpilepticSeizure':
        labels_binary, idx_normal, idx_abnormal = one_class_labeling_sz(data_Y, normal_idx, opt.seed)

    else:
        labels_binary, idx_normal, idx_abnormal = one_class_labeling(data_Y, normal_idx, opt.seed)

    data_N_X = data_X[idx_normal]
    data_N_Y = labels_binary[idx_normal]
    data_A_X = data_X[idx_abnormal]
    data_A_Y = labels_binary[idx_abnormal]

    # Split normal samples
    n_normal = data_N_X.shape[0]
    train_X = data_N_X[:(int(n_normal * 0.6)), ]
    train_Y = data_N_Y[:(int(n_normal * 0.6)), ]

    val_N_X = data_N_X[int(n_normal * 0.6):int(n_normal * 0.8)]
    val_N_Y = data_N_Y[int(n_normal * 0.6):int(n_normal * 0.8)]
    test_N_X = data_N_X[int(n_normal * 0.8):]
    test_N_Y = data_N_Y[int(n_normal * 0.8):]

    data_A_X_len = data_A_X.shape[0]

    val_N_X_len = data_A_X_len // 2
    test_N_X_len = data_A_X_len // 2

    data_A_X_idx = list(range(data_A_X_len))

    val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
    test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

    val_X = np.concatenate((val_N_X, val_A_X))
    val_Y = np.concatenate((val_N_Y, val_A_Y))
    test_X = np.concatenate((test_N_X, test_A_X))
    test_Y = np.concatenate((test_N_Y, test_A_Y))


    dataIndex = list(range(val_X.shape[0]))
    np.random.seed(1024)
    np.random.shuffle(dataIndex)
    val_X = val_X[dataIndex]
    val_Y = val_Y[dataIndex]

    dataIndex = list(range(test_X.shape[0]))
    np.random.seed(1024)
    np.random.shuffle(dataIndex)
    test_X = test_X[dataIndex]
    test_Y = test_Y[dataIndex]

    print('shuffle')

    print("[INFO] Labels={}, normal label={}".format(label_idxs, normal_class))
    print("[INFO] Train: normal={}".format(train_X.shape), )
    print("[INFO] Val normal={}, abnormal={}".format(val_N_X.shape[0], val_A_X.shape[0]), )
    print("[INFO] Test normal={}, abnormal={}".format(test_N_X.shape[0], test_A_X.shape[0]), )

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def one_class_labeling(labels, normal_idx, seed=None):
    """
    将多类别标签转换为一类正常 (0) 和一类异常 (1)。

    :param labels: 原始的多类别标签 (numpy array)
    :param normal_idx: 指定的正常类索引
    :param seed: 随机种子（可选）
    :return: 二分类标签, 正常类索引, 异常类索引
    """
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # 初始化二分类标签
    binary_labels = np.zeros_like(labels, dtype=np.int32)

    # 标记正常类
    binary_labels[labels == normal_idx] = 0

    # 标记异常类
    binary_labels[labels != normal_idx] = 1

    # 获取正常和异常样本的索引
    idx_normal = np.where(binary_labels == 1)[0]
    idx_abnormal = np.where(binary_labels == 0)[0]

    return binary_labels, idx_normal, idx_abnormal

def load_mitbih(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_mitbih_data()
    train_data = np.expand_dims(train_data, axis=1)
    val_normal_data = np.expand_dims(val_normal_data, axis=1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    test_normal_data = np.expand_dims(test_normal_data, axis=1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                         val_normal_data,
                                                                                         val_abnormal_data,
                                                                                         test_normal_data,
                                                                                         test_abnormal_data,
                                                                                         opt.seed)

    return train_data, train_label, val_data, val_label, test_data, test_label

def load_cudb(opt):
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_cudb_data()
    train_data = np.expand_dims(train_data, axis=1)
    val_normal_data = np.expand_dims(val_normal_data, axis=1)
    val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
    test_normal_data = np.expand_dims(test_normal_data, axis=1)
    test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

    train_data, train_label, val_data, val_label, test_data, test_label = get_data_label(train_data,
                                                                                         val_normal_data,
                                                                                         val_abnormal_data,
                                                                                         test_normal_data,
                                                                                         test_abnormal_data,
                                                                                         opt.seed)
    return train_data, train_label, val_data, val_label, test_data, test_label









if __name__ == '__main__':
    from options import Options
    opt = Options().parse()
    train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_cpsc2021(
        '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', 1000)
















def get_data_label(train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data, SEED):
    # train_normal_data, test_normal_data, test_abnormal_data = load_data(root)

    len_train = train_data.shape[0]
    len_val_normal = val_normal_data.shape[0]
    len_val_abnormal = val_abnormal_data.shape[0]

    len_test_normal = test_normal_data.shape[0]
    len_test_abnormal = test_abnormal_data.shape[0]

    train_label = np.zeros(len_train)
    val_data = np.concatenate([val_normal_data, val_abnormal_data], axis=0)
    val_label = np.concatenate([np.zeros(len_val_normal), np.ones(len_val_abnormal)], axis=0)

    test_data = np.concatenate([test_normal_data, test_abnormal_data], axis=0)
    test_label = np.concatenate([np.zeros(len_test_normal), np.ones(len_test_abnormal)], axis=0)

    train_label, train_idx = shuffle_label(train_label, SEED)
    train_data = train_data[train_idx]

    val_label, val_idx = shuffle_label(val_label, SEED)
    val_data = val_data[val_idx]
    val_label = val_label[val_idx]

    test_label, test_idx = shuffle_label(test_label, SEED)
    test_data = test_data[test_idx]
    test_label = test_label[test_idx]

    return train_data, train_label, val_data, val_label, test_data, test_label

def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    return labels.astype("bool"), index
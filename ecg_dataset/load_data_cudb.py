import numpy as np


import numpy as np

def load_cudb_data():
    # vt = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/vt.npy')
    vt = np.load('E:\研究生\data\cudb\\vt.npy')
    vt = vt.reshape(-1, 2500)  # 5,2500
    vt = np.nan_to_num(vt)

    # normal = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/normal.npy')
    normal = np.load('E:\研究生\data\cudb\\normal.npy')
    normal = normal[:49132500].reshape(-1, 2500)
    normal = np.nan_to_num(normal)

    # vfvfl = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/vfvfl.npy')
    vfvfl = np.load('E:\研究生\data\cudb\\vfvfl.npy')
    vfvfl = vfvfl[:770000].reshape(-1, 2500)  # 308 2500
    vfvfl = np.nan_to_num(vfvfl)

    # noisy = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/cudb/noisy.npy')
    noisy = np.load('E:\研究生\data\cudb\\noisy.npy')
    noisy = noisy.reshape(-1, 2500)
    noisy = np.nan_to_num(noisy)

    va = np.concatenate((vt, vfvfl), axis=0)
    no_va = np.concatenate((noisy, normal), axis=0)
    no_va = no_va[:942, :]
    no_va = np.nan_to_num(no_va)

    data_X_va = va
    data_X_normal = no_va

    data_len = len(data_X_normal)
    train_X = data_X_normal[0:data_len * 2 // 3]

    val_normal_X = data_X_normal[data_len * 2 // 3:data_len * 5 // 6]
    val_abnormal_X = data_X_va[0:len(data_X_va) // 2]

    test_normal_X = data_X_normal[data_len * 5 // 6:]
    test_abnormal_X = data_X_va[len(data_X_va) // 2:]

    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X


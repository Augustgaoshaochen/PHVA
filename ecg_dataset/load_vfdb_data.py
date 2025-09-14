import numpy as np


def load_vfdb_data():
    # vt = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/vt.npy')
    # vt = vt.reshape(-1, 2500)  # 186 2500
    # VT_test = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/VT_test.npy')

    # VT_test = np.load('E:\研究生\data\\vfdb\\VT_test.npy')
    VT_test = np.load('/home/jinjiahao/data/vfdb/VT_test.npy')
    VT_test = VT_test[:2617500].reshape(-1, 2500)  # 1047 2500

    # vfl = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/vfl.npy')
    # vfl = vfl.reshape(-1, 2500)  # 196, 2500
    # vfl_test = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/vfl_test.npy')

    # vfl_test = np.load('E:\研究生\data\\vfdb\\vfl_test.npy')
    vfl_test = np.load('/home/jinjiahao/data/vfdb/vfl_test.npy')
    vfl_test = vfl_test[:347500].reshape(-1, 2500)  # 139 2500

    # vfib = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/vfib.npy')
    # vfib = vfib.reshape(-1, 2500)  # 8  2500
    # vfib_test = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/vfib_test.npy')

    # vfib_test = np.load('E:\研究生\data\\vfdb\\vfib_test.npy')
    vfib_test = np.load('/home/jinjiahao/data/vfdb/vfib_test.npy')
    vfib_test = vfib_test[:312500].reshape(-1, 2500)  # 125  2500

    # normal = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/normal.npy')
    # normal = normal[:1052500].reshape(-1, 2500)  # 421  2500
    # normal_test = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/normal_test.npy')


    # normal_test = np.load('E:\研究生\data\\vfdb\\normal_test.npy')
    normal_test = np.load('/home/jinjiahao/data/vfdb/normal_test.npy')
    normal_test = normal_test[:8612500].reshape(-1, 2500)  # 3445  2500
    # normal_test = normal_test[:750000].reshape(-1, 2500)  # 300 2500

    # noisy = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/noisy.npy')
    # noisy = noisy[:360000].reshape(-1, 2500)  # 144 2500
    # noisy_test = np.load('/home/jinjiahao/data/室性心律失常检测/physionet.org/files/vfdb/noisy_test.npy')

    # noisy_test = np.load('E:\研究生\data\\vfdb\\noisy_test.npy')
    noisy_test = np.load('/home/jinjiahao/data/vfdb/noisy_test.npy')
    noisy_test = noisy_test[:3030000].reshape(-1, 2500)  # 1212  2500
    # noisy_test = noisy_test[:750000].reshape(-1, 2500)  # 300 2500

    # data = np.concatenate((vfl_test, vfib_test, VT_test, normal_test, noisy_test), axis=0)
    data_X_va = np.concatenate((vfl_test, vfib_test, VT_test), axis=0)
    data_X_normal = np.concatenate((normal_test, noisy_test), axis=0)
    # label1 = [1] * (vfl_test.shape[0] + vfib_test.shape[0] + VT_test.shape[0])
    # label2 = [0] * (normal_test.shape[0] + noisy_test.shape[0])
    # label = np.concatenate((label1, label2), axis=0)

    # data_X = data
    # data_Y = label

    data_len = len(data_X_normal)
    train_X = data_X_normal[0:data_len * 2 // 3]

    val_normal_X = data_X_normal[data_len * 2 // 3:data_len * 5 // 6]
    val_abnormal_X = data_X_va[0:len(data_X_va) // 2]

    test_normal_X = data_X_normal[data_len * 5 // 6:]
    test_abnormal_X = data_X_va[len(data_X_va) // 2:]

    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X
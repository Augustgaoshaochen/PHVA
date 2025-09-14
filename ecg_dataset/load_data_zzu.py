
import numpy as np
import pickle
def concat(a,b):

    if a is None or a.size == 0:

        return b
    elif b is None or b.size == 0:

        return a

    else:
        return np.concatenate((a,b), axis=0)

def get_ecg_class_data(start_index, end_index, ecg_class, dir):

    data_X = None

    for i in range(start_index, end_index):

        print('Read {}-th to {}-th {}:  {}'.format(start_index, end_index, ecg_class, i))

        with open('{}/YXDECG_{}_data_{}.pickle'.format(dir, i * 1000, ecg_class), 'rb') as handle1:

            data_single = pickle.load(handle1)  # (1000,12,10000)

            data_X = concat(data_X, data_single)

    # data_1lead = data_X[:, 1:2, :1000]    #PPG
    # data_7lead = np.delete(data_X[:,:,:1000], [1, 2, 3, 4, 5], 1)   #ECG

    #return data_1lead, data_7lead
    data=data_X[:,:,:1000]
    return data
def get_data_loader_ECG():

    LEAD = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

    LEAD_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    ECG_class_list = ['FangChan', 'FeiDa', 'GengSi', 'ZuZhi', 'GuoSu', 'QTc', 'ZhengChang']  # GuoSu只有一万

    dir_ALL_100 = '/home/huangxunhua/Data/ALL_100'
    dir_ALL_100_N = '/home/huangxunhua/Data/ALL_100_N'

    data_1lead_train = None
    data_7lead_train = None
    data=None
    label=[]

    for i in range(len(ECG_class_list)):

        if ECG_class_list[i] == 'GengSi':

            data_Gengsi = get_ecg_class_data(1, 15, ECG_class_list[i], dir_ALL_100)
            data=concat(data,data_Gengsi)
            label.extend(np.zeros(data_Gengsi.shape[0],dtype=int))

        elif ECG_class_list[i] == 'GuoSu':

            data_Guosu = get_ecg_class_data(1, 8, ECG_class_list[i], dir_ALL_100)
            data=concat(data,data_Guosu)
            label.extend(np.zeros(data_Guosu.shape[0], dtype=int))

        elif ECG_class_list[i] == 'FangChan':

            data_Fangchan = get_ecg_class_data(1, 11, ECG_class_list[i], dir_ALL_100)
            data = concat(data, data_Fangchan)
            label.extend(np.ones(data_Fangchan.shape[0], dtype=int))

        elif ECG_class_list[i] == 'FeiDa':

            data_Feida = get_ecg_class_data(1, 11, ECG_class_list[i], dir_ALL_100)
            data = concat(data, data_Feida)
            label.extend(np.zeros(data_Feida.shape[0], dtype=int))

        elif ECG_class_list[i] == 'ZuZhi':

            data_Zuzhi = get_ecg_class_data(1, 11, ECG_class_list[i], dir_ALL_100)
            data = concat(data, data_Zuzhi)
            label.extend(np.zeros(data_Zuzhi.shape[0], dtype=int))

        elif ECG_class_list[i] == 'QTc':

            data_QTc = get_ecg_class_data(1, 11, ECG_class_list[i], dir_ALL_100)
            data = concat(data, data_QTc)
            label.extend(np.zeros(data_QTc.shape[0], dtype=int))


        elif ECG_class_list[i] == 'ZhengChang':

            data_N = get_ecg_class_data(1, 11, ECG_class_list[i], dir_ALL_100)
            data = concat(data, data_N)
            label.extend(np.zeros(data_N.shape[0], dtype=int))


    #print(data.shape)
    #print(label.shape)

    return data,np.array(label)
def one_class_labeling(labels, normal_class:int,seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)


    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx
def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    return labels.astype("bool"), index

def load_zzu(opt):
    opt.normal_idx=0
    data_X, data_Y = get_data_loader_ECG()
    label_idxs = np.unique(data_Y)
    class_stat = {}
    opt.normal_idx=0
    for idx in label_idxs:
        class_stat[idx] = len(np.where(data_Y == idx)[0])

    if opt.normal_idx >= len(label_idxs):
        normal_idx = opt.normal_idx % len(label_idxs)
    else:
        normal_idx = opt.normal_idx
    labels_binary, idx_normal, idx_abnormal = one_class_labeling(data_Y, opt.normal_idx, opt.seed)
    data_N_X = data_X[idx_normal]
    data_N_Y = labels_binary[idx_normal]
    data_A_X = data_X[idx_abnormal]
    data_A_Y = np.array([True] * data_A_X.shape[0])

    # Split normal samples
    n_normal = data_N_X.shape[0]
    train_X = data_N_X[:(int(n_normal * 0.8)), ]  # train 0.6
    train_Y = data_N_Y[:(int(n_normal * 0.8)), ]

    val_N_X = data_N_X[int(n_normal * 0.8):int(n_normal * 0.9)]  # train 0.2
    val_N_Y = data_N_Y[int(n_normal * 0.8):int(n_normal * 0.9)]

    test_N_X = data_N_X[int(n_normal * 0.9):]  # train 0.2,0.1
    test_N_Y = data_N_Y[int(n_normal * 0.9):]

    data_A_X_len = data_A_X.shape[0]

    val_N_X_len = val_N_X.shape[0]
    test_N_X_len = test_N_X.shape[0]
    data_A_X_idx = list(range(data_A_X_len))
    val_A_X = data_A_X[data_A_X_idx[:val_N_X_len]]
    val_A_Y = data_A_Y[data_A_X_idx[:val_N_X_len]]
    test_A_X = data_A_X[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]
    test_A_Y = data_A_Y[data_A_X_idx[val_N_X_len:(val_N_X_len + test_N_X_len)]]

    val_X = np.concatenate((val_N_X, val_A_X))
    val_Y = np.concatenate((val_N_Y, val_A_Y))
    test_X = np.concatenate((test_N_X, test_A_X))
    test_Y = np.concatenate((test_N_Y, test_A_Y))

    val_label, val_idx = shuffle_label(val_Y, opt.seed)
    val_data = val_X[val_idx]
    val_label = val_label[val_idx]

    test_label, test_idx = shuffle_label(test_Y, opt.seed)
    test_data = test_X[test_idx]
    test_label = test_label[test_idx]
    return train_X, train_Y, val_data, val_label, test_data, test_label

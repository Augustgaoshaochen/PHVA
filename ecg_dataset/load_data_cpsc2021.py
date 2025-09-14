import glob
import os
import pickle

import numpy as np
from plot.plot_ECG import plot_new2,show,plot_new

# def read_pickle(file_name):
#     res = None
#     if os.path.exists(file_name):
#         with open(file_name, 'rb') as f:
#             res = pickle.load(f)
#     return res

def read_pickle(file_name):
    res = None
    if os.path.exists(file_name):
        try:
            with open(file_name, 'rb') as f:
                res = pickle.load(f)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Warning: Could not load pickle file {file_name}. Error: {e}")
    else:
        print(f"File does not exist: {file_name}")
    return res


def add_normal_patient(normal, root):
    all_normal_patient = []
    for patient_id in normal:
        normal_patient = []
        for i in range(30):
            res = read_pickle(root + "data_{}_{}_normal.pickle".format(patient_id, i))
            if res is not None:
                normal_patient.append(res)
        all_normal_patient.append(normal_patient)
    return all_normal_patient


def add_abnormal_patient(normal, root):
    all_normal_patient = []
    for patient_id in normal:
        normal_patient = []
        for i in range(30):
            res = read_pickle(root + "data_{}_{}_fangchan.pickle".format(patient_id, i))
            if res is not None:
                normal_patient.append(res)
        all_normal_patient.append(normal_patient)
    return all_normal_patient


def add_normal_abnormal_patient(normal, root):
    return add_normal_patient(normal, root), add_abnormal_patient(normal, root)

def List_2_Numpy(list):
    z_list = []  # 每个元素代表一个病人
    for i in list:
        z_list.append(np.concatenate(i, axis=0))
    return np.array(z_list, dtype=object)

def shuffle_label(labels, seed):
    index = [i for i in range(labels.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    return labels.astype("bool"), index

def plot_low(data,type):
    for i,AF_item in enumerate(data[0][0][:20]):
        import matplotlib.pyplot as plt
        for j,item in enumerate(AF_item):
            '''将数据等比例归一化到（-1，1）之间
            # min_val = np.min(item)
            # max_val = np.max(item)
            # scaled_data = (item - min_val) / (max_val - min_val) * 2
            # normalized_data = scaled_data - 1
            '''
            # chk_direcg='{}/{}/{}/{}'.format('/home/changhuihui/learn_project/COCA/plot/cpsc2021',type,'ECG',i)
            # if not os.path.exists(chk_direcg):
            #     os.makedirs(chk_direcg)
            # path="{}/{}".format(chk_direcg, j)
            # plot_new2(item, i, j,path)
            # plot_new(item,i,j)
            import matplotlib.pyplot as plt
            from scipy.signal import stft

            # Assuming you have imported numpy and have AF_item defined
            sampling_rate=100
            time = np.arange(0, len(item) / sampling_rate, 1 / sampling_rate)
            plt.figure(figsize=(15, 4), dpi=300)
            plt.plot(time,item, linewidth=2, color='green')  # 设置线条粗细为2
            # plt.legend(fontsize='large')
            plt.xlabel('Time (s)',fontsize=15)
            plt.ylabel('Values(mV)',fontsize=15)
            plt.xlim(0, 10)
            chk_direcg='{}/{}/{}/{}'.format('/home/changhuihui/learn_project/COCA/plot/cpsc2021',type,'ECG',i)
            if not os.path.exists(chk_direcg):
                os.makedirs(chk_direcg)
            plt.savefig("{}/{}".format(chk_direcg,j) + '.svg',
                                   bbox_inches='tight')
            # f, t, Zxx = stft(item, 100, nperseg=100)
            x_fft = np.fft.fft(item)  # 快速傅里叶变换，从时域转换为频域的算法，
            x_fft = np.abs(x_fft)  # 计算绝对值
            T=1/sampling_rate
            freqs = np.fft.fftfreq(len(time), T)
            # Plotthespectrogram
            # plt.figure(figsize=(10, 4))
            # plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='jet')
            plt.figure(figsize=(15, 4), dpi=300)
            plt.plot(freqs[1:], x_fft[1:], linewidth=2, color='red')
            plt.title('Frequency Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.ylim(0, 80)
            # plt.colorbar(label='Magnitude')
            # plt.ylim(0, 50)  # Limitthey-axisto50Hzforbettervisibilityoftheheartratefrequencies
            plt.tight_layout()
            chk_dir = '{}/{}/{}/{}'.format('/home/changhuihui/learn_project/COCA/plot/cpsc2021', type, 'spectrum', i)
            if not os.path.exists(chk_dir):
                os.makedirs(chk_dir)
            plt.savefig("{}/{}".format(chk_dir, j) + '.png')
import matplotlib.pyplot as plt
def plot_high(data,type):
    for i,AF_item in enumerate(data[0][0][:20]):

        for j,item in enumerate(AF_item):
            sampling_rate = 100
            time = np.arange(0, len(item) / sampling_rate, 1 / sampling_rate)
            chk_direcg='{}/{}/{}/{}'.format('/home/changhuihui/learn_project/COCA/plot/cpsc2021_high',type,'ECG',i)
            if not os.path.exists(chk_direcg):
                os.makedirs(chk_direcg)
            path="{}/{}".format(chk_direcg, j)
            plot_new2(item, i, j,path)
            x_fft = np.fft.fft(item)  # 快速傅里叶变换，从时域转换为频域的算法，
            x_fft = np.abs(x_fft)  # 计算绝对值
            T = 1 / sampling_rate
            freqs = np.fft.fftfreq(len(time), T)
            plt.figure(figsize=(15, 4), dpi=300)
            plt.plot(freqs[1:], x_fft[1:], linewidth=2, color='red')
            plt.title('Frequency Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.xlim(0, sampling_rate / 2)
            plt.tight_layout()
            chk_dir = '{}/{}/{}/{}'.format('/home/changhuihui/learn_project/COCA/plot/cpsc2021_high', type, 'spectrum', i)
            if not os.path.exists(chk_dir):
                os.makedirs(chk_dir)
            plt.savefig("{}/{}".format(chk_dir, j) + '.svg')

def plot_sample2(signal, label, dataname,index):
    LEAD = ["I", "II"]
    try:
        signal = signal[:, :]
        fig = plt.figure(figsize=(24, 6), dpi=300)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)
        idx = [1, 2]
        for i in range(len(signal)):
            plt.subplot(2, 1, idx[i])
            plt.plot(signal[i], color='cadetblue', label=str(i))
            plt.title(LEAD[i] + ": ", fontsize=20, color='blue', loc='left')
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=20)
            plt.ylabel('value (mV)', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

        fig.tight_layout()

        save_dir = "/home/changhuihui/learn_project/COCA/plot_frequency/{}/{}".format(dataname,label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig('{}/{}.pdf'.format(save_dir, str(index)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.png'.format(save_dir, str(index)), transparent=False, bbox_inches='tight')
        plt.savefig('{}/{}.svg'.format(save_dir, str(index)), transparent=False, bbox_inches='tight')

        # plt.show()
        plt.close()
        return True
    except Exception as e:
        print(e)
        return False
def get_data(root):
    # normal_path = glob.glob("/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/*_normal.pickle")
    # fangchan_path = glob.glob("/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/*_fangchan.pickle")

    normal_path = glob.glob("E:\研究生\data\CPSC\\*_normal.pickle")
    fangchan_path = glob.glob("E:\研究生\data\CPSC\\*_fangchan.pickle")
    # 获得患者id
    normal_patient = sorted([int(x.split("/")[-1].split("_")[1]) for x in normal_path])
    fangchan_patient = sorted([int(x.split("/")[-1].split("_")[1]) for x in fangchan_path])

    normal_patient = list(set(normal_patient))
    fangchan_patient = list(set(fangchan_patient))

    normal_abnormal = np.intersect1d(normal_patient, fangchan_patient)  # 23

    normal = np.array([i for i in normal_patient if i not in normal_abnormal])  # 51
    abnormal = np.array([i for i in fangchan_patient if i not in normal_abnormal])  # 31

    # 非房颤
    all_normal_patient = add_normal_patient(normal, root)
    # 房颤
    all_abnormal_patient = add_abnormal_patient(abnormal, root)
    plot_ecg=True
    if plot_ecg==True:
        for index, itemcpsc in enumerate(all_normal_patient[0][0][:20]):
            plot_sample2(itemcpsc, 'NAF', 'CPSC', index)
        for index, itemcpsc in enumerate(all_abnormal_patient[0][0][:20]):
            plot_sample2(itemcpsc, 'AF', 'CPSC', index)
        # plot_low(all_abnormal_patient,'AF')
        # plot_low(all_normal_patient, 'NAF')
        # plot_high(all_abnormal_patient,'AF')
        # plot_high(all_normal_patient, 'NAF')

    all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = add_normal_abnormal_patient(normal_abnormal, root)

    all_normal_patient = List_2_Numpy(all_normal_patient)  # (51,) 96600
    all_abnormal_patient = List_2_Numpy(all_abnormal_patient)  # (31,) 52682
    all_normal_abnormal_patient_N = List_2_Numpy(all_normal_abnormal_patient_N)  # (23,) 8477
    all_normal_abnormal_patient_A = List_2_Numpy(all_normal_abnormal_patient_A)  # (23,) 6105
    # 163864

    return all_normal_patient, all_abnormal_patient, all_normal_abnormal_patient_N, all_normal_abnormal_patient_A


def load_data_cpsc2021(root, len_num=1000, seed=1024):
    all_normal_patient, all_abnormal_patient, \
    all_normal_abnormal_patient_N, all_normal_abnormal_patient_A = get_data(root)
    # 51 normal
    # 23 abnormal
    train_data = []

    np.random.seed(seed)
    np.random.shuffle(all_normal_patient)

    for i in all_normal_patient[:40]:  # 41
        np.random.seed(seed)
        np.random.shuffle(i)
        train_data.append(i[:600, :, :len_num])

    test_normal_data = []
    test_abnormal_data = []
    val_normal_data = []
    val_abnormal_data = []
    test_all_normal_patient = all_normal_patient[40:]
    # 随机病人
    np.random.seed(seed)
    np.random.shuffle(all_abnormal_patient)
    np.random.seed(seed)
    np.random.shuffle(all_normal_abnormal_patient_A)
    np.random.seed(seed)
    np.random.shuffle(all_normal_abnormal_patient_N)
    np.random.seed(seed)
    np.random.shuffle(test_all_normal_patient)
    # val
    for i in all_abnormal_patient[:15]:  # 31
        # 随机单个病人时间段
        np.random.seed(seed)
        np.random.shuffle(i)
        val_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_A[:11]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        val_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in test_all_normal_patient[:5]:  # 11
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[:400, :, :len_num])  # 400
    # k = 0
    for i in all_normal_abnormal_patient_N[:11]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        val_normal_data.append(i[:100, :, :len_num])
    # test
    for i in all_abnormal_patient[15:]:  # 31
        np.random.seed(seed)
        np.random.shuffle(i)
        test_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in all_normal_abnormal_patient_A[11:]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        test_abnormal_data.append(i[:100, :, :len_num])  # 100

    for i in test_all_normal_patient[5:]:  # 11
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[:400, :, :len_num])  # 400

    for i in all_normal_abnormal_patient_N[11:]:  # 23
        np.random.seed(seed)
        np.random.shuffle(i)
        test_normal_data.append(i[:100, :, :len_num])

    train_data = np.concatenate(train_data, axis=0)
    test_abnormal_data = np.concatenate(test_abnormal_data, axis=0)
    test_normal_data = np.concatenate(test_normal_data, axis=0)
    val_normal_data = np.concatenate(val_normal_data, axis=0)
    val_abnormal_data = np.concatenate(val_abnormal_data, axis=0)

    return train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data

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

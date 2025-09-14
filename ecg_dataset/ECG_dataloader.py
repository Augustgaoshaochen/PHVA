import neurokit2 as nk
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from VF_fake.ecg_dataset import transform
# from COCA.ecg_dataset.HRVTool import getP,getfakeP
# from ecg_dataset.getHRV import get_nni_24H_1, get_grv_list
from VF_fake.ecg_dataset.load_data import load_cpsc, load_icentia11k, load_ptbxl, load_chaosuan, load_vfdb, load_mitbih, \
    load_cudb, load_noise
from VF_fake.ecg_dataset.load_data_zzu import load_zzu
from scipy.signal import stft
from data_utils import down_sample
from VF_fake.ecg_dataset.time_series_augmentation import time_warp,getP,getfakeP,jittering,mask_waves_fake
import pywt


def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2 * (seq - np.min(seq)) / (np.max(seq) - np.min(seq)) - 1


def getPercent(data_x, data_y, percent, seed):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=percent, random_state=seed)
    return train_x, test_x, train_y, test_y


def one_class_labeling(labels, normal_class: int, seed):
    normal_idx = np.where(labels == normal_class)[0]
    abnormal_idx = np.where(labels != normal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)

    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


def one_class_labeling_sz(labels, abnormal_class: int, seed):
    normal_idx = np.where(labels != abnormal_class)[0]
    abnormal_idx = np.where(labels == abnormal_class)[0]

    labels[normal_idx] = 0
    labels[abnormal_idx] = 1
    np.random.seed(seed)
    np.random.shuffle(normal_idx)
    np.random.shuffle(abnormal_idx)

    return labels.astype("bool"), normal_idx, abnormal_idx


class RawDataset(data.Dataset):
    def __init__(self, X, Y):
        """
        """

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

        self.Z1 = torch.Tensor(X)
        self.Z2 = torch.Tensor(X)
        self.Z3 = torch.Tensor(X)
        self.Z4 = torch.Tensor(X)
        self.Z5 = torch.Tensor(X)
        self.Z6 = torch.Tensor(X)

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.Y[index], self.Z1[index], self.Z2[index], self.Z3[index], self.Z4[index], self.Z5[index], self.Z6[index]

    def __len__(self):
        return self.X.size(0)


def detect_r_peaks(signal, fs):
    """
    简单的 R 波检测函数。

    :param signal: 单条心电信号
    :param fs: 采样频率
    :return: R 波索引列表
    """
    derivative = np.diff(signal)  # 一阶导数
    threshold = 0.6 * np.max(np.abs(derivative))  # 设置阈值
    r_peaks = np.where(derivative > threshold)[0]
    # 保留距离大于0.2秒的峰值（去除过近的噪声峰）
    r_peaks = r_peaks[np.diff(np.insert(r_peaks, 0, 0)) > int(0.2 * fs)]
    return r_peaks


def zero_r_wave_region(ecg_data, fs=250, zero_offset=50):
    """
    检测 R 波并将其前后共 50 个数据点置零。

    :param ecg_data: 心电数据，形状为 (3104, 1, 2500)
    :param fs: 采样频率
    :param zero_offset: 前后置零的点数
    :return: 处理后的心电数据
    """
    ecg_data = np.squeeze(ecg_data)  # 移除单维，形状变为 (3104, 2500)
    processed_ecg = np.copy(ecg_data)  # 创建副本以进行修改
    zero_range = zero_offset // 2  # 前后各置零的点数

    for i in range(ecg_data.shape[0]):
        signal = ecg_data[i, :]

        # 检测 R 波
        r_peaks = detect_r_peaks(signal, fs)

        # 将 R 波前后共 50 点置零
        for r_peak in r_peaks:
            start_idx = max(0, r_peak - zero_range)
            end_idx = min(len(signal), r_peak + zero_range)
            processed_ecg[i, start_idx:end_idx] = 0

    return processed_ecg


def detect_p_waves(signal, r_peaks, fs):
    """
    P 波检测函数。

    :param signal: 单条心电信号
    :param r_peaks: R 波索引列表
    :param fs: 采样频率
    :return: P 波索引列表
    """
    p_peaks = []
    for r_peak in r_peaks:
        p_search_start = max(0, r_peak - int(0.2 * fs))  # R 波前 200ms
        p_search_end = max(0, r_peak - int(0.05 * fs))  # R 波前 50ms
        p_peak = np.argmax(signal[p_search_start:p_search_end]) + p_search_start
        p_peaks.append(p_peak)
    return p_peaks


def zero_p_wave_region(ecg_data, fs=250, zero_offset=50):
    """
    检测 P 波并将其前后共 50 个数据点置零。

    :param ecg_data: 心电数据，形状为 (3104, 1, 2500)
    :param fs: 采样频率
            :param zero_offset: 前后置零的点数
            :return: 处理后的心电数据
            """
    ecg_data = np.squeeze(ecg_data)  # 移除单维，形状变为 (3104, 2500)
    processed_ecg = np.copy(ecg_data)  # 创建副本以进行修改
    zero_range = zero_offset // 2  # 前后各置零的点数

    for i in range(ecg_data.shape[0]):
         signal = ecg_data[i, :]

         # 检测 R 波
         r_peaks = detect_r_peaks(signal, fs)

         # 检测 P 波
         p_peaks = detect_p_waves(signal, r_peaks, fs)

         # 将 P 波前后共 50 点置零
         for p_peak in p_peaks:
           start_idx = max(0, p_peak - zero_range)
           end_idx = min(len(signal), p_peak + zero_range)
           processed_ecg[i, start_idx:end_idx] = 0

    return processed_ecg

def zero_q_wave(ecg_data, fs=100, q_offset=50):
    """
    检测 Q 波并将前后共 50 个数据点置零。

    :param ecg_data: 形状为 (3104, 1, 2500) 的心电数据
    :param fs: 采样频率，默认为 250 Hz
    :param q_offset: Q 波前后置零的点数（默认共 50 个数据点）
    :return: 处理后的心电数据
    """
    mask_data = []
    # 20000 , 2, 1000

    ecg_data = np.squeeze(ecg_data)  # 移除形状中的单维 (1)，形状变为 (3104, 2500)
    q_range = q_offset // 2  # 每侧置零的点数
    processed_ecg = np.copy(ecg_data)  # 创建副本用于处理

    for beat_idx in range(ecg_data.shape[0]):
        signal = ecg_data[beat_idx, :]

        # 简单 Q 波检测（寻找振幅最小值的点）
        # 这里假设 Q 波的振幅为 R 波前最小的值
        q_peaks = []
        for r_peak in detect_r_peaks(signal, fs):  # 检测 R 波
            q_search_start = max(0, r_peak - int(0.05 * fs))  # R 波前 50ms
            q_search_end = r_peak  # 到 R 波的位置
            q_peak = np.argmin(signal[q_search_start:q_search_end]) + q_search_start  # 最小值点
            q_peaks.append(q_peak)

        # 将 Q 波前后 50 个点置零
        for q_peak in q_peaks:
            start_idx = max(0, q_peak - q_range)
            end_idx = min(signal.shape[0], q_peak + q_range)
            processed_ecg[beat_idx, start_idx:end_idx] = 0


        return processed_ecg
class mask_waves(data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.Tensor(X)
        self.Z = torch.Tensor(X)
        self.Y = torch.Tensor(Y)

        processed_data1 = []
        processed_data2 = []
        processed_data3 = []

        for dim in range(X.shape[1]):
            single_lead_data = X[:, dim, :][:, np.newaxis, :]
            # mask Q R P 原始
            # mask Q
            x_q = zero_q_wave(single_lead_data)
            x_q = np.expand_dims(x_q, axis=1)
            processed_data1.append(x_q)
        x_q = np.concatenate(processed_data1, axis=1)

        # mask R
        for dim in range(X.shape[1]):
            single_lead_data = X[:, dim, :][:, np.newaxis, :]
            x_r = zero_r_wave_region(single_lead_data)
            x_r = np.expand_dims(x_r, axis=1)
            processed_data2.append(x_r)
        x_r = np.concatenate(processed_data2, axis=1)

        # mask P
        for dim in range(X.shape[1]):
            single_lead_data = X[:, dim, :][:, np.newaxis, :]
            x_p = zero_p_wave_region(single_lead_data)
            x_p = np.expand_dims(x_p, axis=1)
            processed_data3.append(x_p)
        x_p = np.concatenate(processed_data3, axis=1)

        # 原始
        nomask = X

        coeffs1 = pywt.wavedec(x_p, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        x_p_f = coeffs1[0]
        coeffs2 = pywt.wavedec(x_q, 'db1', level=1)
        x_q_f = coeffs2[0]
        coeffs3 = pywt.wavedec(x_r, 'db1', level=1)
        x_r_f = coeffs3[0]

        self.x = torch.Tensor(X)
        self.y = torch.Tensor(Y)
        self.x_p_f = torch.Tensor(x_p_f)
        self.x_p = torch.Tensor(x_p)
        self.x_q = torch.Tensor(x_q)
        self.x_q_f = torch.Tensor(x_q_f)
        self.x_r = torch.Tensor(x_r)
        self.x_r_f = torch.Tensor(x_r_f)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_p[index], self.x_q[index], self.x_r[index], self.x_p_f[index], self.x_q_f[index], self.x_r_f[index]

    def __len__(self):
        return self.x.shape[0]






class LorentzDataset(data.Dataset):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        np.zeros_like(x)

        x_lr = []
        x_sdnn = []
        x_rmssd = []

        x_lr_1 = []
        x_lr_2 = []
        x_lr_3 = []

        x_sdnn_1 = []
        x_rmssd_1 = []

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                lr, sdnn, rmssd = get_nni_24H_1(x[i][j], 100)
                lr_1 = lr[0]
                lr_2 = lr[1]
                x_lr_1 = x_lr_1 + lr_1
                x_lr_2 = x_lr_2 + lr_2

                # draw_LR(x_lr_1, x_lr_2)

                x_sdnn_1.append(sdnn)
                x_rmssd_1.append(rmssd)

            x_lr_1 += [0 for i in range(np.size(x, 2) - len(x_lr_1))]
            x_lr_2 += [0 for i in range(np.size(x, 2) - len(x_lr_2))]

            x_lr_3.append(x_lr_1)
            x_lr_3.append(x_lr_2)

            x_lr.append(x_lr_3)
            x_sdnn.append(x_sdnn_1)
            x_rmssd.append(x_rmssd_1)

            x_lr_1 = []
            x_lr_2 = []
            x_lr_3 = []
            x_sdnn_1 = []
            x_rmssd_1 = []

        x_lr = np.asarray(x_lr, dtype=np.float32)
        x_sdnn = np.asarray(x_sdnn, dtype=np.float32)
        x_rmssd = np.asarray(x_rmssd, dtype=np.float32)
        x_sdnn = np.mean(x_sdnn, 1)
        x_rmssd = np.mean(x_rmssd, 1)
        x_sd = np.asarray(list(zip(x_sdnn, x_rmssd)), dtype=np.float32)
        x_sd = np.expand_dims(x_sd, 2).repeat(1000, axis=2)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_lr = torch.Tensor(x_lr)
        self.x_sdnn = torch.Tensor(x_sdnn)
        self.x_rmssd = torch.Tensor(x_rmssd)
        self.x_sd = torch.Tensor(x_sd)

    def __getitem__(self, index):

        return self.x[index], self.x_lr[index], self.x_sd[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class HrvDataset(data.Dataset):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        np.zeros_like(x)
        if len(x[1]) == 1:
            hrv = [[]]
        elif len(x[1]) == 2:
            hrv = [[], []]

        x_hrv = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                hrv_list = get_grv_list(x[i][j], 100)
                hrv[j] = hrv_list
            x_hrv.append(hrv)
        x_hrv = np.asarray(x_hrv, dtype=np.float32)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_hrv = torch.Tensor(x_hrv)

    def __getitem__(self, index):

        return self.x[index], self.x_hrv[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class FftDatasetwavetrain(data.Dataset):
    def __init__(self, x, y,train,opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_fft = np.fft.fft(x)  # 快速傅里叶变换，从时域转换为频域的算法，
        spectrogram_instance = np.abs(x_fft)

        # coeffs = pywt.wavedec(x, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        # spectrogram_instance = coeffs[1]

        x_mask = getP(x, opt.dataset, train, opt.seed, opt.model)

        x_fft_mask = np.fft.fft(x_mask)  # 快速傅里叶变换，从时域转换为频域的算法，
        spectrogram_instance_mask = np.abs(x_fft_mask)

        # 生成伪异常数据
        x_fake = time_warp(x_mask[:, :, ].transpose(0, 2, 1), opt.dataset, opt.seed, train, opt.model, sigma=0.1,
                           knot=4)
        x_fake = x_fake.transpose(0, 2, 1)

        x_fft_fake = np.fft.fft(x_fake)  # 快速傅里叶变换，从时域转换为频域的算法，
        spectrogram_instance_fake = np.abs(x_fft_fake)

        # coeffs_fake = pywt.wavedec(x_fake, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        # spectrogram_instance_fake = coeffs_fake[1]

        # 将伪异常数据进行掩码
        x_fake_mask = getfakeP(x_fake, opt.dataset, train, opt.seed, opt.model)

        x_fft_fake_mask = np.fft.fft(x_fake_mask)  # 快速傅里叶变换，从时域转换为频域的算法，
        spectrogram_instance_fake_mask = np.abs(x_fft_fake_mask)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fake = torch.Tensor(x_fake)
        self.x_fake_mask = torch.Tensor(x_fake_mask)
        self.x_fft = torch.Tensor(spectrogram_instance)
        self.x_fft_fake = torch.Tensor(spectrogram_instance_fake)
        self.x_fft_mask = torch.Tensor(spectrogram_instance_mask)
        self.x_fft_fake_mask = torch.Tensor(spectrogram_instance_fake_mask)

    def __getitem__(self, index):

        return self.x[index], self.x_mask[index], self.y[index], self.x_fake[index], self.x_fake_mask[index],self.x_fft[index],self.x_fft_mask[index],self.x_fft_fake[index],self.x_fft_fake_mask[index]
    def __len__(self):
        return self.x.shape[0]
class FftDatasetwavetest(data.Dataset):
    def __init__(self, x, y,train,opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # coefficients, frequencies = pywt.cwt(x, scales=np.arange(1, 128), wavelet='morl')
        # f, t, Zxx = stft(x, fs=100, window='hann', nperseg=125)
        # spectrogram_instance = np.abs(Zxx)  # (12, 63, 78)

        x_fft = np.fft.fft(x)  # 快速傅里叶变换，从时域转换为频域的算法，
        spectrogram_instance = np.abs(x_fft)
        # coeffs_fake = pywt.wavedec(x, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        # spectrogram_instance = coeffs_fake[1]

        x_mask = getP(x, opt.dataset, train, opt.seed, opt.model)
        x_fft_mask = np.fft.fft(x_mask)  # 快速傅里叶变换，从时域转换为频域的算法，
        spectrogram_instance_mask = np.abs(x_fft_mask)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(spectrogram_instance)
        self.x_fft_mask = torch.Tensor(spectrogram_instance_mask)

    def __getitem__(self, index):

        return self.x[index], self.x_mask[index], self.y[index], self.x[index], self.x_mask[index],self.x_fft[index],self.x_fft_mask[index],self.x_fft[index],self.x_fft_mask[index]

    def __len__(self):
        return self.x.shape[0]
class FftDataset(data.Dataset):
    def __init__(self, x, y):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_fft = np.fft.fft(x)#快速傅里叶变换，从时域转换为频域的算法，
        x_fft = np.abs(x_fft)#计算绝对值
        # import numpy as np
        # import matplotlib.pyplot as plt
        # fs = 100
        # freqs = np.fft.fftfreq(len(x_fft[0][2]), 1 / fs)
        # plt.subplot(2, 1, 2)
        # plt.plot(freqs, np.abs(x_fft[0][2]), color='red', linewidth=0.5)
        # plt.title('Frequency Domain Signal')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_fft = torch.Tensor(x_fft)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class MaskDataset_train2(data.Dataset):
    def __init__(self, x, y,train, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)


        x_mask = getP(x, opt.dataset,train,opt.seed,opt.model)

        #加噪声
        x_noise=x_mask.copy()
        for i, pat in enumerate(x_noise):
            for dim in range(x_noise.shape[1]):
                x_noise[i, dim, :] = jittering(pat[dim,:], mean=0, var=0.001)
        # 加噪声，加NN间期不齐特征
        x_fake = time_warp(x_mask[:,:,].transpose(0, 2, 1), opt.dataset,opt.seed,train,opt.model,sigma=0.1, knot=4)
        x_fake= x_fake.transpose(0, 2, 1)
        # 将伪异常数据进行掩码
        # x_fake_mask = getfakeP(x_fake, opt.dataset,train, opt.seed,opt.model)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fake = torch.Tensor(x_fake)
        self.x_noise = torch.Tensor(x_noise)

    def __getitem__(self, index):
        return self.x[index], self.x_mask[index], self.y[index],self.x_noise[index],self.x_fake[index]

    def __len__(self):
        return self.x.shape[0]
class MaskDataset_test2(data.Dataset):
    def __init__(self, x, y, test,opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        #得到掩码数据
        # x_mask = getP(x,opt.dataset,test,opt.seed,opt.model)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        # self.x_mask = torch.Tensor(x_mask)

    def __getitem__(self, index):
        return self.x[index], self.x[index], self.y[index],self.x[index], self.x[index]

    def __len__(self):
        return self.x.shape[0]

LEAD = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
import numpy as np
import matplotlib.pyplot as plt
def plot_sample12(signal, save_dir=None, datename=None):
    try:
        # Plot
        signal = signal[:, :]
        fig = plt.figure(figsize=(30, 12), dpi=100)
        x = np.arange(0, 1000, 100)
        x_labels = np.arange(0, 10)

        idx = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12]
        for i in range(len(signal)):
            plt.subplot(6, 2, idx[i])
            plt.plot(signal[i], color='green', label=str(i))
            # plt.title(LEAD[i] + ": ", fontsize=16)
            plt.title(LEAD[i], fontsize=16)
            plt.xticks(x, x_labels)
            plt.xlabel('time (s)', fontsize=16)
            plt.ylabel('value (mV)', fontsize=16)
        fig.tight_layout()
        plt.savefig("{}/sample_rate_{}".format(save_dir, datename) + '.png', bbox_inches='tight')
        # plt.show()
        plt.close()
        return True

    except Exception as e:
        print(e)
        return False
class fakeDataset_train(data.Dataset):
    def __init__(self, x, y,train, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # x_mask = getP(x, opt.dataset,train,opt.seed,opt.model)
        # 生成伪异常数据
        # x_fake = time_warp(x_mask[:,:,].transpose(0, 2, 1), opt.dataset,opt.seed,train,opt.model,sigma=0.1, knot=4)
        x_fake = np.zeros_like(x)
        for i,item in enumerate(x):
            x_RR = mask_waves_fake(i,item)
            # dim=x_RR.shape[0]
            # if dim==12:
            #     select_dim=[1,2,3,4,5,6]
            #     for lie in select_dim:
            #         x_RR[lie,:] = jittering(x_RR[lie,:], mean=0, var=0.001)
            # else:
            #     for lie in range(dim):
            #         x_RR[lie,:] = jittering(x_RR[lie,:], mean=0, var=0.001)
            x_fake[i,:,:]=x_RR
            # plot_sample12(x_fake[i,:,:], '/home/changhuihui/learn_project/COCA/ecg_dataset/plot_method1', 'fake_{}'.format(i))
        # x_fake= x_fake.transpose(0, 2, 1)
        # 将伪异常数据进行掩码
        # x_fake_mask = getfakeP(x_fake, opt.dataset,train, opt.seed,opt.model)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        # self.x_mask = torch.Tensor(x_mask)
        self.x_fake = torch.Tensor(x_fake)
        # self.x_fake_mask = torch.Tensor(x_fake_mask)

    def __getitem__(self, index):
        return self.x[index],self.y[index],self.x_fake[index]

    def __len__(self):
        return self.x.shape[0]
class fakeDataset_test(data.Dataset):
    def __init__(self, x, y, test,opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        #得到掩码数据
        # x_mask = getP(x,opt.dataset,test,opt.seed,opt.model)
        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        # self.x_mask = torch.Tensor(x_mask)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x[index]

    def __len__(self):
        return self.x.shape[0]
class MaskDataset_train(data.Dataset):
    def __init__(self, x, y,train, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)


        x_mask = getP(x, opt.dataset,train,opt.seed,opt.model)

        # 生成伪异常数据
        # x_fake = time_warp(x_mask[:,:,].transpose(0, 2, 1), opt.dataset,opt.seed,train,opt.model,sigma=0.1, knot=4)

        x_fake = np.zeros_like(x)
        for i, item in enumerate(x):
            x_RR = mask_waves_fake(i, item)

            # dim=x_RR.shape[0]
            # if dim==12:
            #     select_dim=[1,2,3,4,5,6]
            #     for lie in select_dim:
            #         x_RR[lie,:] = jittering(x_RR[lie,:], mean=0, var=0.001)
            # else:
            #     for lie in range(dim):
            #         x_RR[lie,:] = jittering(x_RR[lie,:], mean=0, var=0.001)
            x_fake[i, :, :] = x_RR
        np.save(("/home/changhuihui/learn_project/COCA/augmentation_dataset/new_{}_{}fake_".format(opt.model,
                                                                                               opt.dataset) + train + str(
            opt.seed) + ".npy"), np.array(x_fake))
        # x_fake= x_fake.transpose(0, 2, 1)
        # 将伪异常数据进行掩码
        x_fake_mask = getfakeP(x_fake, opt.dataset,train, opt.seed,opt.model)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fake = torch.Tensor(x_fake)
        self.x_fake_mask = torch.Tensor(x_fake_mask)

    def __getitem__(self, index):
        return self.x[index], self.x_mask[index], self.y[index],self.x_fake[index],self.x_fake_mask[index]

    def __len__(self):
        return self.x.shape[0]
class MaskDataset_test(data.Dataset):
    def __init__(self, x, y, test,opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        #得到掩码数据
        x_mask = getP(x,opt.dataset,test,opt.seed,opt.model)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)

    def __getitem__(self, index):
        return self.x[index], self.x_mask[index], self.y[index],self.x[index], self.x_mask[index]

    def __len__(self):
        return self.x.shape[0]


class ChaosuankDataset(data.Dataset):
    def __init__(self, x, y, x_a, opt):
        x = np.asarray(x, dtype=np.float32)
        x_a = np.asarray(x_a, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_a)

    def __getitem__(self, index):
        return self.x[index], self.x_mask[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDataset(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if opt.isMasked:
            if opt.dataset == 'cpsc2021':
                x_mask = np.load('cpsc2021_{}.npy'.format(opt.seed))
            elif opt.dataset == 'icentia11k':
                x_mask = np.load('icentia11k_{}.npy'.format(opt.seed))
                x_mask = np.expand_dims(x_mask, axis=1)
            elif opt.dataset == 'ptbxl':
                x_mask = np.load('ptbxl_{}.npy'.format(opt.seed))
        else:
            x_mask = getP(x, opt.dataset, opt.seed)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class FMDSDataset(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if opt.isMasked:
            if opt.dataset == 'cpsc2021':
                x_mask = np.load('cpsc2021_{}.npy'.format(opt.seed))
            elif opt.dataset == 'icentia11k':
                x_mask = np.load('icentia11k_{}.npy'.format(opt.seed))
                x_mask = np.expand_dims(x_mask, axis=1)
            elif opt.dataset == 'ptbxl':
                x_mask = np.load('ptbxl_{}.npy'.format(opt.seed))
        else:
            x_mask = getP(x, opt.dataset, opt.seed)

        oncatenated = np.concatenate((x_mask, x_mask), axis=2)
        for i in range(oncatenated.shape[0]):
            for j in range(oncatenated.shape[1]):
                x_mask[i][j] = down_sample(oncatenated[i][j], 2000, 1500)[:1000]

        for i in range(x_mask.shape[0]):
            input = np.expand_dims(x_mask[i], 0)
            input = input.transpose(0, 2, 1)
            # input = input.permute(0, 2, 1)
            input_w = time_warp(input)
            input_w = input_w.transpose(0, 2, 1)
            input_w = np.squeeze(input_w)
            x_mask[i] = input_w

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        x_fft_m = np.fft.fft(x_mask)
        x_fft_m = np.abs(x_fft_m)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x_mask)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft_m)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class FMDSNODataset(data.Dataset):
    def __init__(self, x, y, opt):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        x_fft = np.fft.fft(x)
        x_fft = np.abs(x_fft)

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_mask = torch.Tensor(x)
        self.x_fft = torch.Tensor(x_fft)
        self.x_fft_m = torch.Tensor(x_fft)

    def __getitem__(self, index):
        return self.x[index], self.x_fft[index], self.x_mask[index], self.x_fft_m[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

def frequency_modulate(time, signal, frequency_base, modulation_index):
    # frequency_base 是信号的基础频率
    # modulation_index 是调频指数，它决定了频率的变化程度

    # 生成一个频率随信号变化的频率信号
    modulated_frequency = frequency_base + modulation_index * signal

    # 产生FM调制信号
    modulated_signal = np.cos(2 * np.pi * modulated_frequency * time)
    return modulated_signal


# from ecg_dataset.draw import TransformDataset_SelectChannel


class TransformDataset_SelectChannel:
    def __init__(self, x, y, snr, fs, amplitude, p, min_winsize, is_selectwin, line_num):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        straight = transform.ChannelStraight_1D(p=p, is_selectwin=is_selectwin)
        gussian = transform.Gussian(snr=snr, p=p, is_selectwin=is_selectwin)
        gamma = transform.Gamma(snr=snr, p=p, is_selectwin=is_selectwin)
        rayleign = transform.Rayleign(snr=snr, p=p, is_selectwin=is_selectwin)
        exponential = transform.Exponential(snr=snr, p=p, is_selectwin=is_selectwin)
        poisson = transform.Poisson(snr=snr, p=p, is_selectwin=is_selectwin)
        uniform = transform.Uniform(snr=snr, p=p, is_selectwin=is_selectwin)
        am = transform.Am(fs=fs, amplitude=amplitude, p=p, is_selectwin=is_selectwin)
        fm = transform.Fm(fs=fs, amplitude=amplitude, p=p, is_selectwin=is_selectwin)
        bw = transform.Bw(fs=fs, amplitude=amplitude, p=p, is_selectwin=is_selectwin)

        transforms_list = {
            'gussian': gussian,
            'gamma': gamma,
            'rayleign': rayleign,
            'exponential': exponential,
            'poisson': poisson,
            'uniform': uniform,
            'fm': fm,
            'am': am,
            'bw': bw,
            'line': straight
        }

        trans = transform.Compose_Select(transforms_list.values(), min_winsize)
        x1 = x.reshape(-1, 2500)
        noisy = trans(x1)
        noisy = noisy.reshape(-1, 1, 2500)

        time = np.linspace(0, 10, 2500)  # 时间范围从0到1，共2500个数据点
        modulated_signal = frequency_modulate(time, x, 2, 0.1)
        # 小波变换
        coeffs = pywt.wavedec(x, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
        x_wavelet = coeffs[0]
        coeffs = pywt.wavedec(noisy, 'db1', level=1)
        n_wavelet = coeffs[0]
        coeffs = pywt.wavedec(modulated_signal, 'db1', level=1)
        v_wavelet = coeffs[0]

        self.x = torch.Tensor(x)
        self.y = torch.Tensor(y)
        self.x_wavelet = torch.Tensor(x_wavelet)
        self.x_noisy = torch.Tensor(noisy)
        self.n_wavelet = torch.Tensor(n_wavelet)
        self.x_vf = torch.Tensor(modulated_signal)
        self.v_wavelet = torch.Tensor(v_wavelet)

    def __getitem__(self, index):
        return self.x[index], self.x_wavelet[index], self.y[index], self.x_noisy[index], self.n_wavelet[index], \
               self.x_vf[index], self.v_wavelet[index]

    def __len__(self):
        return self.x.shape[0]

def get_dataloader(opt):
    global train_dataset, val_dataset, test_dataset

    if opt.dataset == 'cpsc2021':
        train_data, train_label, val_data, val_label, test_data, test_label = load_cpsc(opt)
    elif opt.dataset == 'icentia11k':
        train_data, train_label, val_data, val_label, test_data, test_label = load_icentia11k(opt)
    elif opt.dataset == 'ptbxl':
        train_data, train_label, val_data, val_label, test_data, test_label = load_ptbxl(opt)
    elif opt.dataset == 'chaosuan':
        train_data, train_label, val_data, val_label, test_data, test_label, train_data_abnormal = load_chaosuan(opt)
    elif opt.dataset == 'ZZU':
        train_data, train_label, val_data, val_label, test_data, test_label = load_zzu(opt)
    elif opt.dataset == 'vfdb':
        train_data, train_label, val_data, val_label, test_data, test_label = load_vfdb(opt)
    elif opt.dataset == 'mitbih':
        train_data, train_label, val_data, val_label, test_data, test_label = load_mitbih(opt)
    elif opt.dataset == 'cudb':
        train_data, train_label, val_data, val_label, test_data, test_label = load_cudb(opt)
    elif opt.dataset == 'BUTQDB':
        train_data, train_label, val_data, val_label, test_data, test_label = load_noise(opt)
        train_data = np.expand_dims(train_data, axis=1)
        val_data = np.expand_dims(val_data, axis=1)
        test_data = np.expand_dims(test_data, axis=1)

    print("[INFO] Train: normal={}".format(train_data.shape), )
    print("[INFO] Val: normal={}".format(val_data.shape), )
    print("[INFO] Test: normal={}".format(test_data.shape), )

    if opt.is_all_data:
        train_X = train_data
        train_Y = train_label
        val_X = val_data
        val_Y = val_label
        test_X = test_data
        test_Y = test_label
    else:
        train_X = train_data[:5000, :, :]
        train_Y = train_label[:5000]
        val_X = val_data[:2000, :, :]
        val_Y = val_label[:2000]
        test_X = test_data[:2000, :, :]
        test_Y = test_label[:2000]

    # Wavelet transform
    X_length = train_X.shape[-1]

    signal_length = [0]

    if opt.augmentation == "maskp":
        train_dataset = MaskDataset_train(train_X, train_Y,'train',opt)
        val_dataset = MaskDataset_test(val_X, val_Y,'val', opt)
        test_dataset = MaskDataset_test(test_X, test_Y,'test',opt)
    elif opt.augmentation == "fake1":
        train_dataset = fakeDataset_train(train_X, train_Y,'train',opt)
        val_dataset = fakeDataset_test(val_X, val_Y,'val', opt)
        test_dataset = fakeDataset_test(test_X, test_Y,'test',opt)
    elif opt.augmentation == "maskp2":
        train_dataset = MaskDataset_train2(train_X, train_Y,'train',opt)
        val_dataset = MaskDataset_test2(val_X, val_Y,'val', opt)
        test_dataset = MaskDataset_test2(test_X, test_Y,'test',opt)
    elif opt.augmentation == "hrv":
        train_dataset = HrvDataset(train_X, train_Y)
        val_dataset = HrvDataset(val_X, val_Y)
        test_dataset = HrvDataset(test_X, test_Y)
    elif opt.augmentation == "lr":
        train_dataset = LorentzDataset(train_X, train_Y)
        val_dataset = LorentzDataset(val_X, val_Y)
        test_dataset = LorentzDataset(test_X, test_Y)
    elif opt.augmentation == "fft":
        train_dataset = FftDataset(train_X, train_Y)
        val_dataset = FftDataset(val_X, val_Y)
        test_dataset = FftDataset(test_X, test_Y)
    elif opt.augmentation == "ffw":
        train_dataset = FftDatasetwavetrain(train_X, train_Y,'train',opt)
        val_dataset = FftDatasetwavetest(val_X, val_Y,'val', opt)
        test_dataset = FftDatasetwavetest(test_X, test_Y,'test',opt)
    elif opt.augmentation == "fft_mask":
        train_dataset = FMDataset(train_X, train_Y, opt)
        val_dataset = FMDataset(val_X, val_Y, opt)
        test_dataset = FMDataset(test_X, test_Y, opt)
    elif opt.augmentation == "fft_mask_downSample":
        train_dataset = FMDSDataset(train_X, train_Y, opt)
        val_dataset = FMDSDataset(val_X, val_Y, opt)
        test_dataset = FMDSDataset(test_X, test_Y, opt)
    elif opt.augmentation == "chaosuan":
        train_dataset = ChaosuankDataset(train_X, train_Y, train_data_abnormal, opt)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)
    elif opt.augmentation == "TF_mask":
        train_dataset = mask_waves(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)
    elif opt.augmentation == "fake":
        train_dataset = TransformDataset_SelectChannel(train_X, train_Y, 20, 0.05, 0.8, 0.3, 100, False, 1)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)
    else:
        train_dataset = RawDataset(train_X, train_Y)
        val_dataset = RawDataset(val_X, val_Y)
        test_dataset = RawDataset(test_X, test_Y)

    dataloader = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "val": DataLoader(
            dataset=val_dataset,
            batch_size=opt.batchsize,
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),

        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=opt.batchsize,  # mini batch size
            shuffle=False,
            num_workers=int(opt.workers),
            drop_last=True),
    }

    return dataloader, X_length, signal_length

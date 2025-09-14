import os

import numpy as np
import pywt
import torch
from matplotlib import pyplot as plt

import transform

from load_vfdb_data import load_vfdb_data
from transform import Gussian, Rayleign, Gamma, Poisson, Exponential, Uniform, Am, Fm, Bw, Compose_Add, Win, \
    Compose, \
    ChannelStraight_1D, ToTensor

os.makedirs('Plot_ECG_and_noisy/', exist_ok=True)


class TransformDataset_SelectChannel:
    def __init__(self, X, Y, snr, fs, amplitude, p, min_winsize, is_selectwin, line_num):
        """
        """
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
        Noisy = trans(X)

        self.X = X
        self.X_Noisy = Noisy
        self.Y = Y

    def __getitem__(self, index):
        # Get path of input image and ground truth

        return self.X[index], self.X_Noisy[index], self.Y[index]

    def __len__(self):
        return self.X.size(0)


def frequency_modulate(time, signal, frequency_base, modulation_index):
    # frequency_base 是信号的基础频率
    # modulation_index 是调频指数，它决定了频率的变化程度

    # 生成一个频率随信号变化的频率信号
    modulated_frequency = frequency_base + modulation_index * signal

    # 产生FM调制信号
    modulated_signal = np.cos(2 * np.pi * modulated_frequency * time)
    return modulated_signal


train_data, train_label, val_data, val_label, test_data = load_vfdb_data()

# data = np.expand_dims(train_data, axis=1)
data = train_data
# label = np.array(data.shape[0] * [0])
# data = TransformDataset_SelectChannel(data, label, 20, 0.05, 0.8, 0.3, 100, False, 1)
# time = np.linspace(0, 10, 2500)  # 时间范围从0到1，共2500个数据点
# modulated_signal = frequency_modulate(time, train_data, 2, 0.1)
# signal = data.X
# noisy = data.X_Noisy
# 小波变换
coeffs = pywt.wavedec(data, 'db1', level=1)  # 选择小波基函数（'db1'为Daubechies 1），level为分解的层数
x_wavelet = coeffs[1]
# coeffs = pywt.wavedec(noisy, 'db1', level=1)
#
# n_wavelet = coeffs[1]
# coeffs = pywt.wavedec(modulated_signal, 'db1', level=1)
# v_wavelet = coeffs[1]
# 应用噪声并可视化
for i in range(x_wavelet.shape[0]):
    plt.figure(figsize=(15, 3))

    # 创建第一个子图：原始信号
    plt.subplot(1, 2, 1)  # (行数, 列数, 子图编号)
    plt.plot(data[i], label='Original Signal')
    plt.legend()
    plt.title('Original Signal')

    plt.subplot(1, 2, 2)  # (行数, 列数, 子图编号)
    plt.plot(x_wavelet[i], label='frequence')
    plt.legend()
    plt.title('frequence')

    # # 创建第二个子图：噪声信号
    # plt.subplot(3, 2, 3)
    # plt.plot(noisy[i], label='Noisy Signal')
    # plt.legend()
    # plt.title('Noisy Signal')
    #
    # plt.subplot(3, 2, 4)
    # plt.plot(n_wavelet[i], label='Noisy Signal')
    # plt.legend()
    # plt.title('Noisy Signal')
    #
    # plt.subplot(3, 2, 5)
    # plt.plot(modulated_signal[i], label='VF Signal')
    # plt.legend()
    # plt.title('VF Signal')
    #
    # plt.subplot(3, 2, 6)
    # plt.plot(v_wavelet[i], label='VF Signal')
    # plt.legend()
    # plt.title('VF Signal')

    plt.tight_layout()  # 调整子图布局
    plt.savefig('Plot_ECG/{}.svg'.format(i))
    plt.show()

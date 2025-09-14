import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import random
import scipy
import os
import pickle as pk
# EEG_data = pk.load(open("EEG_datasets/data0.p", 'rb'), encoding='iso-8859-1')
import torch

first_load = False
def jittering(sample, mean=0, var=None):
    # usually set var as 1, here use a large number to be better demostration of jittering
    jittering_sample = sample + np.random.normal(mean, var ** 0.5, sample.shape)
    return jittering_sample
def flipping(sample):
    flipping_sample = np.flip(sample)
    return flipping_sample

import numpy as np
import scipy.interpolate as spi

def down_sample(signal, sample_rate_1, sample_rate_2):  # sample_rate_1 ----> sample_rate_2
    X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_1))  # 4s  步长0.004对应250HZ  1000个点
    new_X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_2))  # 4S  步长0.002对应500HZ  2000个点
    ipo3 = spi.splrep(X[:len(signal)], signal, k=3)
    iy3 = spi.splev(new_X, ipo3)

    return iy3

def time_warp(x,datasetName,seed,train,method,sigma=0.1, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    if first_load == True:
        for i, pat in enumerate(x):
            for dim in range(x.shape[2]):
                time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(
                    orig_steps)
                scale = (x.shape[1] - 1) / time_warp[-1]

                # 对数据进行抖动处理，加噪
                data = pat[:, dim].copy()
                jittering_sample = jittering(pat[:, dim], mean=0, var=0.001)
                pat[:, dim] = jittering(pat[:, dim], mean=0, var=0.001)

                ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
            if i%1000==0:
                print('-----加载数据:{}------'.format(i/x.shape[0]))
    else:
        if os.path.exists(
                "/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}fake_".format(method,
                                                                                              datasetName) + train + str(
                    seed) + ".npy"):
            ret = np.load(
                "/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}fake_".format(method,
                                                                                              datasetName) + train + str(
                    seed) + ".npy")
        else:
            for i, pat in enumerate(x):
                for dim in range(x.shape[2]):
                    time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(
                        orig_steps)
                    scale = (x.shape[1] - 1) / time_warp[-1]
                    # 对数据进行抖动处理，加噪
                    data = pat[:, dim].copy()
                    jittering_sample = jittering(pat[:, dim], mean=0, var=0.001)

                    pat[:, dim] = jittering(pat[:, dim], mean=0, var=0.001)
                    ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
                    # if i<10:
                    #     plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为中文黑体
                    #     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
                    #     plt.figure(figsize=(40, 5), dpi=300)
                    #     plt.subplot(3, 1, 1)
                    #     plt.plot(data, color='black', linewidth=2.0, label='原始数据')
                    #     plt.legend()
                    #     plt.subplot(3, 1, 2)
                    #     plt.plot(jittering_sample, color='blue', linewidth=2.0, label='抖动数据')
                    #     plt.legend()
                    #     plt.subplot(3, 1, 3)
                    #     plt.plot(ret[i, :, dim], color='black', linewidth=2.0, label='轮转数据')
                    #     plt.legend()
                    #     plt.savefig("{}/{}_{}".format('/home/changhuihui/learn_project/COCA/ecg_dataset/plot_method1', i,
                    #                                   dim) + '.png')
                    # plt.show()
                if i % 1000 == 0:
                    print('-----加载数据:{}------'.format(i / x.shape[0]))
            np.save("/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}fake_".format(method,
                                                                                                  datasetName) + train + str(seed) + ".npy", np.array(ret))
    print("{}数据模拟房颤完成".format(datasetName))
    return ret

import neurokit2 as nk
def mask_waves3(ecg):
    try:
        rpeaks = (nk.ecg_peaks(ecg, sampling_rate=100)[1]['ECG_R_Peaks']).astype(int)
        ppeaks = rpeaks - 13
        ppeaks = ppeaks[ppeaks >= 0]
        if ppeaks.shape[0] == 0:
            ppeaks = []
        # rpeaks=[23,123,200,300,400,500,600,700,800,900]
    except Exception as e:
        rpeaks = []
        ppeaks=[]
        print(e)
        print('检测不到R峰')
    try:
        # ppeak=ppeaks.copy()
        # ppeak = ppeak.tolist()
        # for j in ppeaks:
        #     for key in range(j - 7, j + 8):
        #         ppeak.append(key)
        # plt.figure(figsize=(30, 3), dpi=300)
        # plt.plot(ecg, label='ecg')
        # plt.scatter(rpeaks, np.array(ecg)[rpeaks], label='R-peak')
        # plt.scatter(ppeak, np.array(ecg)[ppeak], label='p-peak')
        # plt.legend()
        # plt.show()
        for ppeak in ppeaks:
            if ppeak - 7 < 0:
                ecg[0:ppeak + 8] = [(ecg[0] + ecg[ppeak + 8]) / 2] * 15
            elif ppeak - 7 >= 0 and ppeak + 8 < 1000:
                ecg[ppeak - 7:ppeak + 8] = [(ecg[ppeak - 7] + ecg[ppeak + 8]) / 2] * 15
            elif ppeak + 8 > 1000:
                ecg[ppeak - 7:1000] = [(ecg[ppeak - 7] + ecg[1000]) / 2] * 15
    except Exception as e:
        print(e)
    return ecg
from tqdm import tqdm
def getP(ecg_signal,datasetName, train, seed,method):
    res_ecg = []
    ecg_signal = ecg_signal.astype('float32')
    if first_load == True:
        for index, data in enumerate(ecg_signal):
            res = []
            for lead in range(data.shape[0]):
                data[lead] = mask_waves3(data[lead])
                res.append(data[lead])
            res_ecg.append(res)
            if index % 1000 == 0:
                print('-----加载数据:{}------'.format(index / ecg_signal.shape[0]))
    else:
        if os.path.exists(
                "/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}mask_".format(method,
                                                                                              datasetName) + train + str(
                    seed) + ".npy"):
            res_ecg = np.load(
                "/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}mask_".format(method,
                                                                                              datasetName) + train + str(
                    seed) + ".npy")
        else:
            for index, data in enumerate(ecg_signal):
                res = []
                for lead in range(data.shape[0]):
                    data[lead] = mask_waves3(data[lead])
                    res.append(data[lead])
                res_ecg.append(res)
                if index % 1000 == 0:
                    print('-----加载数据:{}------'.format(index / ecg_signal.shape[0]))
            np.save(
                ("/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}mask_".format(method,
                                                                                               datasetName) + train + str(
                    seed) + ".npy"), np.array(res_ecg))

    print("MASK OK!")
    return np.array(res_ecg)
def getfakeP(ecg_signal, datasetName,train, seed,method):
    res_ecg = []
    ecg_signal = ecg_signal.astype('float32')
    if first_load == True:
        for index, data in enumerate(ecg_signal):
            res = []
            for lead in range(data.shape[0]):
                data[lead] = mask_waves3(data[lead])
                res.append(data[lead])
            res_ecg.append(res)
            if index % 1000 == 0:
                print('-----加载数据:{}------'.format(index / ecg_signal.shape[0]))
    else:
        if os.path.exists(
                "/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}mask_fake_".format(method,
                                                                                                   datasetName) + train + str(
                    seed) + ".npy"):
            res_ecg = np.load(
                "/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}mask_fake_".format(method,
                                                                                                   datasetName) + train + str(
                    seed) + ".npy")
        else:
            for index, data in enumerate(ecg_signal):
                res = []
                for lead in range(data.shape[0]):
                    data[lead] = mask_waves3(data[lead])
                    res.append(data[lead])
                res_ecg.append(res)
                if index % 1000 == 0:
                    print('-----加载数据:{}------'.format(index / ecg_signal.shape[0]))
        np.save(("/home/changhuihui/learn_project/COCA/augmentation_dataset/{}_{}mask_fake_".format(method,
                                                                                                    datasetName) + train + str(
            seed) + ".npy"), np.array(res_ecg))
    print("MASK OK!")
    return np.array(res_ecg)


def permutation(item, n_segments=5, pertub_mode="random", seg_mode="equal"):
    """This function is for univariate, equal segmentation, and random permuation. But it's easy to expand to more modes."""

    x=item[1]
    T = x.shape[0]
    sublength = int(T / n_segments)
    augmented = np.zeros_like(item)
    idx = np.random.permutation(n_segments)  # random shuffling the order
    #     print(idx)
    for dim in range(item.shape[0]):
        for i in range(n_segments):
            j = idx[i]
            augmented[dim][i * sublength: (i + 1) * sublength] = item[dim][j * sublength: (j + 1) * sublength]
    return augmented
def permutation2(item, n_segments=8, pertub_mode="random", seg_mode="equal"):
    """This function is for univariate, equal segmentation, and random permuation. But it's easy to expand to more modes."""

    x=item[1]
    T = x.shape[0]
    sublength = int(T / n_segments)
    augmented = np.zeros_like(item)
    idx = np.random.permutation(n_segments)  # random shuffling the order
    #     print(idx)
    for dim in range(item.shape[0]):
        for i in range(n_segments):
            j = idx[i]
            augmented[dim][i * sublength: (i + 1) * sublength] = item[dim][j * sublength: (j + 1) * sublength]
    return augmented

def all_zeros(array):
    return all(x == 0 for x in array)
def mask_waves_fake(lie,item):
    ecg=item[1]
    try:
        rpeaks = (nk.ecg_peaks(ecg, sampling_rate=100)[1]['ECG_R_Peaks']).astype(int)

        ppeaks = rpeaks - 13
        ppeaks = ppeaks[ppeaks >= 0]
        peak_on = rpeaks - 20
        peak_on = peak_on[peak_on >= 0]
        peak_off = rpeaks + 40
        peak_off = peak_off[peak_off<1000]
        if ppeaks.shape[0] == 0:
            ppeaks = []
        # rpeaks=[23,123,200,300,400,500,600,700,800,900]
    except Exception as e:
        rpeaks = []
        ppeaks=[]
        peak_on=[]
        peak_off=[]
        print(e)
        print('检测不到R峰')
    try:
        # ppeak = ppeaks.copy()
        # ppeak = ppeak.tolist()
        # for j in ppeaks:
        #     for key in range(j - 7, j + 8):
        #         ppeak.append(key)
        # plt.figure(figsize=(30, 3), dpi=300)
        # plt.plot(item[1], label='ecg')
        # plt.scatter(rpeaks, np.array(item[1])[rpeaks], label='R-peak')
        # plt.scatter(ppeak, np.array(item[1])[ppeak], label='p-peak')
        # plt.scatter(peak_on, np.array(item[1])[peak_on], label='peak_on')
        # plt.scatter(peak_off, np.array(item[1])[peak_off], label='peak_off')
        # plt.legend()
        # plt.savefig("{}/dingwei_{}".format('/home/changhuihui/learn_project/COCA/plot/fake', lie) + '.png', bbox_inches='tight')
        # plt.show()
        for dim in range(item.shape[0]):
            for ppeak in ppeaks:
                if ppeak - 7 < 0:
                    item[dim][0:ppeak + 8] = [(item[dim][0] + item[dim][ppeak + 8]) / 2] * 15
                elif ppeak - 7 >= 0 and ppeak + 8 < 1000:
                    item[dim][ppeak - 7:ppeak + 8] = [(item[dim][ppeak - 7] + item[dim][ppeak + 8]) / 2] * 15
                elif ppeak + 8 > 1000:
                    item[dim][ppeak - 7:1000] = [(item[dim][ppeak - 7] + item[dim][1000]) / 2] * 15
        # plot_sample12(item, '/home/changhuihui/learn_project/COCA/plot/fake', 'fake_{}'.format(dim))
        augmented = np.zeros_like(item)
        if len(peak_on) < len(peak_off):
            peak_off = np.delete(peak_off, 0)
        elif len(peak_on) > len(peak_off):
            peak_on = np.delete(peak_on, len(peak_on) - 1)
        if len(peak_on) < 7:
            augmented = permutation(item)
            # augmented=time_warp2(item)
        elif len(peak_on) > 16:
            augmented = permutation2(item)
        else:
            peak_on = np.delete(peak_on, 0)
            peak_off = np.delete(peak_off, len(peak_off) - 1)
            RR_interal = peak_on - peak_off
            min_RR=np.min(RR_interal)
            if min_RR > 0 and min_RR < 8:
                augmented = permutation(item)
            elif min_RR > 8:
                selected_numbers = sorted(random.sample(range(0, len(RR_interal)), 6))
                idx_sec = random.sample(range(2, 7), 5)
                idx_odd=[]
                idx_even=[]
                for sortdata in idx_sec:
                    if sortdata%2==0:
                        idx_odd.append(sortdata)
                    else:
                        idx_even.append(sortdata)
                idx_sec=np.concatenate((idx_odd,idx_even))
                inter_seg = [0]
                for i in selected_numbers:
                    if i%2==0:
                        random_off = int(RR_interal[i] / 2)
                        random_on = int(random_off / 2)
                        seg_num = random.sample(range(1, random_off), 1)[0]
                    elif i%2==1:
                        random_off = int(RR_interal[i] / 2)
                        random_on = int(random_off / 2)
                        seg_num = random.sample(range(random_off, RR_interal[i]), 1)[0]
                    # seg_num = random.sample(range(0,random_on), 1)[0]
                    seg_off = peak_off[i] + seg_num
                    inter_seg.append(seg_off)
                inter_seg.append(1000)
                for dim in range(item.shape[0]):
                    signal=item[dim][:inter_seg[1]]
                    for num in idx_sec:
                        signal = np.concatenate((signal, item[dim][inter_seg[num - 1]:inter_seg[num]]))
                    signal = np.concatenate((signal, item[dim][inter_seg[len(selected_numbers)]:]))
                    augmented[dim][0:1000] = signal
            else:
                RR_interal=np.abs(RR_interal)
                min_RR2 = np.min(RR_interal)
                if min_RR2<8:
                    augmented = permutation(item)
                else:
                    selected_numbers = sorted(random.sample(range(0, len(RR_interal)), 6))
                    idx_sec = random.sample(range(2, 7), 5)
                    idx_odd = []
                    idx_even = []
                    for sortdata in idx_sec:
                        if sortdata % 2 == 0:
                            idx_odd.append(sortdata)
                        else:
                            idx_even.append(sortdata)
                    idx_sec = np.concatenate((idx_odd, idx_even))
                    inter_seg = [0]
                    for i in selected_numbers:
                        if i % 2 == 0:
                            random_off = int(RR_interal[i] / 2)
                            random_on = int(random_off / 2)
                            seg_num = random.sample(range(1, random_off), 1)[0]
                        elif i % 2 == 1:
                            random_off = int(RR_interal[i] / 2)
                            random_on = int(random_off / 2)
                            seg_num = random.sample(range(random_off, RR_interal[i]), 1)[0]
                        # random_off = int(RR_interal[i] / 2)
                        # random_on = int(random_off / 2)
                        # seg_num = random.sample(range(1, random_on), 1)[0]
                        seg_off = peak_on[i] + seg_num
                        inter_seg.append(seg_off)
                    inter_seg.append(1000)
                    for dim in range(item.shape[0]):
                        signal = item[dim][:inter_seg[1]]
                        for num in idx_sec:
                            signal = np.concatenate((signal, item[dim][inter_seg[num - 1]:inter_seg[num]]))
                        signal = np.concatenate((signal, item[dim][inter_seg[len(selected_numbers)]:]))
                        if signal.shape[0]!=1000:
                            break
                        else:
                            augmented[dim][0:1000] = signal
                    if augmented.any()==False:#如果全部是零元素
                        augmented = permutation(item)
    except Exception as e:
        print(lie)
        print(e)
    return augmented

if __name__ == '__main__':
    ECG_data = np.load('/home/changhuihui/learn_project/COCA/ptb.npy')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    ###房颤伪异常方法一：
    # warped_data = time_warp(ECG_data[10:50, :, ].transpose(0, 2, 1), sigma=0.1, knot=4)
    ###房颤伪异常方法二：
    for i, item in enumerate(ECG_data[10:50, :, ]):
        sample = item.transpose(1, 0)[0:1000, 0]
        # print(sample.shape)
        jittering_sample = jittering(sample, mean=0, var=0.001)
        jittering_sample2 = jittering(sample, mean=0, var=0.0005)
        flipping_sample = flipping(sample)
        permutation_sample = permutation(sample)

        plt.figure(figsize=(40, 5), dpi=300)
        plt.subplot(3, 1, 1)
        plt.plot(sample, color='grey', linewidth=2.0, label='原始数据')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(jittering_sample, color='red', linewidth=2.0, label='抖动数据')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(jittering_sample2, color='red', linewidth=2.0, label='置换数据')
        plt.legend()
        # plt.savefig("{}/{}".format('/home/changhuihui/learn_project/COCA/ecg_dataset/plot_method2', i) + '.png')
        plt.show()






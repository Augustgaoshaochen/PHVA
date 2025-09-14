import numpy as np
import matplotlib.pyplot as plt

from load_vfdb_data import load_vfdb_data
from load_data_mitbih import load_mitbih_data
from load_data_cudb import load_cudb_data

time = np.linspace(0, 10, 2500)  # 时间范围从0到1，共1000个数据点
# train_data, train_label, val_data, val_label, test_data = load_vfdb_data()
# train_data, train_label, val_data, val_label, test_data = load_mitbih_data()
train_data, train_label, val_data, val_label, test_data = load_cudb_data()

data = train_data
original_signal = data  # 一个示例原始信号（频率为250Hz）


# 进行FM调制
def frequency_modulate(time, signal, frequency_base, modulation_index):
    # frequency_base 是信号的基础频率
    # modulation_index 是调频指数，它决定了频率的变化程度

    # 生成一个频率随信号变化的频率信号
    modulated_frequency = frequency_base + modulation_index * signal

    # 产生FM调制信号
    modulated_signal = np.cos(2 * np.pi * modulated_frequency * time)
    return modulated_signal


modulated_signal = frequency_modulate(time, original_signal, 2, 0.1)
save_path = '/home/gaoshaochen/data/vffake_cudb'
np.save(save_path, modulated_signal)
# # 绘制原始信号和调制后的信号
# for i in range(original_signal.shape[0]):
#     plt.figure(figsize=(15, 10))
#
#     # 创建第一个子图：原始信号
#     plt.subplot(2, 1, 1)  # (行数, 列数, 子图编号)
#     plt.plot(original_signal[i], label='Original Signal')
#     plt.legend()
#     plt.title('Original Signal')
#
#     # 创建第二个子图：噪声信号
#     plt.subplot(2, 1, 2)
#     plt.plot(modulated_signal[i], label='Noisy Signal')
#     plt.legend()
#     plt.title('Noisy Signal')
#
#     plt.tight_layout()  # 调整子图布局
#     plt.show()
#     # 显示图形
#     plt.show()

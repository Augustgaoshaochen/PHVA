import os

import numpy as np
import matplotlib.pyplot as plt
def plot_sample1(signal,label,datasetname,index):

    signal = np.squeeze(signal)
    fig = plt.figure(figsize=(24, 3), dpi=300)
    x = np.arange(0, 1000, 100)
    x_labels = np.arange(0, 10)

    plt.plot(signal, color='cadetblue', label="II")
    plt.title("II" + ": ", fontsize=20, color='blue', loc='left')

    plt.xticks(x, x_labels)
    plt.xlabel('time (s)', fontsize=20)
    plt.ylabel('value (mV)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.tight_layout()

    save_dir = "/home/changhuihui/learn_project/COCA/plot_frequency/{}/{}".format(datasetname,label)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig('{}/{}.pdf'.format(save_dir, str(index)), transparent=False, bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(save_dir, str(index)), transparent=False, bbox_inches='tight')
    plt.savefig('{}/{}.svg'.format(save_dir, str(index)), transparent=False, bbox_inches='tight')

    # plt.show()
    plt.close()

    # plt.close()
    # print('Plot')

    plt.show()
    plt.close()
def load_icentia11k_data(data_dir, seed):
    # 获取文件夹下所有npy文件的路径
    data_dirs_afib = [os.path.join(data_dir + "afib", f) for f in os.listdir(data_dir + "afib") if f.endswith(".npy")]
    data_dirs_aflut = [os.path.join(data_dir + "aflut", f) for f in os.listdir(data_dir + "aflut") if
                       f.endswith(".npy")]
    data_dirs_normal = [os.path.join(data_dir + "normal", f) for f in os.listdir(data_dir + "normal") if
                        f.endswith(".npy")]

    random_files_afib = data_dirs_afib[1000:1001]
    random_files_normal = data_dirs_normal[1000:1003]
    random_files_normal.append(data_dirs_aflut[4])
    # random_files_normal.append(data_dirs_aflut[1])

    # random_files_afib = data_dirs_afib[:1]
    # random_files_normal = data_dirs_normal[:3]
    # random_files_normal.append(data_dirs_aflut[1])

    # random_files_afib = data_dirs_afib[:2]
    # random_files_normal =np.concatenate([data_dirs_normal[3:4],data_dirs_aflut[:4]])
    np.random.seed(seed)
    np.random.shuffle(random_files_afib)
    np.random.shuffle(random_files_normal)
    print(random_files_afib)
    print(random_files_normal)

    # random_files_afib = data_dirs_afib[2000:2001]
    # random_files_normal = data_dirs_normal[1000:1003]

    data_X_afib = np.concatenate([np.load(file) for file in random_files_afib])
    data_X_normal = np.concatenate([np.load(file) for file in random_files_normal])
    plot_ecg = True
    if plot_ecg == True:
        for index, itemcpsc in enumerate(data_X_normal[500:600]):
            plot_sample1(itemcpsc, 'NAF', 'icencia', index)
        for index, itemcpsc in enumerate(data_X_afib[500:600]):
            plot_sample1(itemcpsc, 'AF', 'icencia', index)


    np.random.shuffle(data_X_afib)
    np.random.shuffle(data_X_normal)

    data_len = len(data_X_normal)
    train_X = data_X_normal[0:data_len * 2 // 3]

    val_normal_X = data_X_normal[data_len * 2 // 3:data_len * 5 // 6]
    val_abnormal_X = data_X_afib[0:len(data_X_afib) // 2]

    test_normal_X = data_X_normal[data_len * 5 // 6:]
    test_abnormal_X = data_X_afib[len(data_X_afib) // 2:]

    return train_X, val_normal_X, val_abnormal_X, test_normal_X, test_abnormal_X

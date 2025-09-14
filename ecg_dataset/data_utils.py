import numpy as np
import scipy.interpolate as spi

def down_sample(signal, sample_rate_1, sample_rate_2):  # sample_rate_1 ----> sample_rate_2
    X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_1))  # 4s  步长0.004对应250HZ  1000个点
    new_X = np.arange(0, len(signal) * (1 / sample_rate_1), (1 / sample_rate_2))  # 4S  步长0.002对应500HZ  2000个点
    ipo3 = spi.splrep(X[:len(signal)], signal, k=3)
    iy3 = spi.splev(new_X, ipo3)

    return iy3

def time_warp(x, sigma=0.1, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    return ret
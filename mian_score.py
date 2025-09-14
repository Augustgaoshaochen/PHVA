import os
import random

from ecg_dataset.ECG_dataloader import get_dataloader
# from utils_2023.TSNE import tsne_3D, do_tsne

from utils_2023.draw_utils import plot_tsne_sns, plot_hist

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
from options import Options
# from dataset.UCR_dataloader import load_data
import numpy as np
from correlation import plot_distribution,plot_dist

device = torch.device("cuda:1" if
                      torch.cuda.is_available() else "cpu")
opt = Options().parse()

DATASETS_NAME = {
    'cpsc2021': 1,
    # 'ptbxl': 1,
    # 'icentia11k': 1,
}
SEEDS = [
    0, 2, 3
    # 0, 1, 2
    # 1,2
]

if __name__ == '__main__':

    opt.lr = 0.001
    opt.batchsize = 128
    opt.nz = 64
    opt.isMasked = True
    opt.is_all_data = True
    opt.NT = device

    opt.sigm = 1
    opt.sigm_f = 1

    opt.sigm1 = 0.7
    opt.sigm2 = 0.3


    opt.model = 'ae_coca_hrv3_FT_LLLLL'

    if opt.model == 'ae_CP':
        from model.ae_CP import ModelTrainer

        opt.augmentation = 'mask'  # mask lr hrv no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_CP_no':
        from model.ae_CP_no import ModelTrainer

        opt.augmentation = 'mask'  # mask lr hrv no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_TF_CP':
        from model.ae_TF_CP import ModelTrainer

        opt.augmentation = 'fft_mask_downSample'  # mask lr hrv no fft  fft_mask_downSample
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_TF_CP_COCA':
        from model.ae_TF_CP_COCA import ModelTrainer

        opt.augmentation = 'fft_mask'  # mask lr hrv no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_T_CP':
        from model.ae_T_CP import ModelTrainer

        opt.augmentation = 'fft_mask'  # mask lr hrv no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_F_CP':
        from model.ae_F_CP import ModelTrainer

        opt.augmentation = 'fft_mask'  # mask lr hrv no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_TF_CP_nop':
        from model.ae_TF_CP_nop import ModelTrainer

        opt.augmentation = 'fft_mask'  # mask lr hrv no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_coca_hrv3_FT_LLLLL':
        from model.ae_coca_hrv3_TF_LLLLL import ModelTrainer

        opt.augmentation = 'fft'  # mask lr hrv no fft
        opt.nz = 64
        opt.nz_m = 32
    elif opt.model == 'ae_TF_CP_M':
        from model.ae_TF_CP_M import ModelTrainer
        opt.augmentation = 'fft_mask_downSample'  # mask lr hrv no fft fft_mask_downSample
        opt.nz = 64
        opt.nz_m = 32
    else:
        raise Exception("no this model:{}".format(opt.model))

    for dataset_name in list(DATASETS_NAME.keys()):

        results_dir = './ECG/{}/{}/{}/{}'.format(opt.model, dataset_name, opt.sigm1,'z3')

        opt.outf = results_dir

        for normal_idx in range(DATASETS_NAME[dataset_name]):
            opt.numclass = DATASETS_NAME[dataset_name]
            opt.dataset = dataset_name
            if opt.dataset in ['cpsc2021']:
                opt.nc = 2
            elif opt.dataset in ['ptbxl']:
                opt.nc = 12
            elif opt.dataset in ['icentia11k']:
                opt.nc = 1

            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))

            for seed in SEEDS:
                # Set seed
                if seed != -1:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    torch.backends.cudnn.deterministic = True

                opt.seed = seed
                opt.normal_idx = normal_idx

                dataloader, opt.isize, opt.signal_length = get_dataloader(opt)

                print(opt)
                print("################", dataset_name, "##################")

                seed_index = "SEED" + str(seed)

                if opt.model == "ae_TF_CP":
                    model = ModelTrainer(opt, dataloader, device)

                    m = torch.load(
                        "./ECG_CP/{}/{}/model/model_{}_{}.pth".format(opt.model, opt.dataset, opt.model, opt.seed))
                    model.Backbone.load_state_dict(m["Backbone"])
                    model.c = m["center"].to(device)
                    y_true, y_pred, p_t, p_f = model.predict_drew(model.dataloader["test"])
                    plot_hist(y_pred, y_true, opt)
                    plot_tsne_sns(p_t, y_true, opt)
                    # plot_tsne_sns(p_f, y_true, opt)
                    # tsne_3D(p_t, y_true)
                    # do_tsne(p_t, y_true)

                if opt.model == "ae_TF_CP_M":
                    model = ModelTrainer(opt, dataloader, device)

                    m = torch.load(
                        "./ECG_CP/{}/{}/model/model_{}_{}.pth".format(opt.model, opt.dataset, opt.model, opt.seed))
                    model.Backbone.load_state_dict(m["Backbone"])
                    model.c = m["center"].to(device)
                    y_true, y_pred, z, p = model.predict_drew(model.dataloader["test"])
                    # plot_hist(y_pred, y_true, opt)
                    # plot_tsne_sns(z, y_true, opt)
                    plot_tsne_sns(p, y_true, opt)
                if opt.model == 'ae_coca_hrv3_FT_LLLLL':
                    model = ModelTrainer(opt, dataloader, device)

                    m = torch.load(
                        "./ECG/{}/{}/{}/model/model_{}_{}.pth".format(opt.model, opt.dataset,opt.sigm1,opt.model,opt.seed))
                    model.Backbone.load_state_dict(m["Backbone"])
                    model.c = m["c"].to(device)
                    model.c_ft = m["c_ft"].to(device)
                    auc_prc, roc_auc, Pre, Recall, f1 = model.test()
                    print('AUC:',roc_auc,'AP:',auc_prc,'seed:',opt.seed)
                    print('Load model success, ready to train  auc {}   ap {}'.format(roc_auc, auc_prc))
                    # y_true, y_pred, p_t = model.predict_drew(model.dataloader["test"])

                    # max_val = max(np.max(y_pred), np.max(y_true))
                    # min_val = min(np.min(y_pred), np.min(y_true))
                    # # corr = norm(corr)
                    # normal = y_pred[np.where(y_true == 0)]
                    # # normal =  np.where(normal > 0.2, 0.19, normal)
                    # abnormal = y_pred[np.where(y_true != 0)]
                    # num = len(abnormal)
                    # normal = np.sort(y_pred[np.where(y_true == 0)])
                    # num_normal = len(normal)
                    # # index = np.random.randint(0,num_normal,size=[num])
                    # normal = normal[:num_normal]
                    # # abnormal = np.where(n:ormal < 0.2, 0.21, abnormal)
                    # max_val = max(np.max(normal), np.max(abnormal))
                    # min_val = min(np.min(normal), np.min(abnormal))
                    # normal = (normal - min_val) / (max_val - min_val)
                    # abnormal = (abnormal - min_val) / (max_val - min_val)
                    # plot_distribution(abnormal, normal, "Abnormal", "Normal", results_dir,opt)
                    # plot_dist(abnormal, normal, "Abnormal", "Normal", results_dir)

                    # plot_hist(y_pred, y_true, opt)
                    # plot_tsne_sns(p_t, y_true, opt)



import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fastdtw import fastdtw
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, classification_report, confusion_matrix, \
    precision_recall_fscore_support
# from COCA.model.metric_my import evaluate

dirname = os.path.dirname
sys.path.insert(0, dirname(dirname(os.path.abspath(__file__))))

class Encoder(nn.Module):
    def __init__(self, ngpu, opt):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(opt.nc, opt.ndf, 4, 2, 1, bias=False),  # (1,32,4) --> (32,)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),  # 32,64
            nn.BatchNorm1d(opt.ndf * 2),  # 归一化处理 参数为需要归一化的维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),  # 64 ,128
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 4, opt.nz, 2, 1, 0, bias=False),  # 128,50
            nn.AdaptiveAvgPool1d(1)  # 平均池化，将最低维度数转化为1个数
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Decoder(nn.Module):
    def __init__(self, ngpu, opt):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(opt.nz, opt.ngf * 4, 10, 1, 0, bias=False),
            # (batch_size, opt.nz, 1) -> (batch_size, opt.ngf * 4, 10)
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 4, 10) -> (batch_size, opt.ngf * 2, 20)
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 2, 20) -> (batch_size, opt.ngf, 40)
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf, 40) -> (batch_size, opt.nc, 80)
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, 30),  # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.fc(out)

        return out


class Encoder1(nn.Module):
    def __init__(self, opt):
        super(Encoder1, self).__init__()
        self.nc = opt.nc
        self.linear = nn.Linear(opt.nc * 30, opt.nz)

    def forward(self, x):
        x = x.view(-1, self.nc * 30)
        x = self.linear(x)
        return x


class Encoder_m(nn.Module):
    def __init__(self, ngpu, opt):
        super(Encoder_m, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv1d(1, opt.ndf, 4, 2, 1, bias=False),  # (1,32,4) --> (32,)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),  # 32,64
            nn.BatchNorm1d(opt.ndf * 2),  # 归一化处理 参数为需要归一化的维度
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),  # 64 ,128
            nn.BatchNorm1d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(opt.ndf * 4, opt.nz_m, 2, 1, 0, bias=False),  # 128,50
            nn.AdaptiveAvgPool1d(1)  # 平均池化，将最低维度数转化为1个数
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class Decoder_m(nn.Module):
    def __init__(self, ngpu, opt):
        super(Decoder_m, self).__init__()
        self.ngpu = ngpu

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(opt.nz_m, opt.ngf * 4, 10, 1, 0, bias=False),
            # (batch_size, opt.nz, 1) -> (batch_size, opt.ngf * 4, 10)
            nn.BatchNorm1d(opt.ngf * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 4, 10) -> (batch_size, opt.ngf * 2, 20)
            nn.BatchNorm1d(opt.ngf * 2),
            nn.ReLU(True),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf * 2, 20) -> (batch_size, opt.ngf, 40)
            nn.BatchNorm1d(opt.ngf),
            nn.ReLU(True),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose1d(opt.ngf, 1, 4, 2, 1, bias=False),
            # (batch_size, opt.ngf, 40) -> (batch_size, opt.nc, 80)
            nn.Tanh(),
        )
        self.fc = nn.Sequential(
            nn.Linear(80, opt.nz * 2),  # (batch_size, opt.nc, 80) -> (batch_size, opt.nc, opt.isize)
        )

    def forward(self, z):
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.fc(out)

        return out


class base_Model(nn.Module):
    def __init__(self, opt, device):
        super(base_Model, self).__init__()
        self.input_channels = opt.nc
        self.final_out_channels = 16
        self.features_len = 248
        self.project_channels = 20
        self.hidden_size = 64
        self.window_size = 16
        self.device = device
        self.num_layers = 3
        self.kernel_size = 4
        self.stride = 1
        self.dropout = 0.45

        self.nz = opt.nz

        self.conv_block1 = Encoder(opt.ngpu, opt).to(device)
        self.conv_block2 = Encoder(opt.ngpu, opt).to(device)

        self.conv_block3 = Encoder(opt.ngpu, opt).to(device)
        self.conv_block4 = Encoder(opt.ngpu, opt).to(device)

        self.deconv_block = Decoder(opt.ngpu, opt).to(device)

        self.encoder_m = Encoder_m(opt.ngpu, opt).to(device)
        self.decoder_m = Decoder_m(opt.ngpu, opt).to(device)

        self.output_layer = nn.Linear(self.hidden_size, opt.nz * 2)

        self.projection_head = nn.Sequential(
            nn.Linear(opt.nz * 2, 32),
            nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, self.project_channels),
        )

        self.projection_head1 = nn.Sequential(
            nn.Linear(opt.nz, 32),
            nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, self.project_channels),
        )
        self.projection_head2 = nn.Sequential(
            nn.Linear(opt.nz, 32),
            nn.BatchNorm1d(32),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, self.project_channels),
        )

    def init_hidden_state(self, batch_size):
        h = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self, x_t, x_f, x_noisy, n_f, x_vf, v_f):
        if torch.isnan(x_t).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        z_x = self.conv_block1(x_t)
        z_x_f = self.conv_block1(x_f)
        z_x_f = z_x_f.view(128, self.nz, 1)
        # z2 = z2.view(128, self.nz, 1)
        z_n = self.conv_block1(x_noisy)
        z_n_f = self.conv_block1(n_f)
        z_n_f = z_n_f.view(128, self.nz, 1)
        z_x_vf = self.conv_block1(x_vf)
        z_v_f = self.conv_block1(v_f)
        z_v_f = z_v_f.view(128, self.nz, 1)

        z1 = torch.squeeze(z_x)
        z2 = torch.squeeze(z_x_f)
        z3 = torch.squeeze(z_n)
        z4 = torch.squeeze(z_n_f)
        z5 = torch.squeeze(z_x_vf)
        z6 = torch.squeeze(z_v_f)

        # project = self.projection_head(z1)
        # rec_project = self.projection_head(z2)

        return z1, z2, z3, z4, z5, z6


class ModelTrainer(nn.Module):
    def __init__(self, opt, dataloader, device):
        super(ModelTrainer, self).__init__()
        self.niter = opt.niter  # 训练次数 #1000
        self.dataset = opt.dataset  # 数据集
        self.model = opt.model
        self.outf = opt.outf  # 输出文件夹路径

        self.dataloader = dataloader
        self.device = device
        self.opt = opt
        self.sigm = opt.sigm
        # self.sigm2 = opt.sigm2

        self.all_loss = []
        self.rec_loss = []
        self.pre_loss = []
        # 每epoch loss
        self.all_loss_epoch = []
        self.rec_loss_epoch = []
        self.pre_loss_epoch = []

        self.batchsize = opt.batchsize  # input batch size 32
        self.nz = opt.nz  # 潜在z向量大小 8
        self.niter = opt.niter

        self.Backbone = base_Model(opt, device).to(device)

        self.bc_criterion = nn.CrossEntropyLoss()
        self.bce_criterion = nn.BCELoss()
        self.mse_criterion = nn.MSELoss()
        self.BCELoss = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            [
                {"params": self.Backbone.parameters()}],
            lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch = 0

        self.c = torch.zeros(self.opt.nz, device=self.device)

        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        # 输入伪异常
        self.x_noisy = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                   device=self.device)
        self.x_vf = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                device=self.device)
        # 输入频域信号
        self.input_hrv = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.signal_length[0]),
                                     dtype=torch.float32, device=self.device)
        # 输入频域伪异常
        self.n_wavelet = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                     device=self.device)
        self.v_wavelet = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32,
                                     device=self.device)

        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.long, device=self.device)

        self.latent_i_raw = None
        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None

    def train(self):
        self.train_hist = {}
        self.train_hist['loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        print("Train FT_svdd1_fake.")
        start_time = time.time()
        best_result = 0
        best_ap = 0
        best_auc = 0
        best_auc_epoch = 0

        early_stop_epoch = 0
        early_stop_auc = 0

        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch += 1

                # Train
                self.train_epoch()

                self.all_loss_epoch.append(np.sum(self.all_loss) / len(self.all_loss))
                self.rec_loss_epoch.append(np.sum(self.rec_loss) / len(self.rec_loss))
                self.pre_loss_epoch.append(np.sum(self.pre_loss) / len(self.pre_loss))
                self.all_loss = []
                self.rec_loss = []
                self.pre_loss = []

                # Val
                ap, auc, Pre, Recall, f1 = self.validate()

                if auc > best_result:

                    self.save_model()

                    best_result = auc
                    best_auc = auc
                    best_ap = ap
                    best_auc_epoch = self.cur_epoch

                    # Test
                    ap_test, auc_test, Pre_test, Recall_test, f1_test = self.test()

                    if epoch == 1:
                        early_stop_auc = auc_test

                if auc_test <= early_stop_auc:
                    early_stop_epoch = early_stop_epoch + 1
                else:
                    early_stop_epoch = 0
                    early_stop_auc = auc_test

                if early_stop_epoch == self.opt.early_stop:
                    break

                if epoch < 1:
                    self.c = self.center_c(self.dataloader["train"])

                f.write(
                    "EPOCH [{}] auc:{:.4f} \t  ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch, auc, ap, best_auc, best_ap, best_auc_epoch, auc_test, ap_test,
                        early_stop_epoch))
                print(
                    "EPOCH [{}]  loss:{:.4f} \t auc:{:.4f}  \t ap:{:.4f} \t BEST VAL auc:{:.4f} \t  VAL_ap:{:.4f} \t in epoch[{}] \t TEST  auc:{:.4f} \t  ap:{:.4f} \t EarlyStop [{}] \t".format(
                        self.cur_epoch, self.loss, auc, ap, best_auc, best_ap,
                        best_auc_epoch, auc_test, ap_test,
                        early_stop_epoch))
                print("val: pre:{:.4f} \t recall:{:.4f}  \t f1:{:.4f} \t".format(Pre, Recall, f1))
                print("test: pre:{:.4f} \t recall:{:.4f}  \t f1:{:.4f} \t".format(Pre_test, Recall_test, f1_test))
                # self.scheduler.step()
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))

        with open(os.path.join(self.outf, self.model, self.dataset, "loss.txt"), "a+") as f:
            f.write("\nall_loss\n")
            f.write(str(self.all_loss))
            f.write("\nrec_loss\n")
            f.write(str(self.rec_loss))
            f.write("\npre_loss\n")
            f.write(str(self.pre_loss))

        return ap_test, auc_test, best_auc_epoch, Pre_test, Recall_test, f1_test

    def train_epoch(self):

        epoch_start_time = time.time()
        self.Backbone.train()

        epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1
            self.set_input(data)
            self.optimize()
            errors = self.get_errors()
            self.train_hist['loss'].append(errors["loss"])
            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

    def set_input(self, input):
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.input_hrv.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[2].size()).copy_(input[2])
            self.x_noisy.resize_(input[3].size()).copy_(input[3])
            self.n_wavelet.resize_(input[4].size()).copy_(input[4])
            self.x_vf.resize_(input[5].size()).copy_(input[5])
            self.v_wavelet.resize_(input[6].size()).copy_(input[6])

    def optimize(self):
        self.update_net()

    def DTWLoss(self, x, y):
        return fastdtw(x, y)

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def update_net(self):
        self.optimizer.zero_grad()

        # feature1, feature2,  z1, z2 = self.Backbone(self.input, self.input_hrv)
        feature1, feature2, feature3, feature4, feature5, feature6 = self.Backbone(self.input, self.input_hrv,
                                                                                   self.x_noisy, self.n_wavelet,
                                                                                   self.x_vf, self.v_wavelet)
        # feature3 = torch.cat((feature1, feature2), dim=0)

        loss1, score1 = self.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6, self.c)
        # loss2, score2 = self.get_loss_score(feature4, feature5, feature6, self.c2)
        # loss3, score3 = self.get_loss_score((feature1 + feature4), (feature2 + feature5), (feature3 + feature6),
        #                                     self.c3)
        # loss1 = self.mse_criterion(z1, z2)

        self.loss = loss1

        self.all_loss.append(self.loss.item())
        # self.rec_loss.append(self.loss_rec.item())
        # self.pre_loss.append(self.loss_pre.item())

        self.loss.backward()
        self.optimizer.step()

    def center_c(self, train_loader, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(64, device=self.device)

        self.Backbone.eval()

        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                input, input_hrv, label, x_noisy, n_f, x_vf, v_f = data
                input = input.float().to(self.device)
                input_hrv = input_hrv.float().to(self.device)
                x_noisy = x_noisy.float().to(self.device)
                n_f = n_f.float().to(self.device)
                x_vf = x_vf.float().to(self.device)
                v_f = v_f.float().to(self.device)

                outputs, dec, z3, z4, z5, z6 = self.Backbone(input, input_hrv, x_noisy, n_f, x_vf, v_f)
                n_samples += outputs.shape[0]
                all_feature = torch.cat((outputs, z3), dim=0)
                # all_feature = outputs
                c += torch.sum(all_feature, dim=0)

        c /= (2 * n_samples)

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


    def normalized(self,distance):#归一化处理
        range = np.max(distance.cpu().detach().numpy()) - np.min(distance.cpu().detach().numpy())
        normalized_distance = (distance.cpu().detach().numpy() - np.min(
            distance.cpu().detach().numpy())) / range
        normalized_distance = torch.tensor(normalized_distance).to(self.device)
        return normalized_distance

    def get_loss_score(self, feature1, feature2, feature3, feature4, feature5, feature6, center):

        #  将第一 2  4 5 个特征拉近球心，伪异常的特征拉远

        center = center.unsqueeze(0)  # 添加维度
        center = F.normalize(center, dim=1)  # 归一化
        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)
        feature3 = F.normalize(feature3, dim=1)
        feature4 = F.normalize(feature4, dim=1)
        feature5 = F.normalize(feature5, dim=1)
        feature6 = F.normalize(feature6, dim=1)

        feature7 = feature1 + feature2
        feature8 = feature4 + feature3
        feature9 = feature5 + feature6

        distance1 = F.cosine_similarity(feature1, center, eps=1e-6)
        distance2 = F.cosine_similarity(feature2, center, eps=1e-6)
        distance3 = F.cosine_similarity(feature3, center, eps=1e-6)
        distance4 = F.cosine_similarity(feature4, center, eps=1e-6)
        distance5 = F.cosine_similarity(feature5, center, eps=1e-6)
        distance6 = F.cosine_similarity(feature6, center, eps=1e-6)
        distance7 = F.cosine_similarity(feature7, center, eps=1e-6)
        distance8 = F.cosine_similarity(feature8, center, eps=1e-6)
        distance9 = F.cosine_similarity(feature9, center, eps=1e-6)

        distance1 = 1 - distance1
        distance2 = 1 - distance2
        distance3 = 1 - distance3
        distance4 = 1 - distance4
        distance7 = 1 - distance7
        distance8 = 1 - distance8

        # distance9 = 1 + distance9
        # distance5 = 1 + distance5
        # distance6 = 1 + distance6

        # Prevent model collapse
        sigma_aug1 = torch.sqrt(feature1.var([0]) + 0.0001)
        sigma_aug2 = torch.sqrt(feature2.var([0]) + 0.0001)
        sigma_aug3 = torch.sqrt(feature7.var([0]) + 0.0001)
        sigma_loss1 = torch.max(torch.zeros_like(sigma_aug1), (1 - sigma_aug1))
        sigma_loss2 = torch.max(torch.zeros_like(sigma_aug2), (1 - sigma_aug2))
        sigma_loss7 = torch.max(torch.zeros_like(sigma_aug3), (1 - sigma_aug3))
        loss_sigam1 = torch.mean((sigma_loss1 + sigma_loss2 + sigma_loss7) / 3)

        sigma_aug4 = torch.sqrt(feature3.var([0]) + 0.0001)
        sigma_aug5 = torch.sqrt(feature4.var([0]) + 0.0001)
        sigma_aug6 = torch.sqrt(feature8.var([0]) + 0.0001)
        sigma_loss4 = torch.max(torch.zeros_like(sigma_aug4), (1 - sigma_aug4))
        sigma_loss5 = torch.max(torch.zeros_like(sigma_aug5), (1 - sigma_aug5))
        sigma_loss6 = torch.max(torch.zeros_like(sigma_aug6), (1 - sigma_aug6))
        loss_sigam2 = torch.mean((sigma_loss4 + sigma_loss5 + sigma_loss6) / 3)

        sigma_aug7 = torch.sqrt(feature5.var([0]) + 0.0001)
        sigma_aug8 = torch.sqrt(feature6.var([0]) + 0.0001)
        sigma_aug9 = torch.sqrt(feature9.var([0]) + 0.0001)
        sigma_loss7 = torch.max(torch.zeros_like(sigma_aug7), (sigma_aug7))
        sigma_loss8 = torch.max(torch.zeros_like(sigma_aug8), (sigma_aug8))
        sigma_loss9 = torch.max(torch.zeros_like(sigma_aug9), (sigma_aug9))
        loss_sigam3 = torch.mean((sigma_loss7 + sigma_loss8 + sigma_loss9) / 3)

        loss_sigam = (loss_sigam1 + loss_sigam2 + loss_sigam3) / 3

        # The Loss function that representations reconstruction
        score = (distance1 + distance2 + distance6)/3

        triplet_loss = F.relu(distance1 - distance3 + 0.5) + F.relu(distance1 - distance5 + 0.5)
        triplet_loss = torch.mean(triplet_loss)
        triplet_loss2 = F.relu(distance2 - distance4 + 0.5) + F.relu(distance2 - distance6 + 0.5)
        triplet_loss2 = torch.mean(triplet_loss2)
        triplet_loss3 = F.relu(distance7 - distance8 + 0.5) + F.relu(distance7 - distance9 + 0.5)
        triplet_loss3 = torch.mean(triplet_loss3)
        triplet_loss4 = F.relu(distance5 - distance6 + 0.5) + F.relu(distance5-distance9 + 0.5)

        positive_distance = F.cosine_similarity(feature1, feature3, eps=1e-6)
        positive_distance2 = F.cosine_similarity(feature2, feature4, eps=1e-6)
        positive_distance3 = F.cosine_similarity(feature7, feature8, eps=1e-6)
        positive_distance4 = F.cosine_similarity(feature5, feature6, eps=1e-6)


        # 正样本之间的距离
        positive_distance_loss = F.relu(positive_distance - torch.tensor(0.5))
        positive_distance_loss2 = F.relu(positive_distance2 - torch.tensor(0.5))
        positive_distance_loss3 = F.relu(positive_distance3 - torch.tensor(0.5))
        positive_distance_loss4 = F.relu(positive_distance4 - torch.tensor(0.5))

        # 总损失
        # loss1 = triplet_loss + positive_distance_loss
        # loss1 = torch.mean(loss1)
        # loss2 = triplet_loss2 + positive_distance_loss2
        # loss2 = torch.mean(loss2)
        # loss3 = triplet_loss3 + positive_distance_loss3
        # loss3 = torch.mean(loss3)
        # loss4 = triplet_loss4 + positive_distance_loss4
        # loss4 = torch.mean(loss4)
        # loss = (loss1 + loss2 + loss3 + loss4) / 4
        loss1 = triplet_loss
        loss1 = torch.mean(loss1)
        loss2 = triplet_loss2
        loss2 = torch.mean(loss2)
        loss3 = triplet_loss3
        loss3 = torch.mean(loss3)
        loss4 = triplet_loss4
        loss4 = torch.mean(loss4)
        loss = (loss1 + loss2 + loss3 + loss4) / 3

        # print("loss_oc : {}    loss_sigam : {}".format(loss_oc, loss_sigam))
        loss = 1 * loss
        # loss = 1 * loss_oc + 0.1 * loss_sigam

        return loss, score


    def get_errors(self):
        errors = {
            # 'loss_pre': self.loss_pre.item(),
            # 'loss_rec': self.loss_rec.item(),
            'loss': self.loss.item()
        }
        return errors

    def evaluate(self, labels, scores,es_th=None, saveto=None):
        '''
        metric for auc/ap
        :param labels:
        :param scores:
        :param res_th:
        :param saveto:
        :return:
        '''
        scores = torch.from_numpy(scores)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        fpr, tpr, ths = roc_curve(labels, scores)
        # acc = accuracy_score(labels, scores)
        roc_auc = auc(fpr, tpr)
        best_threshold = 0
        def get_percentile(scores, normal_ratio):
            per = np.percentile(scores, normal_ratio)
            return per

        auc_prc = average_precision_score(labels, scores)

        test_normal_ratio = int(len(np.where(labels == 0)[0]) / len(labels) * 100)
        per = get_percentile(scores, test_normal_ratio)
        y_pred = (scores >= per)

        Pre, Recall, f1, _ = precision_recall_fscore_support(labels, y_pred, average='binary')
        return auc_prc, roc_auc, Pre, Recall, f1
    def test(self):
        '''
        test by auc value
        :return: auc
        '''
        y_true, y_pred = self.predict(self.dataloader["test"])
        auc_prc, roc_auc, Pre, Recall, f1 = self.evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1

    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_true, y_pred = self.predict(self.dataloader["val"])
        auc_prc, roc_auc, Pre, Recall, f1 = self.evaluate(y_true, y_pred)
        return auc_prc, roc_auc, Pre, Recall, f1


    def predict(self, dataloader, scale=False):
        with torch.no_grad():
            # 异常评分
            self.an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores1 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores2 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores3 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.latent_i = torch.zeros(size=(len(dataloader.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            # 标签
            self.gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)

            for i, data in enumerate(dataloader, 0):
                self.set_input(data)
                feature1, feature2, feature3, feature4, feature5, feature6 = self.Backbone(self.input, self.input_hrv,
                                                                                           self.x_noisy, self.n_wavelet,
                                                                                           self.x_vf, self.v_wavelet)
                # feature3 = torch.cat((feature1, feature2), dim=0)
                self.latent_i = feature1
                loss1, score1 = self.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6, self.c)
                # loss2, score2 = self.get_loss_score(feature4, feature5, feature6, self.c2)
                # loss3, score3 = self.get_loss_score((feature1 + feature4), (feature2 + feature5), (feature3 + feature6),
                #                                     self.c3)

                self.an_scores1[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score1.size(0))] = score1.reshape(
                    score1.size(0))
                self.gt_labels[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score1.size(0))] = self.label.reshape(
                    score1.size(0))
                self.an_scores = self.an_scores1

            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()
            latent = self.latent_i.cpu().numpy()

            return y_true, y_pred

    def predict_drew(self, dataloader, scale=False):
        with (torch.no_grad()):
            # 异常评分
            self.an_scores = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores1 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores2 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_scores3 = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.float32, device=self.device)
            self.an_feature = torch.zeros(size=(len(dataloader.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)
            # 标签
            self.gt_labels = torch.zeros(size=(len(dataloader.dataset),), dtype=torch.long, device=self.device)

            for i, data in enumerate(dataloader, 0):
                self.set_input(data)
                feature1, feature2, feature3, feature4, feature5, feature6 = self.Backbone(self.input, self.input_hrv,
                                                                                           self.x_noisy, self.n_wavelet,
                                                                                           self.x_vf, self.v_wavelet)
                # feature3 = torch.cat((feature1, feature2), dim=0)

                loss1, score1 = self.get_loss_score(feature1, feature2, feature3, feature4, feature5, feature6, self.c)
                # loss2, score2 = self.get_loss_score(feature4, feature5, feature6, self.c2)
                # loss3, score3 = self.get_loss_score((feature1 + feature4), (feature2 + feature5), (feature3 + feature6),
                #                                     self.c3)

                self.an_scores1[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score1.size(0))] = score1.reshape(
                    score1.size(0))
                self.gt_labels[
                (i * self.opt.batchsize):(i * self.opt.batchsize + score1.size(0))] = self.label.reshape(
                    score1.size(0))

                self.an_scores = self.an_scores1
                self.an_feature[i * self.opt.batchsize: i * self.opt.batchsize + score1.size(0), :] = feature1.reshape(
                    score1.size(0), 64)



                # self.latent1[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = latent1.reshape(
                # error.size(0), self.opt.nz)

            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                        torch.max(self.an_scores) - torch.min(self.an_scores))

            y_true = self.gt_labels.cpu().numpy()
            y_pred = self.an_scores.cpu().numpy()
            feature1=self.an_feature.cpu().numpy()


            return y_true, y_pred, feature1
    def save_model(self):
        model_result = {}
        model_result['c'] = self.c
        # model_result['c_ft'] = self.c_ft
        model_result['Backbone'] = self.Backbone.state_dict()
        chk_dir = '{}/{}'.format(self.opt.outf, "model")
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        torch.save(model_result, chk_dir + '/model_' + self.opt.model + '_' + str(self.opt.seed) + '.pth')

        print('=================================  save model: {}  ================================='.format(
            self.opt.model))

    def load_model(self, model_path):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)
        self.c = model_dict['c']
        self.Backbone.load_state_dict = model_dict['Backbone']
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=100, help='Number of source training epochs')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1, help='Random seed. -1 means do not manually set. ')
parser.add_argument('--data_amount', type=int, default=3, help='0: full data, 30/7/3 correspond to days of data')
parser.add_argument('--data',type=str,default='source/',help='data path')
parser.add_argument('--source', type=str, default="data/pems-bay_", help='pems-bay, metr-la, didi-sz, didi-cd')
parser.add_argument('--target', type=str, default="data/metr-la_", help='Number of source training epochs')
parser.add_argument('--s_num', type=int, default=325, help="number of source nodes")
parser.add_argument('--t_num', type=int, default=207, help="number of target nodes")
args = parser.parse_args()
lrate = args.learning_rate
wdecay = args.weight_decay

if args.seed != -1:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    print('cuda:0')
    device = torch.device('cuda:0')
else:
    print('cpu')
    device = torch.device('cpu')


class GraphLearner(nn.Module):
    def __init__(self, in_dim=288*3):
        super().__init__()
        self.linear1 = nn.Linear(96, 16)
        # self.linear2 = nn.Linear(16, 8)
        self.norm = nn.LayerNorm(normalized_shape=16)

    def forward(self, source_region, target_region):
        sre = self.linear1(source_region)
        tre = self.linear1(target_region)
        sre = self.norm(sre)
        tre = self.norm(tre)
        graph = torch.matmul(sre, tre.T)
        graph_source = torch.matmul(sre, sre.T)
        graph_target = torch.matmul(tre, tre.T)
        return sre, tre, graph, graph_source, graph_target

class trainer():
    def __init__(self):
        self.model = GraphLearner()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        cor2 = np.load("pc_cor.npz")["arr_0"]
        self.pearson_cor = torch.from_numpy(cor2).to(device)
        self.ou_cor = torch.from_numpy(np.load("ou_cor.npz")["arr_0"]).to(device)
        self.cort_cor = torch.from_numpy(np.load("cort_cor.npz")["arr_0"]).to(device)

        # DTW args.t_num->args.s_num
        dtw = np.load("dtw.npz")["arr_0"]
        print(dtw.shape)
        print(dtw.std())
        std = dtw.std()
        for i in range(args.s_num):
            for j in range(args.t_num):
                dtw[i, j] = np.exp((-(dtw[i, j] * dtw[i, j] / (std * std))))
        dtw_cor = dtw

        temp = []
        for i in range(args.s_num):
            sum_line = 0
            for j in range(args.t_num):
                sum_line = sum_line + dtw_cor[i][j]
            temp.append(sum_line)
        for i in range(args.s_num):
            for j in range(args.t_num):
                if temp[i] != 0:
                    dtw_cor[i][j] = dtw_cor[i][j] / temp[i]
                else:
                    dtw_cor[i][j] = dtw_cor[i][j]
        self.dtw_cor = torch.from_numpy(dtw_cor).to(device)

    def train(self, source_region, target_region):
        self.model.train()
        self.optimizer.zero_grad()
        sre, tre, graph, graph_source, graph_target = self.model(source_region, target_region)

        # fusion weight
        emb_rel = torch.matmul(sre, tre.T)
        graph_fused = torch.tensor(self.dtw_cor + self.cort_cor + self.pearson_cor + self.ou_cor + emb_rel)
        graph_fused = F.softmax(graph_fused, dim=1)
        graph_fused = graph_fused.view(-1)

        p_cor = self.pearson_cor.view(-1)
        graph_pearson = p_cor * graph_fused.T
        graph_dtw = self.dtw_cor.view(-1) * graph_fused.T
        graph_ou = self.ou_cor.view(-1) * graph_fused.T
        graph_cort = self.cort_cor.view(-1) * graph_fused.T
        sum_cor = torch.exp(graph_pearson.sum(0)) + torch.exp(graph_ou.sum(0)) + torch.exp(graph_cort.sum(0)) + torch.exp(graph_dtw.sum(0))

        graph_pearson = torch.exp(graph_pearson.sum(0)) / sum_cor
        graph_dtw = torch.exp(graph_dtw.sum(0)) / sum_cor
        graph_ou = torch.exp(graph_ou.sum(0)) / sum_cor
        graph_cort = torch.exp(graph_cort.sum(0)) / sum_cor

        loss_ps = torch.mean(torch.abs(self.pearson_cor - graph))
        loss_dtw = torch.mean(torch.abs(self.dtw_cor - graph))
        loss_ou = torch.mean(torch.abs(self.ou_cor - graph))
        loss_cort = torch.mean(torch.abs(self.cort_cor - graph))

        loss = graph_pearson * loss_ps + graph_ou * loss_ou + graph_cort * loss_cort + graph_dtw * loss_dtw
        loss.backward()
        self.optimizer.step()

    def eval(self, source_region, target_region):
        self.model.eval()
        sre, tre, graph, graph_source, graph_target = self.model(source_region, target_region)
        loss_ps = torch.mean(torch.abs(self.pearson_cor - graph))
        loss_dtw = torch.mean(torch.abs(self.dtw_cor - graph))
        loss_ou = torch.mean(torch.abs(self.ou_cor - graph))
        loss_cort = torch.mean(torch.abs(self.cort_cor - graph))
        loss = loss_ps + loss_dtw + loss_ou + loss_cort
        return loss


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class Max01Scaler():
    def __init__(self, max, min):
        self.max = max
        self.min = min
        print("max", max)
        print("min", min)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


source_region = torch.from_numpy(np.load("source_region.npz")["arr_0"]).to(device)
target_region = torch.from_numpy(np.load("target_region.npz")["arr_0"][:args.t_num]).to(device)
print()
print("source_region", source_region)
print("target_region", target_region)
model = trainer()
for i in range(6000):
    model.train(source_region, target_region)
    loss = model.eval(source_region, target_region)
    print("inter", i, ":", loss)
trained_model = model.model
sre, tre, loss, graph_source, graph_target = trained_model(source_region, target_region)
print(torch.matmul(sre, tre.T).shape)
print(torch.matmul(sre, tre.T).T)

auto_pearson_dtw_cor = torch.matmul(sre, tre.T)
auto_pearson_dtw_cor = auto_pearson_dtw_cor
temp = []
for i in range(args.s_num):
    sum_line = 0
    for j in range(args.t_num):
        if auto_pearson_dtw_cor[i][j] >= 0:
            sum_line = sum_line + auto_pearson_dtw_cor[i][j]
        else:
            auto_pearson_dtw_cor[i][j] = 0
    temp.append(sum_line)
for i in range(args.s_num):
    for j in range(args.t_num):
        if temp[i] != 0:
            auto_pearson_dtw_cor[i][j] = auto_pearson_dtw_cor[i][j] / temp[i]
        else:
            auto_pearson_dtw_cor[i][j] = auto_pearson_dtw_cor[i][j]

print(auto_pearson_dtw_cor.shape)
print(auto_pearson_dtw_cor)

np.savez("cor", auto_pearson_dtw_cor.detach().numpy())

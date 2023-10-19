import numpy as np
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=100, help='Number of source training epochs')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1, help='Random seed. -1 means do not manually set. ')
parser.add_argument('--data_amount', type=int, default=3, help='0: full data, 30/7/3 correspond to days of data')
parser.add_argument('--data',type=str,default='data/',help='data path')
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
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class Autocoder(nn.Module):
    def __init__(self, in_dim=288*3):
        super().__init__()
        # Encoder
        self.mlp_1_1 = nn.Linear(in_dim, 288, bias=True)
        self.mlp_1_2 = nn.Linear(288, 96, bias=True)
        # Decoder
        self.mlp_2_1 = nn.Linear(96, 288, bias=True)
        self.mlp_2_2 = nn.Linear(288, in_dim, bias=True)
        self.activate = nn.Sigmoid()
        self.norm_1_1 = nn.LayerNorm(normalized_shape=288)
        self.norm_1_2 = nn.LayerNorm(normalized_shape=96)
        self.norm_2_1 = nn.LayerNorm(normalized_shape=288)
        self.norm_2_2 = nn.LayerNorm(normalized_shape=in_dim)

    def forward(self, x):
        encoder_x1 = self.mlp_1_1(x)
        encoder_x1 = self.norm_1_1(encoder_x1)
        encoder_x2 = self.mlp_1_2(encoder_x1)
        encoder_x2 = self.norm_1_2(encoder_x2)
        context = encoder_x2

        decoder_x1 = self.mlp_2_1(context)
        decoder_x1 = self.norm_2_1(decoder_x1)
        decoder_x2 = self.mlp_2_2(decoder_x1)
        decoder_x2 = self.norm_2_2(decoder_x2)
        return context, decoder_x2


class trainer():
    def __init__(self, in_dim=288*3):
        self.model = Autocoder(in_dim)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)

    def train(self, x, y, scaler):
        self.model.train()
        self.optimizer.zero_grad()
        context, output = self.model(x)
        output = scaler.inverse_transform(output)
        loss = torch.mean(torch.abs(output - y))
        loss.backward()
        self.optimizer.step()

    def eval(self, x, y, scaler):
        self.model.eval()
        context, output = self.model(x)
        output = scaler.inverse_transform(output)
        loss = torch.mean(torch.abs(output - y))
        # print(loss)
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
        self.min = 0
        print("max", max)
        print("min", min)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


def load_dataset(source, target, batch_size, valid_batch_size= 1, test_batch_size=1):
    # source
    data = {}
    for category in ['train', 'val', 'test']:
        data['x_' + category + "_source"] = np.load(source + "x_" + category + ".npz")['arr_0']
        data['y_' + category + "_source"] = np.load(source + "x_" + category + ".npz")['arr_0']
        print("x_" + category + "_source", data['x_' + category + "_source"].shape)
        print("y_" + category + "_source", data['y_' + category + "_source"].shape)
    scaler = Max01Scaler(max=data['x_train' + "_source"][..., 0].max(), min=data['x_train' + "_source"][..., 0].min())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category + "_source"][..., 0] = scaler.transform(data['x_' + category + "_source"][..., 0])
    data['x_train' + "_source"] = data['x_train' + "_source"][:288 * 3, 0, :, 0]
    data['y_train' + "_source"] = data['y_train' + "_source"][:288 * 3, 0, :, 0]
    data['x_train' + "_source"] = data['x_train' + "_source"].T
    data['y_train' + "_source"] = data['y_train' + "_source"].T
    print("data['x_train']", data['x_train' + "_source"].shape)
    data['train_loader' + "_source"] = DataLoader(data['x_train' + "_source"], data['y_train' + "_source"], batch_size)
    data['val_loader' + "_source"] = DataLoader(data['x_val' + "_source"], data['y_val' + "_source"], valid_batch_size)
    data['test_loader' + "_source"] = DataLoader(data['x_test' + "_source"], data['y_test' + "_source"], test_batch_size)
    data['scaler' + "_source"] = scaler
    data['source_train_data'] = data['x_train' + "_source"]

    # target
    data_target = {}
    for category in ['train', 'val', 'test']:
        data_target['x_' + category + "_target"] = np.load(target + "x_" + category + ".npz")['arr_0']
        data_target['y_' + category + "_target"] = np.load(target + "x_" + category + ".npz")['arr_0']
        print("x_" + category + "_target", data_target['x_' + category + "_target"].shape)
        print("y_" + category + "_target", data_target['y_' + category + "_target"].shape)
    scaler1 = scaler
    # Data format
    for category in ['train', 'val', 'test']:
        data_target['x_' + category + "_target"][..., 0] = scaler1.transform(data_target['x_' + category + "_target"][..., 0])
    data_target['x_train' + "_target"] = data_target['x_train' + "_target"][:288 * 3, 0, :, 0]
    data_target['y_train' + "_target"] = data_target['y_train' + "_target"][:288 * 3, 0, :, 0]
    data_target['x_st'] = np.concatenate([data_target['x_train' + "_target"].T, data['x_train' + "_source"]])
    data_target['y_st'] = np.concatenate([data_target['y_train' + "_target"].T, data['y_train' + "_source"]])
    data_target['x_train' + "_target"] = np.concatenate([data_target['x_train' + "_target"].T, data['x_train' + "_source"]])
    data_target['y_train' + "_target"] = np.concatenate([data_target['y_train' + "_target"].T, data['y_train' + "_source"]])
    print("data_target['y_train']", data_target['y_train' + "_target"].shape)
    print("data_target['x_train']", type(data_target['x_train' + "_target"]))
    data_target['train_loader' + "_target"] = DataLoader(data_target['x_train' + "_target"], data_target['y_train' + "_target"], batch_size)
    data_target['val_loader' + "_target"] = DataLoader(data_target['x_val' + "_target"], data_target['y_val' + "_target"], valid_batch_size)
    data_target['test_loader' + "_target"] = DataLoader(data_target['x_test' + "_target"], data_target['y_test' + "_target"], test_batch_size)
    data_target["st_loader"] = DataLoader(data_target["x_st"], data_target["y_st"], batch_size)
    data_target['scaler' + "_target"] = scaler1
    data_target['target_train_data'] = data_target['x_train' + "_target"][:args.t_num]

    data_combine = {}

    return data, data_target, data_combine


# test the code
Auto_coder_DataLoader_source, Auto_coder_DataLoader_target, Auto_coder_DataLoader_combine = load_dataset(args.source, args.target, args.batch_size, args.batch_size, args.batch_size)
Auto_coder_DataLoader_source['train_loader_source'].shuffle()
Auto_coder_DataLoader_target['train_loader_target'].shuffle()
source_loader = Auto_coder_DataLoader_source['train_loader_source']
target_loader = Auto_coder_DataLoader_target['train_loader_target']
model = trainer(args.data_amount * 288)
source_scaler = Auto_coder_DataLoader_source["scaler_source"]
target_scaler = Auto_coder_DataLoader_target["scaler_target"]


for i in range(200):
    for iter, (x, y) in enumerate(target_loader.get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)
        model.train(trainx, trainy, target_scaler)
    mae = []
    for iter, (x, y) in enumerate(source_loader.get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)
        metrics = model.eval(trainx, trainy, source_scaler)
        mae.append(metrics)
    print("souece iter", i, ":", mae[-1])
    for iter, (x, y) in enumerate(target_loader.get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainy = torch.Tensor(y).to(device)
        metrics = model.eval(trainx, trainy, target_scaler)
        mae.append(metrics)
    print("target iter", i, ":", mae[-1])


source_region = []
target_region = []
source_region_data = Auto_coder_DataLoader_source["source_train_data"]
target_region_data = Auto_coder_DataLoader_target["target_train_data"]
model.model.eval()
AE_source = model.model
AE_target = model.model
for i in range(source_region_data.shape[0]):
    source_context, source_output = AE_source(torch.tensor(source_region_data[i]).to(device))
    source_region.append(source_context.detach().numpy())
for i in range(target_region_data.shape[0]):
    target_context, target_output = AE_target(torch.tensor(target_region_data[i]).to(device))
    target_region.append(target_context.detach().numpy())

print(source_region[0:3])
print(target_region[0:3])
np.savez("source_region", source_region)
np.savez("target_region", target_region)

auto_cor = np.zeros((args.t_num, args.s_num))
for i in range(args.t_num):
    for j in range(args.s_num):
        cor = np.dot(target_region[i], source_region[j])
        if cor <= 0:
            cor = 0
        auto_cor[i][j] = cor
auto_cor = auto_cor.astype(np.float32)

temp = []
for i in range(args.t_num):
    sum_line = 0
    for j in range(args.s_num):
        sum_line = sum_line + auto_cor[i][j]
    temp.append(sum_line)
for i in range(args.t_num):
    for j in range(args.s_num):
        if temp[i] != 0:
            auto_cor[i][j] = auto_cor[i][j] / temp[i]
        else:
            auto_cor[i][j] = auto_cor[i][j]

print(auto_cor)
np.savez("auto_cor", auto_cor.T)

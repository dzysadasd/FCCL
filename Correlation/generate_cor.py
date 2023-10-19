import numpy as np
import argparse
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=100, help='Number of source training epochs')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--source', type=str, default="pems-bay_", help='pems-bay, metr-la, didi-sz, didi-cd')
parser.add_argument('--target', type=str, default="metr-la_", help='Number of source training epochs')
parser.add_argument('--s_num', type=int, default=325, help="number of source nodes")
parser.add_argument('--t_num', type=int, default=207, help="number of target nodes")

args = parser.parse_args()
lrate = args.learning_rate
wdecay = args.weight_decay

baytry = np.load("data/" + args.source + "y_train.npz")['arr_0'][:288*3, 0, :, 0]
baytrx = np.load("data/" + args.source + "x_train.npz")['arr_0'][:288*3, 0, :, 0]
latrx = np.load("data/" + args.target + "x_train.npz")['arr_0'][:288*3, 0, :, 0]
latry = np.load("data/" + args.target + "y_train.npz")['arr_0'][:288*3, 0, :, 0]

if args.source == "pems-bay_" or args.source == "metr-la_":
    baytry = np.load("data/" + args.source + "y_train.npz")['arr_0'][288:288 * 3, 0, :, 0]
    baytrx = np.load("data/" + args.source + "x_train.npz")['arr_0'][288:288 * 3, 0, :, 0]
    latrx = np.load("data/" + args.target + "x_train.npz")['arr_0'][:288 * 2, 0, :, 0]
    latry = np.load("data/" + args.target + "y_train.npz")['arr_0'][:288 * 2, 0, :, 0]

baytry = np.transpose(baytry, (1, 0))
baytrx = np.transpose(baytrx, (1, 0))
latrx = np.transpose(latrx, (1, 0))
latry = np.transpose(latry, (1, 0))

# 1.Pearson
pc_matrix = np.zeros((args.s_num, args.t_num))

for i in range(args.s_num):
    for j in range(args.t_num):
        pc = pearsonr(baytrx[i], latrx[j])
        pc_matrix[i][j] = pc[0]
        # print(pc)

for i in range(args.s_num):
    for j in range(args.t_num):
        if pc_matrix[i][j] < 0.3:
            pc_matrix[i][j] = 0

temp = []
for i in range(args.s_num):
    sum_line = 0
    for j in range(args.t_num):
        sum_line = sum_line + pc_matrix[i][j]
    temp.append(sum_line)
for i in range(args.s_num):
    for j in range(args.t_num):
        if temp[i] != 0:
            pc_matrix[i][j] = pc_matrix[i][j] / temp[i]
        else:
            pc_matrix[i][j] = pc_matrix[i][j]

sum_line = np.sum(pc_matrix, axis=1)
pc_matrix = pc_matrix.astype(np.float32)
np.savez("pc_cor", pc_matrix)
print("pc_cor")

# 2.OU-distance-cor
ou_matrix = np.zeros((args.s_num, args.t_num))
def calEuclidean(x, y):
    dist = np.sqrt(np.sum(np.square(x-y)))   # 注意：np.array 类型的数据可以直接进行向量、矩阵加减运算。np.square 是对每个元素求平均~~~~
    return dist

for i in range(args.s_num):
    for j in range(args.t_num):
        ou = calEuclidean(baytrx[i], latrx[j])
        ou_matrix[i][j] = ou
        # print(ou)
std = ou_matrix.std()
ou_cor = np.zeros((args.s_num, args.t_num))
for i in range(args.s_num):
    for j in range(args.t_num):
        ou_cor[i, j] = np.exp((-(ou_matrix[i, j] * ou_matrix[i, j] / (std * std))))
# norm
temp = []
for i in range(args.s_num):
    sum_line = 0
    for j in range(args.t_num):
        sum_line = sum_line + ou_cor[i][j]
    temp.append(sum_line)
for i in range(args.s_num):
    for j in range(args.t_num):
        if temp[i] != 0:
            ou_cor[i][j] = ou_cor[i][j] / temp[i]
        else:
            ou_cor[i][j] = ou_cor[i][j]
ou_cor = ou_cor.astype(np.float32)
np.savez("ou_cor", ou_cor)
print("ou_cor")

# 3.FOTCC
cort_matrix = np.zeros((args.s_num, args.t_num))
for i in range(args.s_num):
    for j in range(args.t_num):
        sumxy = 0
        sumx = 0
        sumy = 0
        for k in range(288*2-1):
            xt = baytrx[i][k+1] - baytrx[i][k]
            yt = latrx[j][k+1] - latrx[j][k]
            sumxy = sumxy + xt * yt
            sumx = sumx + xt * xt
            sumy = sumy + yt * yt
        sumx = np.sqrt(sumx)
        sumy = np.sqrt(sumy)
        cort = sumxy / (sumx * sumy)
        if cort > 0:
            cort_matrix[i][j] = cort
        else:
            cort_matrix[i][j] = 0
        # print(cort, sumxy, sumx, sumy)

temp = []
for i in range(args.s_num):
    sum_line = 0
    for j in range(args.t_num):
        sum_line = sum_line + cort_matrix[i][j]
    temp.append(sum_line)
for i in range(args.s_num):
    for j in range(args.t_num):
        if temp[i] != 0:
            cort_matrix[i][j] = cort_matrix[i][j] / temp[i]
        else:
            cort_matrix[i][j] = cort_matrix[i][j]

cort_matrix = cort_matrix.astype(np.float32)
np.savez("cort_cor", cort_matrix)
print("cort_cor")


# 4.DTW-Distance
dis_arr = np.zeros((args.s_num, args.t_num))

for i in tqdm(range(args.s_num)):
    for j in range(args.t_num):
        distance, path = fastdtw(baytrx[i], baytrx[j], dist=euclidean)
        dis_arr[i][j] = distance

print(dis_arr)
np.savez("dtw", dis_arr)

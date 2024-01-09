import numpy as np
import random
import math
import os
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from math import sqrt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['font.size'] = 20  # 设置全局字体大小

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
定义可能会用到的函数
"""


# 转换时间格式，将字符串转换成 datatime 格式
def convert_to_time(hmm):
    year, month, day, hour, minute, second = int(hmm[0]), int(hmm[1]), int(hmm[2]), int(hmm[3]), int(hmm[4]), int(
        hmm[5])
    return datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)


# 加载 mat 文件(matlab数据格式)
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split('/')[-1].split('.')[0]

    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0]
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'], d1['time'], d1['data'] = str(col[i][0][0]), int(col[i][1][0]), str(
            convert_to_time(col[i][2][0])), d2
        data.append(d1)

    return data


# 提取锂电池容量
def getBatteryCapacity(Battery):
    # 初始化两个空列表，用于存储循环次数和对应的容量
    cycle, capacity = [], []
    # 初始化循环计数器
    i = 1

    # 遍历Battery列表中的每个电池操作
    for Bat in Battery:
        # 判断当前电池操作的类型是否为'discharge'
        if Bat['type'] == 'discharge':
            # 从数据字典中提取出容量，添加到capacity列表中
            capacity.append(Bat['data']['Capacity'][0])
            # 将当前循环次数添加到cycle列表中，然后递增计数器i
            cycle.append(i)
            i += 1

    # 返回一个包含循环次数列表和容量列表的列表
    return [cycle, capacity]


# 获取锂电池充电或放电时的测试数据
def getBatteryValues(Battery, Type='charge'):
    data = []
    for Bat in Battery:
        if Bat['type'] == Type:
            data.append(Bat['data'])
    return data


"""
导入数据
"""
Battery_list = ['B0005', 'B0006', 'B0007', 'B0018']
dir_path = 'dataset/'

Battery = {}  # 将数据存储到Battery中
for name in Battery_list:
    print('Load Dataset ' + name + '.mat ...')
    path = dir_path + name + '.mat'
    data = loadMat(path)
    Battery[name] = getBatteryCapacity(data)  # 放电时的容量数据

"""
四组电池的寿命情况
"""
fig, ax = plt.subplots(1, figsize=(12, 8))
color_list = ['b-', 'g--', 'r.', 'c-.']
c = 0
for name, color in zip(Battery_list, color_list):
    df_result = Battery[name]
    ax.plot(df_result[0], df_result[1], color, label=name)

ax.set(xlabel='循环次数/次', ylabel='容量/Ah')
ax.legend(loc='upper right')  # 添加图例，位置设为右上角
plt.grid()
plt.show()

"""
对数据划分和评价指标
"""


def build_sequences(text, window_size):
    # 构建序列数据，将输入的text列表按照给定的window_size划分为输入序列x和目标序列y
    x, y = [], []
    for i in range(len(text) - window_size):
        # 获取输入序列
        sequence = text[i:i + window_size]
        # 获取目标序列，目标序列是输入序列的下一个元素
        target = text[i + 1:i + 1 + window_size]

        x.append(sequence)
        y.append(target)

    return np.array(x), np.array(y)


def get_train_test(data_dict, name, window_size=8):
    """
    留一评估：一组数据为测试集，其他所有数据全部拿来训练
    :param data_dict:
    :param name:
    :param window_size:表示窗口大小，默认为8，表示前8个数据为训练集的输入序列，第九个是目标序列
    :return:
    """
    # 获取指定名称的数据序列作为测试集
    data_sequence = data_dict[name][1]
    train_data, test_data = data_sequence[:window_size + 1], data_sequence[window_size + 1:]
    # 构建训练集的输入序列和目标序列
    train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    # 将除了测试集之外的所有数据作为训练集的一部分
    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v[1], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]

    return train_x, train_y, list(train_data), list(test_data)


def relative_error(y_test, y_predict, threshold):
    """

    :param y_test:测试集的真实值，一个包含实数的列表或数组。
    :param y_predict:预测值，一个包含实数的列表或数组。
    :param threshold:阈值，一个实数，用于比较真实值和预测值。
    :return:
    """
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        # 寻找测试集中真实值小于等于阈值，且下一个真实值大于阈值的位置
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        # 寻找预测值小于等于阈值的位置
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re) / true_re


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)  # 平均绝对误差
    mse = mean_squared_error(y_test, y_predict)  # 均方误差
    rmse = sqrt(mean_squared_error(y_test, y_predict))  # 均方根误差
    return mae, rmse


def setup_seed(seed):
    """
    这段函数主要是为了保证在相同的随机种子下，生成的随机数和使用的随机算法具有确定性的
    :param seed:
    :return:
    """
    np.random.seed(seed)  # Numpy模块随机种子
    random.seed(seed)  # Python随机模块随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


"""
网络设计
"""


class Net(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, n_class=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # 添加Dropout层
        self.dropout = nn.Dropout(p=0.3)

        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):  # x shape: (batch_size, seq_len, input_size)
        out, _ = self.cell(x)
        out = out.reshape(-1, self.hidden_dim)

        # 使用Dropout
        out = self.dropout(out)
        out = self.linear(out)  # out shape: (batch_size, n_class=1)
        return out


"""
训练函数
"""


def train(lr=0.001, feature_size=16, hidden_dim=128, num_layers=2, weight_decay=0.0, mode='LSTM', EPOCH=1000, seed=0):
    score_list, result_list = [], []# 存储每个电池的评分和结果列表
    for i in range(4):
        name = Battery_list[i]                                                                                          # 获取电池名称
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size=feature_size)               # 获取训练和测试数据
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(input_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)                   # 创建模型
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)                              # 定义优化器
        criterion = nn.MSELoss()

        test_x = train_data.copy()                                                                                      # 初始化测试数据
        loss_list, y_ = [0], []                                                                                         # 存储损失和预测结果
        mae, rmse, re = 1, 1, 1                                                                                         # 初始化评估指标
        score_, score = 1, 1                                                                                            # 初始化评分指标
        for epoch in range(EPOCH):                                                                                      # 迭代训练多个 epoch
            X = np.reshape(train_x / Rated_Capacity, (-1, 1, feature_size)).astype(
                np.float32)  # (batch_size, seq_len, input_size)                                                        # 将训练数据转换为张量形式
            y = np.reshape(train_y[:, -1] / Rated_Capacity, (-1, 1)).astype(np.float32)                        # shape 为 (batch_size, 1)，将训练标签转换为张量形式

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)                                       # 将训练数据和标签转移到指定设备
            output = model(X)                                                                                           # 模型前向传播
            output = output.reshape(-1, 1)                                                                              # 调整输出形状
            loss = criterion(output, y)                                                                                 # 计算损失
            optimizer.zero_grad()                                                                                       # 清除梯度
            loss.backward()                                                                                             # 反向传播，计算梯度
            optimizer.step()                                                                                            # 更新模型参数

            if (epoch + 1) % 100 == 0:
                test_x = train_data.copy()  # 每100次重新预测一次
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity, (-1, 1, feature_size)).astype(
                        np.float32)
                    x = torch.from_numpy(x).to(device)  # shape: (batch_size, 1, input_size)
                    pred = model(x)
                    next_point = pred.data.numpy()[0, 0] * Rated_Capacity
                    test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
                    point_list.append(next_point)  # 保存输出序列最后一个点的预测值
                y_.append(point_list)  # 保存本次预测所有的预测值
                loss_list.append(loss)
                mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity * 0.7)
                print(
                    'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mae,
                                                                                                       rmse, re))
            score = [re, mae, rmse]
            if (loss < 1e-3) and (score_[0] < score[0]):
                break
            score_ = score.copy()

        score_list.append(score_)
        result_list.append(y_[-1])
    return score_list, result_list



window_size = 16
EPOCH = 1000
lr = 0.001           # learning rate
hidden_dim = 256
num_layers = 2
weight_decay = 0.0
mode = 'LSTM'        # RNN, LSTM, GRU
Rated_Capacity = 2.0

SCORE = []
for seed in range(10):
    print('seed: ', seed)
    score_list, _ = train(lr=lr, feature_size=window_size, hidden_dim=hidden_dim, num_layers=num_layers,
                         weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed)
    print('------------------------------------------------------------------')
    for s in score_list:
        SCORE.append(s)

mlist = ['re', 'mae', 'rmse']
for i in range(3):
    s = [line[i] for line in SCORE]
    print(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))
print('------------------------------------------------------------------')
print('------------------------------------------------------------------')
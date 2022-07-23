from calendar import EPOCH
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
import math
from torch.utils import data

music_ids = []
current_id = ''
time_vs_beat_dic = {} # { '2298': {'time': [], 'beat_0_to_8': []}}
model_input_dic = {}

current_time = 1000000
current_beat = 1000000

with open('time_vs_beat.txt') as file:
    for line in file:
        if line[0:2] == 'id':
            music_id = line[5:9]
            music_ids.append(music_id)

            current_id = music_id
            time_vs_beat_dic[current_id] = {'time': [], 'beat_0_to_8': [], 'beat': []}

        else:
            time = int(line.replace('\n','').split(' ')[0])
            beat = float(line.replace('\n','').split(' ')[1])
            
            if not (time == current_time and beat == current_beat):
                time_vs_beat_dic[current_id]['time'].append(time)
                time_vs_beat_dic[current_id]['beat_0_to_8'].append(beat)
                current_time = time
                current_beat = beat

for id in music_ids:
    time_list = time_vs_beat_dic[id]['time']
    beat_list = time_vs_beat_dic[id]['beat_0_to_8']
    model_input_dic[id] = {}

    current_beat = 0
    count_8_beat = 0
    for t, b in zip(time_list, beat_list):
        delta = b - current_beat
        if delta < -3.5:
            count_8_beat += 1
        current_beat = b
        time_vs_beat_dic[id]['beat'].append(b+count_8_beat*8)
        
        try:
            model_input_dic[id][b+count_8_beat*8].append(t)
        except:
            model_input_dic[id][b+count_8_beat*8] = []
            model_input_dic[id][b+count_8_beat*8].append(t)

    plt.plot(time_vs_beat_dic[id]['time'], time_vs_beat_dic[id]['beat'], label=id)
    
plt.xlabel('onset time')
plt.ylabel('onset beat')
plt.legend()
plt.savefig('musicnet_time_vs_beat/time_vs_beat.jpg', dpi=3000)

# print('==========================================================================')
# print(model_input_dic)

x_beat = []
y_time = []
for id in music_ids[1:2]:
    print('music id:',id)
    for beat in model_input_dic[id]:
        # print(beat, sum(model_input_dic[id][beat])/len(model_input_dic[id][beat]))
        x_beat.append(beat)
        y_time.append(sum(model_input_dic[id][beat])/len(model_input_dic[id][beat]))

class PolyRegression(nn.Module):#继承nn.module
    def __init__(self):
        super(PolyRegression,self).__init__()
        self.linear=nn.Linear(4,1)
        
    def forward(self,x):
        out=self.linear(x)
        return out

class GRU_REG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, seq=4):
        super(GRU_REG, self).__init__()
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers) # rnn
        
        for name, param in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        self.reg = nn.Linear(hidden_size * seq, output_size) # 回归
        
    def forward(self, x):
        x, _ = self.gru(x) # (seq, batch, hidden)
        x = torch.transpose(x, 1, 0) # 调整为 (batch, seq, hidden)
        b, s, h = x.shape
        x = x.reshape(b, s*h) # 转换成线性层的输入格式
        x = self.reg(x)
        return x

# #准备数据
x = torch.FloatTensor(len(x_beat), 1).zero_()
y = torch.FloatTensor(len(y_time), 1).zero_()

for i,b,t in zip(range(len(x_beat)),x_beat,y_time):
    x[i][0] = b
    # x[i][1] = b * b
    # x[i][2] = b * b * b
    # x[i][3] = b * b * b * b
    y[i][0] = t

print(x.shape, y.shape)

# 以8为window size来滑动
tau = 8 # 时间维度/序列长度
features = torch.zeros((len(x_beat) - tau, tau, 1))  # （batch, seq, feature）
for i in range(tau):
    features[:, i] = torch.tensor(y_time)[i: len(x_beat) - tau + i].unsqueeze(1)

labels = torch.tensor(y_time)[tau:]
batch_size, n_train = 16, len(x_beat) # 批量大小、训练样本数量

# 只有前n_train个样本用于训练
data_arrays = (features[:n_train], labels[:n_train])
dataset = data.TensorDataset(*data_arrays)
train_iter = data.DataLoader(dataset, batch_size, shuffle=True) # 打乱顺序

# 实例化一个模型
model = GRU_REG(input_size=1, hidden_size=5, output_size=1, num_layers=2, seq=tau)

#实例化一个损失函数
criterion = nn.MSELoss()

#实例化一个优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# n_epochs = 100000001
n_epochs = 1000
best_loss = 100000000000
patience = 100000
counter = 0
best_model = model

model.train()
for epoch in range(n_epochs):
    epoch_loss = 0
    for num, (X, y) in enumerate(train_iter):
        X = torch.transpose(X, 1, 0) # 将batch和seq的维度置换一下
        y = y.reshape(-1, 1)
        optimizer.zero_grad()
        
        # forward
        output = model(X)
        
        # backward
        loss = criterion(output, y)
        loss.backward()
        # optimize
        optimizer.step()
        
    with torch.no_grad():
        epoch_loss += loss.detach().item()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch + 1}: Loss {epoch_loss:f}') 

    # y_predict=model(x) #y的预测值
    # loss=criteron(y,y_predict) #计算loss
    # optimizer.zero_grad() #梯度清0
    # loss.backward() #反向传播
    # optimizer.step() #更新参数

    # current_loss = loss.item()

    # if epoch % 10000 == 0:
    #     print('Epoch:', epoch, '\tError', current_loss, '\tLearning rate', optimizer.state_dict()['param_groups'][0]['lr'])
    
    # if current_loss < best_loss:
    #     best_loss = current_loss
    #     best_model = model
    # else:
    #     counter += 1
    #     if counter == patience:
    #         print('Early stopping at epoch', epoch, '\tError', best_loss, '\tLearning rate', optimizer.state_dict()['param_groups'][0]['lr'])
    #         break



# # inference, get all time value for all beat value
# print(len(x))
# max_beat = 0
# for i in x:
#     beat_value = i[0]
#     max_beat = max(max_beat, beat_value)

# max_int_beat = math.floor(max_beat)
# all_int_beats = range(max_int_beat)
# missing_int_beats = list(set(all_int_beats) - set(x_beat))
# print(missing_int_beats)
# x_missing_int_beats = torch.FloatTensor(len(missing_int_beats), 1).zero_()
# for index,beat in enumerate(missing_int_beats):
#     x_missing_int_beats[index][0] = beat
#     # x_missing_int_beats[index][1] = beat * beat
#     # x_missing_int_beats[index][2] = beat * beat * beat
#     # x_missing_int_beats[index][3] = beat * beat * beat * beat

# best_model.eval() #进入评估模式


# x_missing_int_beats_features = torch.zeros((len(x_missing_int_beats) - tau, tau, 1))  # （batch, seq, feature）
# for i in range(tau):
#     x_missing_int_beats_features[:, i] = torch.tensor(y_time)[i: len(x_beat) - tau + i].unsqueeze(1)

# for i in range(len(features)//batch_size):

#     inference_predict = best_model(x_missing_int_beats) # 模型的输出
#     inference_output = inference_predict.detach().numpy() # 转成数据准备画图
    
#     predict = best_model(x) # 模型的输出
#     output = predict.detach().numpy() # 转成数据准备画图

# plt.figure()
# plt.scatter(x.numpy()[:, 0],y,c='b',s=0.05,marker='.') # 原始的数据 蓝色
# plt.scatter(x.numpy()[:, 0],output,c='g',s=0.05,marker='.') # 拟合的数据 绿色
# plt.scatter(x_missing_int_beats.numpy()[:, 0],inference_output,c='r',s=2,marker='.') # inference的数据 红色
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'music id = {id}')
# plt.savefig(f'musicnet_time_vs_beat/regression_{id}.jpg', dpi=3000)



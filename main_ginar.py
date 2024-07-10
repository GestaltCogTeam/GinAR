import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from metric.mask_metric import masked_mae,masked_mape,masked_rmse,masked_mse
from model1.ginar_arch import GinAR
from data_solve import load_adj
import time

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

def Inverse_normalization(x,max,min):
    return x * (max - min) + min


### PEMS-BAY、METR-LA、PeMS04、PeMS08
data_name = 'PEMS08'
data_file = "data/" + data_name + "/data.npz"
raw_data = np.load(data_file,allow_pickle=True)

### graph
adj_mx, _ = load_adj("data/" + data_name + "/adj_"  + data_name + ".pkl", "doubletransition")


print(raw_data.files)
batch_size = 16
epoch = 100
IF_mask = 0.25
lr_rate = 0.006

### Hyperparameter
input_len= 12
num_id= 170
out_len=12
in_size=3
emb_size=16
grap_size = 8
layer_num = 2
dropout = 0.15
adj_mx = [torch.tensor(i).float() for i in adj_mx]
max_norm = 5      #Gradient pruning
max_num =  100

###learning rate
num_lr = 5
gamme = 0.5
milestone = [1,15,40,70,90]


### Train_data
if IF_mask == 0.25:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_25"]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_50"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
elif IF_mask == 0.75:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_75"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
elif IF_mask == 0.9:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_90"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
else:
    train_data = torch.cat([torch.tensor(raw_data["train_x_raw"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)

train_data = DataLoader(train_data,batch_size=batch_size,shuffle=True)


### Valid_data
if IF_mask == 0.25:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_25"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_50"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_75"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.9:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask90"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
else:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_raw"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)

valid_data = DataLoader(valid_data,batch_size=batch_size,shuffle=False)

### test_data
if IF_mask == 0.25:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_25"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_50"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_75"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.9:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_90"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
else:
    test_data = torch.cat([torch.tensor(raw_data["test_x_raw"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)

test_data = DataLoader(test_data,batch_size=batch_size,shuffle=False)

max_min = raw_data['max_min']
max_data, min_data = max_min[0],max_min[1]

###CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cpu")

### model
my_net = GinAR(input_len, num_id, out_len, in_size, emb_size, grap_size, layer_num, dropout, adj_mx)

my_net = my_net.to(device)
optimizer = optim.Adam(params=my_net.parameters(),lr=lr_rate)
num_vail = 0
min_vaild_loss = float("inf")

### train
for i in range(epoch):
    num = 0
    loss_out = 0.0
    my_net.train()
    start = time.time()
    for data in train_data:
        my_net.zero_grad()

        train_feature = data[:, :, :,0:in_size].to(device)
        train_target = data[:, :, :,-1].to(device)
        train_pre = my_net(train_feature)
        loss_data = masked_mae(train_pre,train_target,0.0)

        num += 1
        loss_data.backward()

        if max_norm > 0 and i < max_num:
            nn.utils.clip_grad_norm_(my_net.parameters(), max_norm=max_norm)
        else:
            pass
        num += 1

        optimizer.step()
        loss_out += loss_data
    loss_out = loss_out/num
    end = time.time()


    num_va = 0
    loss_vaild = 0.0
    my_net.eval()
    with torch.no_grad():
        for data in valid_data:

            valid_x = data[:, :, :,0:in_size].to(device)
            valid_y = data[:, :, :,-1].to(device)
            valid_pre = my_net(valid_x)
            loss_data = masked_mae(valid_pre, valid_y,0.0)

            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va


    if (i + 1) in milestone:
        for params in optimizer.param_groups:
            params['lr'] *= gamme
            params["weight_decay"] *= gamme

    print('Loss of the {} epoch of the training set: {:02.4f}, Loss of the validation set Loss:{:02.4f}, training time: {:02.4f}:'.format(i+1,loss_out,loss_vaild,end - start))

my_net.eval()
my_net = my_net.to(device2)
with torch.no_grad():
    all_pre = 0.0
    all_true = 0.0
    num = 0
    for data in valid_data:
        test_feature = data[:, :, :,0:in_size].to(device2)
        test_target = data[:, :, :,-1].to(device2)
        test_pre = my_net(test_feature)
        if num == 0:
            all_pre = test_pre
            all_true = test_target
        else:
            all_pre = torch.cat([all_pre, test_pre], dim=0)
            all_true = torch.cat([all_true, test_target], dim=0)
        num += 1

final_pred = Inverse_normalization(all_pre, max_data, min_data)
final_target = Inverse_normalization(all_true, max_data, min_data)


mae,mape,rmse = masked_mae(final_pred, final_target,0.0),\
                masked_mape(final_pred, final_target,0.0)*100,masked_rmse(final_pred, final_target,0.0)
print('RMSE: {}, MAPE: {}, MAE: {}'.format(rmse,mape,mae))



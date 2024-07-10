import numpy as np
import pandas as pd
import copy
import random
import csv

seed = 42
random.seed(seed)
np.random.seed(seed)

data_name = "PEMS08"
data_file_path = data_name + "/" + data_name + ".npz"

history_seq_len = 12
future_seq_len = 12
train_ratio = 0.6
valid_ratio = 0.2
target_channel = 0 # target channel(s)
mask_ratio = 0.25

data = np.load(data_file_path)["data"]
data = data[..., target_channel]
print("raw time series shape: {0}".format(data.shape))

###数据划分
l, n = data.shape
num_samples = data.shape[0] - history_seq_len - future_seq_len + 1
train_num_short = round(num_samples * train_ratio)
valid_num_short = round(num_samples * valid_ratio)
test_num_short = num_samples - train_num_short - valid_num_short

masx_samples_1 = round(n * mask_ratio)

print("number of training samples:{0}".format(train_num_short))
print("number of validation samples:{0}".format(valid_num_short))
print("number of test samples:{0}".format(test_num_short))
print("number of masx samples:{0}".format(masx_samples_1))

###数据标准化
def normalize(x,max_data,min_data):
    return (x - min_data) / (max_data - min_data)

max_data,min_data = data.max(), data.min()
max_min = [max_data,min_data]
max_min = np.array(max_min)


print("----------------0.25 Mask rate-------------------")

mask_id_1 = random.sample(range(n),masx_samples_1)
mask_id_1 = sorted(mask_id_1)

print(mask_id_1)
target_data = data[:,mask_id_1]
feature_data = copy.deepcopy(data)
feature_data[:,mask_id_1] = 0

### normalize
data_new = normalize(data,max_data,min_data)
feature_new = normalize(feature_data,max_data,min_data)
target_new = normalize(target_data,max_data,min_data)

print(feature_new.shape)
print(target_new.shape)

###Partition dataset

def feature_target(data,input_len,output_len):
    fin_feature = []
    fin_target = []
    data_len = data.shape[0]
    for i in range(data_len-input_len - output_len + 1):
        lin_fea_seq = data[i:i+input_len,:]
        lin_tar_seq = data[i+input_len:i+input_len + output_len,:]
        fin_feature.append(lin_fea_seq)
        fin_target.append(lin_tar_seq)
    fin_feature = np.array(fin_feature).transpose((0,2,1))
    fin_target = np.array(fin_target).transpose((0,2,1))
    return fin_feature, fin_target

raw_feature, fin_target = feature_target(data_new ,history_seq_len, future_seq_len)

### Train_data
train_x_raw = raw_feature[0:train_num_short,:,:]
train_y = fin_target[0:train_num_short,:,:]
train_y = train_y.transpose(0,2,1)
train_y = np.expand_dims(train_y,axis=-1)
### vaild_data
vail_x_raw = raw_feature[train_num_short:train_num_short+valid_num_short,:,:]
vail_y = fin_target[train_num_short:train_num_short+valid_num_short,:,:]
vail_y = vail_y.transpose(0,2,1)
vail_y = np.expand_dims(vail_y,axis=-1)
### test_data
test_x_raw = raw_feature[train_num_short+valid_num_short:,:,:]
test_y = fin_target[train_num_short+valid_num_short:,:,:]
test_y = test_y.transpose(0,2,1)
test_y = np.expand_dims(test_y,axis=-1)

mask_id_1 = np.array(mask_id_1)

print("------------------data withou mask------------------")

print(train_x_raw.shape)
print(train_y.shape)

print(vail_x_raw.shape)
print(vail_y.shape)

print(test_x_raw.shape)
print(test_y.shape)

print(train_x_raw[:,6,:])

print("------------------data with mask------------------")

###data with mask
mask_feature, _ = feature_target(feature_new ,history_seq_len, future_seq_len)

### train
train_x_mask1 = mask_feature[0:train_num_short,:,:]
train_x_mask1 = train_x_mask1.transpose(0,2,1)
train_x_mask1 = np.expand_dims(train_x_mask1,axis=-1)
### vaild
vail_x_mask1 = mask_feature[train_num_short:train_num_short+valid_num_short,:,:]
vail_x_mask1 = vail_x_mask1.transpose(0,2,1)
vail_x_mask1 = np.expand_dims(vail_x_mask1, axis=-1)
### test
test_x_mask1 = mask_feature[train_num_short+valid_num_short:,:,:]
test_x_mask1 = test_x_mask1.transpose(0,2,1)
test_x_mask1 = np.expand_dims(test_x_mask1,axis=-1)

print(train_x_mask1.shape)
print(vail_x_mask1.shape)
print(test_x_mask1.shape)

####0.5 Mask rate
print("-----------------0.5 Mask rate-------------------")
mask_ratio_2 = 0.5
masx_samples_2 = round(n * mask_ratio_2)

### 获取被mask掉的序列以及对应的target
mask_id_2 = random.sample(range(n),masx_samples_2)
mask_id_2 = sorted(mask_id_2)

print("number of masx samples:{0}".format(masx_samples_2))

print(mask_id_2)
feature_data_2 = copy.deepcopy(data)
feature_data_2[:,mask_id_2] = 0

feature_new_2 = normalize(feature_data_2,max_data,min_data)
print(feature_new_2.shape)
mask_feature_2, _ = feature_target(feature_new_2 ,history_seq_len, future_seq_len)

### train
train_x_mask_2 = mask_feature_2[0:train_num_short,:,:]
train_x_mask_2 = train_x_mask_2.transpose(0,2,1)
train_x_mask_2 = np.expand_dims(train_x_mask_2,axis=-1)

### vaild
vail_x_mask_2 = mask_feature_2[train_num_short:train_num_short+valid_num_short,:,:]
vail_x_mask_2 = vail_x_mask_2.transpose(0,2,1)
vail_x_mask_2 = np.expand_dims(vail_x_mask_2, axis=-1)

### test
test_x_mask_2 = mask_feature_2[train_num_short+valid_num_short:,:,:]
test_x_mask_2 = test_x_mask_2.transpose(0,2,1)
test_x_mask_2 = np.expand_dims(test_x_mask_2,axis=-1)

print(train_x_mask_2.shape)
print(vail_x_mask_2.shape)
print(test_x_mask_2.shape)
mask_id_2 = np.array(mask_id_2)

# print(train_x_mask_2[:,6,:])

####mask百分之75
print("----------------0.75 Mask rate-------------------")

mask_ratio_3 = 0.75
masx_samples_3 = round(n * mask_ratio_3)
### 获取被mask掉的序列以及对应的target
mask_id_3 = random.sample(range(n),masx_samples_3)
mask_id_3 = sorted(mask_id_3)

print("number of masx samples:{0}".format(masx_samples_3))

print(mask_id_3)
feature_data_3 = copy.deepcopy(data)
feature_data_3[:,mask_id_3] = 0

feature_new_3 = normalize(feature_data_3,max_data,min_data)
print(feature_new_3.shape)
mask_feature_3, _ = feature_target(feature_new_3 ,history_seq_len, future_seq_len)

### train
train_x_mask_3 = mask_feature_3[0:train_num_short,:,:]
train_x_mask_3 = train_x_mask_3.transpose(0,2,1)
train_x_mask_3 = np.expand_dims(train_x_mask_3,axis=-1)

### vaild
vail_x_mask_3 = mask_feature_3[train_num_short:train_num_short+valid_num_short,:,:]
vail_x_mask_3 = vail_x_mask_3.transpose(0,2,1)
vail_x_mask_3 = np.expand_dims(vail_x_mask_3,axis=-1)

### test
test_x_mask_3 = mask_feature_3[train_num_short+valid_num_short:,:,:]
test_x_mask_3 = test_x_mask_3.transpose(0,2,1)
test_x_mask_3 = np.expand_dims(test_x_mask_3,axis=-1)

mask_id_3 = np.array(mask_id_3)

print(train_x_mask_3.shape)
print(vail_x_mask_3.shape)
print(test_x_mask_3.shape)

print(train_x_mask_3[:,6,:])

### 0.9 Mask rate
print("----------------0.9 Mask rate-------------------")

mask_ratio_4 = 0.9
masx_samples_4 = round(n * mask_ratio_4)
### 获取被mask掉的序列以及对应的target
mask_id_4 = random.sample(range(n),masx_samples_4)
mask_id_4 = sorted(mask_id_4)

print("number of masx samples:{0}".format(masx_samples_4))

print(mask_id_4)
feature_data_4 = copy.deepcopy(data)
feature_data_4[:,mask_id_4] = 0

feature_new_4 = normalize(feature_data_4,max_data,min_data)
print(feature_new_4.shape)
mask_feature_4, _ = feature_target(feature_new_4 ,history_seq_len, future_seq_len)

### 训练集划分
train_x_mask_4 = mask_feature_4[0:train_num_short,:,:]
train_x_mask_4 = train_x_mask_4.transpose(0,2,1)
train_x_mask_4 = np.expand_dims(train_x_mask_4,axis=-1)

### 验证集划分
vail_x_mask_4 = mask_feature_4[train_num_short:train_num_short+valid_num_short,:,:]
vail_x_mask_4 = vail_x_mask_4.transpose(0,2,1)
vail_x_mask_4 = np.expand_dims(vail_x_mask_4,axis=-1)

### 测试集划分
test_x_mask_4 = mask_feature_4[train_num_short+valid_num_short:,:,:]
test_x_mask_4 = test_x_mask_4.transpose(0,2,1)
test_x_mask_4 = np.expand_dims(test_x_mask_4,axis=-1)

mask_id_4 = np.array(mask_id_4)

print(train_x_mask_4.shape)
print(vail_x_mask_4.shape)
print(test_x_mask_4.shape)
# print(train_x_mask_4[:,6,:])


print("----------------temporal embeding-------------------")

steps_per_day = 288

tod = [i % steps_per_day /
       steps_per_day for i in range(data_new.shape[0])]
tod = np.array(tod)
tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
tod_tiled,_ = feature_target(tod_tiled[:,:,-1] ,history_seq_len, future_seq_len)
tod_tiled = tod_tiled.transpose(0,2,1)
tod_tiled = np.expand_dims(tod_tiled,axis=-1)

### train
train_tod_tiled = tod_tiled[0:train_num_short,:,:,:]

### vaild
vail_tod_tiled = tod_tiled[train_num_short:train_num_short+valid_num_short,:,:,:]

### test
test_tod_tiled = tod_tiled[train_num_short+valid_num_short:,:,:,:]


dow = [(i // steps_per_day) % 7 / 7 for i in range(data_new.shape[0])]
dow = np.array(dow)
dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
dow_tiled,_ = feature_target(dow_tiled[:,:,-1] ,history_seq_len, future_seq_len)
dow_tiled = dow_tiled.transpose(0,2,1)
dow_tiled = np.expand_dims(dow_tiled,axis=-1)

### train
train_dow_tiled = dow_tiled[0:train_num_short,:,:,:]

### vaild
vail_dow_tiled = dow_tiled[train_num_short:train_num_short+valid_num_short,:,:,:]

### test
test_dow_tiled = dow_tiled[train_num_short+valid_num_short:,:,:,:]


print("----------------final_input_output-------------------")
### train

train_x_mask1 = np.concatenate((train_x_mask1, train_tod_tiled, train_dow_tiled), axis=-1)

train_x_mask_2 = np.concatenate((train_x_mask_2, train_tod_tiled, train_dow_tiled), axis=-1)

train_x_mask_3 =  np.concatenate((train_x_mask_3, train_tod_tiled, train_dow_tiled), axis=-1)

train_x_mask_4 =  np.concatenate((train_x_mask_4, train_tod_tiled, train_dow_tiled), axis=-1)

### vaild
vail_x_mask1 = np.concatenate((vail_x_mask1, vail_tod_tiled, vail_dow_tiled), axis=-1)

vail_x_mask_2 = np.concatenate((vail_x_mask_2, vail_tod_tiled, vail_dow_tiled), axis=-1)

vail_x_mask_3 =  np.concatenate((vail_x_mask_3, vail_tod_tiled, vail_dow_tiled), axis=-1)

vail_x_mask_4 =  np.concatenate((vail_x_mask_4, vail_tod_tiled, vail_dow_tiled), axis=-1)

### test
test_x_mask1 = np.concatenate((test_x_mask1, test_tod_tiled, test_dow_tiled), axis=-1)

test_x_mask_2 = np.concatenate((test_x_mask_2, test_tod_tiled, test_dow_tiled), axis=-1)

test_x_mask_3 = np.concatenate((test_x_mask_3, test_tod_tiled, test_dow_tiled), axis=-1)

test_x_mask_4 = np.concatenate((test_x_mask_4, test_tod_tiled, test_dow_tiled), axis=-1)


### data
np.savez(data_name + "/data" + ".npz",
         train_x_raw=train_x_raw,
         train_x_mask_25 = train_x_mask1,
         train_x_mask_50 = train_x_mask_2,
         train_x_mask_75 = train_x_mask_3,
         train_x_mask_90 = train_x_mask_4,
         train_y = train_y,
         vail_x_raw=vail_x_raw,
         vail_x_mask_25=vail_x_mask1,
         vail_x_mask_50=vail_x_mask_2,
         vail_x_mask_75=vail_x_mask_3,
         vail_x_mask_90=vail_x_mask_4,
         vail_y=vail_y,
         test_x_raw=test_x_raw,
         test_x_mask_25=test_x_mask1,
         test_x_mask_50=test_x_mask_2,
         test_x_mask_75=test_x_mask_3,
         test_x_mask_90=test_x_mask_4,
         test_y=test_y,
         max_min = max_min,
         mask_id_25 = mask_id_1,
         mask_id_50 = mask_id_2,
         mask_id_75 = mask_id_3,
         mask_id_90 = mask_id_4,
         )


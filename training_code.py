
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score



# In[4]:


import torch
from torch import optim
from ANN import ANN
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,r2_score
from copy import deepcopy
from ann_att import ann_att
from sklearn.utils import shuffle


# train_num = 2000;
# other_num = 600;
# feature_num = 21;

# df = pd.read_csv('./training data for GlodAP.csv',usecols=[i for i in range (0,22)],nrows=train_num + 2 * other_num)

# In[5]:


train_num = 160;
other_num = 30;
feature_num = 8;

df = pd.read_excel('./training data for MOSAIC.csv',usecols=[i for i in range (0,9)],nrows=train_num + 2 * other_num)


# In[6]:


data = df.values


# In[7]:


data.shape


# In[8]:


#数据归一化
mean_data = np.mean(data,axis=0)
var_data = np.var(data,axis=0)


# #模拟数据
# x = np.random.rand(3500,1)
# 
# temp = np.zeros((3500,feature_num - 1))
# 
# y = 6*x + 0.5
# 
# mid_value = np.append(x, temp,axis= 1)
# 
# data = np.append(mid_value, y,axis= 1)

# In[9]:


normaliz_data = data


# In[10]:


train_data = normaliz_data[0:train_num,0:feature_num]
train_lable = normaliz_data[0:train_num,-1]

val_data = normaliz_data[train_num:train_num + other_num,0:feature_num]
val_lable = normaliz_data[train_num:train_num + other_num,-1]

test_data = normaliz_data[train_num + other_num:train_num + 2 * other_num,0:feature_num]
test_lable = normaliz_data[train_num + other_num:train_num + 2 * other_num,-1]


# In[11]:


print('train_data.shape：',train_data.shape, ' '+ 'train_lable.shape：',train_lable.shape)
print('val_data.shape：',val_data.shape, ' '+ 'val_lable.shape：',val_lable.shape)
print('test_data.shape：',test_data.shape, ' '+ 'test_lable.shape：',test_lable.shape)


# #数据标准化
# mean = depth[0:90000].mean(axis =0)
# std = depth[0:90000].std(axis =0)
# 
# data = (depth[0:90000] - mean)/std
# 
# mean_lable = oxygen[0:90000].mean(axis =0)
# std_lable = oxygen[0:90000].std(axis =0)
# 
# lable = (oxygen[0:90000] - mean_lable)/std_lable

# In[12]:


#训练参数设置
batch = 32
DEVICE = 'cuda:0'
epoch = 50
input_dim = feature_num
hidden_dim = 8
lr_rate = 0.001
#保存参数设置
Best_PATH = './global_parameters/beststate.pth'


# In[13]:


#创建训练数据载入器
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data).float(), torch.from_numpy(train_lable).float())

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=False,drop_last = True)

#创建验证数据载入器
val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_data).float(), torch.from_numpy(val_lable).float())

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch, shuffle=False,drop_last = True)


#创建测试数据载入器
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data).float(), torch.from_numpy(test_lable).float())

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, shuffle=False,drop_last = True)


# In[14]:


net = ann_att(input_dim,hidden_dim,1).to(DEVICE)


# In[15]:


net


# In[16]:


#定义损失和优化器
optimizer = optim.Adam(net.parameters(), lr = lr_rate)

criterion = nn.SmoothL1Loss().to(DEVICE)


# In[17]:


r2_loss = 0

best_loss = np.inf


# In[18]:


train_record = []
valid_record = []


stop = 0;


# In[19]:


for i in range(100):
   
    step = 0
    v_step = 0
    
    train_loss_value =0    
    valid_loss_value =0    
    test_loss_value =0    

    for batch_id,batch_data in enumerate(train_loader):

        inputs,lables = batch_data

        inputs = inputs.to(DEVICE)
        lables = lables.to(DEVICE)

        optimizer.zero_grad()
        outputs = net(inputs).squeeze()
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        train_loss_value += r2_score(outputs.detach().cpu(),lables.detach().cpu())

        step = step + 1

    print('epoch',i,':'+'train_loss =',train_loss_value/step)
    train_record.append(train_loss_value/step)


    for batch_id,batch_data in enumerate(val_loader):

        inputs,lables = batch_data

        inputs = inputs.to(DEVICE)
        lables = lables.to(DEVICE)

        outputs = net(inputs).squeeze()
        loss = criterion(outputs, lables)

        valid_loss_value += loss.item()

        v_step = v_step + 1
        
    print('valid_loss =',valid_loss_value/v_step)
    valid_record.append(valid_loss_value/v_step)
    
    if valid_loss_value/v_step < best_loss:
       
        best_loss = valid_loss_value/v_step
        
        Best_state = {'net':deepcopy(net.state_dict()),'loss':valid_loss_value/v_step}
        
        stop = 0
    else:
        stop = stop + 1

    if stop == 100:
        
        print(i)
        
             
        
r2_loss = 0

test_net = ann_att(input_dim,hidden_dim,1).to(DEVICE)


test_net.load_state_dict(Best_state['net'])


r2_loss = r2_score(torch.from_numpy(test_lable).float(),test_net(torch.from_numpy(test_data).float().to(DEVICE)).squeeze().detach().cpu())

print(r2_loss)


r2_score(torch.from_numpy(val_lable).float(),test_net(torch.from_numpy(val_data).float().to(DEVICE)).squeeze().detach().cpu())

r2_score(torch.from_numpy(train_lable).float(),test_net(torch.from_numpy(train_data).float().to(DEVICE)).squeeze().detach().cpu())





train_pre_data = np.expand_dims(np.array(test_net(torch.from_numpy(train_data).float().to(DEVICE)).squeeze().detach().cpu()),1)
val_pre_data = np.expand_dims(np.array(test_net(torch.from_numpy(val_data).float().to(DEVICE)).squeeze().detach().cpu()),1)
test_pre_data = np.expand_dims(np.array(test_net(torch.from_numpy(test_data).float().to(DEVICE)).squeeze().detach().cpu()),1)

all_result = np.vstack((train_pre_data,val_pre_data,test_pre_data))


import pandas as pd
import numpy as np
 

df = pd.DataFrame(all_result)
 
# 导出DataFrame到Excel文件
df.to_excel('ANN_att_predict_CN.xlsx', index=False)






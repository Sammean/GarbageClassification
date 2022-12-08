import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import garbageLoader, tools
from model import resnet50
from run import train, validation

# 添加训练设备（if have gpu）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
batch_size = 64
train_set = garbageLoader.GarbageDataset('train.txt')
val_set = garbageLoader.GarbageDataset('test.txt')
train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)

# 构建网络模型
classes_num = 214  # 分类任务的类别数目
model = resnet50.Clean(classes_num)
model = model.cuda(device)

# loss函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda(device)

#优化器
lr_init = 1e-4
weight_decay = 0.0005
optim = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)


# 动态学习率
# step_size : 每多少轮循环后更新一次学习率(lr)
# gamma : 每次更新lr的gamma倍
lr_stepsize = 10
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_stepsize, gamma=0.1)

# 记录训练过程
writer = SummaryWriter('./resnet50')
epoch = 50
best_prec1 = 0
for i in range(epoch):
    print('----------第{}轮训练开始----------'.format(i + 1))
    scheduler.step()
    train.train(train_loader, model, loss_fn, optim, i+1, writer, device)
    # 在测试集上验证
    print('----------第{}轮测试----------'.format(i + 1))
    valid_prec1 = validation.validation(val_loader, model, loss_fn, i+1, writer, device)
    is_best = valid_prec1 > best_prec1
    best_prec1 = max(valid_prec1, best_prec1)
    tools.save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'resnet50',
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optim.state_dict(),
    }, is_best,
        filename='checkpoint_resnet50.pth.tar')

writer.close()


import torch
import os
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from PIL import Image
from tactilemodel import *



import torchvision.transforms as transforms

# 自定义变换：将图像转换为灰度图像，并转换为张量
custom_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root=".\\imagesnew\\train",
                                                 transform=custom_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=".\\imagesnew\\test",
                                                transform=custom_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)



tactile=TactNet4()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(tactile.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 50

# 添加tensorboard
writer = SummaryWriter("./logs_tactile.train")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    tactile.train()#只对特定的Module有作用，例如Dropout,Batchnormal
    for data in train_loader:
        imgs, targets = data

        #if torch.cuda.is_available():
        #    imgs = imgs.cuda()
        #   targets=targets.cuda()

        outputs = tactile(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tactile.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # if torch.cuda.is_available():
            #    imgs = imgs.cuda()
            #   targets=targets.cuda()

            outputs = tactile(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/len(test_dataset)))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/len(test_dataset), total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tactile, "tactile_{}.pth".format(i))
    print("模型已保存")

writer.close()
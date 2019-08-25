import os
import sys
import argparse
import network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
import os
import matplotlib.pyplot as plt
from torch.utils.data import dataloader
import cv2 as cv

def train(n_epoch=5,lr=0.005):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)


    # train_list = os.listdir(os.path.join(data_path, 'train/', 'images/'))
    # label_list = os.listdir((os.path.join(data_path, 'train/', 'labels/')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = network.VGG16(num_class=10)
    #put txo GPU to accelerate calculate
    model.to(device)
    criterior = nn.CrossEntropyLoss()
    print(len(trainset))
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    file = open('SGD.txt',mode="w")
    figure= plt.figure()
    ax_train_acc = figure.add_subplot(1,1,1)
    ax_train_acc.set_ylabel('train accuracy')
    ax_train_loss =ax_train_acc.twinx()
    ax_train_loss.set_ylabel('train loss')
    acc_list = []
    loss_list= []
    for epoch in range(n_epoch):
        time_start = time.time()
        print("epoch:{}/{}".format(epoch,n_epoch))
        file.writelines("epoch:{}/{}".format(epoch,n_epoch)+'\n')
        print('--'*10)
        run_loss = 0
        run_correct = 0
        i=0
        for data in trainloader:
            #print(data.shape)
            #i+=1
            optimizer.zero_grad()
            x_train,y_train =data
            x_train,y_train=torch.autograd.Variable(x_train),torch.autograd.Variable(y_train)
            x_train,y_train = x_train.cuda(),y_train.cuda()
            output = model(x_train)
            loss = criterior(output,y_train)
            loss.backward()
            optimizer.step()
            _ ,pred = torch.max(output.data,1)
            run_loss+=loss
            #print(pred==y_train)
            run_correct += torch.sum(pred==y_train)
        run_correct=run_correct.cpu()
        print("loss:{:.4f},accuracy:{:.4f}%".format(run_loss / len(trainset), 100 * run_correct / len(trainset)))
        loss = float(run_loss / len(trainset))
        acc = float(100 * run_correct / len(trainset))
        loss_list.append(loss)
        acc_list.append(acc)
        if acc==99:
            break
        # ax_train_acc.plot(epoch,acc,label = 'train_acc')
        # ax_train_loss.plot(epoch,loss,label = 'train_loss')
        file.writelines("loss:{:.4f},accuracy:{:.4f}%".format(run_loss / len(trainset), 100 * run_correct / len(trainset))+'\n')
    file.close()
    ax_train_acc.plot(range(len(acc_list)),acc_list,label = 'train_acc')
    ax_train_loss.plot(range(len(loss_list)),loss_list,label = 'train_loss')
    plt.show()
    plt.savefig('./tran_loss.png')
    torch.save(model,'vgg16.pkl')


def test():
    transform1 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                             shuffle=False, num_workers=2)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='pytorch vgg16net')
    parser.add_argument('--model',type=str,default='vgg16')
    #parser.add_argument('--data_path',type=str,default='data/')
    parser.add_argument('--lr',type=float,default=0.05)
    parser.add_argument('--epoch',type=int,default=100)
    args = parser.parse_args()
    #data_path = args.data_path
    epoch = args.epoch
    lr = args.lr
    print(args)
    train(epoch,lr)
    #test(data_path)
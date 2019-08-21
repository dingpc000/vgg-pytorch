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

def train(n_epoch=1000,lr=0.05):
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
    # model = torchvision.models.vgg16()
    # model.classifier=nn.Sequential(
    #     nn.Sequential(
    #         nn.Linear(512 * 1 * 1, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 4096),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(4096, 10),
    #     )
    #
    # )
    #put to GPU to accelerate calculate
    model.to(device)
    criterior = nn.CrossEntropyLoss()
    print(len(trainset))
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    for epoch in range(n_epoch):
        time_start = time.time()
        print("epoch:{}/{}".format(epoch,n_epoch))
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
        print("loss:{:.4f},accuracy:{:.4f}%".format(run_loss / len(trainset), 100 * run_correct / len(trainset)))
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
    parser.add_argument('--epoch',type=int,default=1000)
    args = parser.parse_args()
    #data_path = args.data_path
    epoch = args.epoch
    lr = args.lr
    print(args)
    train(epoch,lr)
    #test(data_path)
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from djpegnet import Djpegnet
from data import SingleDoubleDataset, SingleDoubleDatasetValid

def train(dataloader, epoch):
    print('[Epoch %d]' % (epoch+1))
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    for batch_idx, samples in enumerate(dataloader):
        Ys, qvectors, labels = samples[0].to(device), samples[1].to(device), samples[2].to(device)
        Ys = Ys.float()
        Ys = torch.unsqueeze(Ys, axis=1)
        qvectors = qvectors.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(Ys, qvectors)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx % args.loss_interval == (args.loss_interval-1):
            print('[%d, %5d] loss: %.6f' %
                (epoch + 1, batch_idx + 1, running_loss / args.loss_interval))
            running_loss = 0.0

        if args.split_training:
            if batch_idx > 10*args.loss_interval:
                break


def valid(dataloader, epoch):
    classes = ('single', 'double')
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_acc = list(0. for i in range(2))
    with torch.no_grad():
        for samples in dataloader:
            Ys, qvectors, labels = samples[0].to(device), samples[1].to(device), samples[2].to(device)
            Ys = Ys.float()
            Ys = torch.unsqueeze(Ys, axis=1)
            qvectors = qvectors.float()

            # feed forward
            outputs = net(Ys, qvectors)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(args.batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(2):
        class_acc[i] = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %.2f %%' % (
            classes[i], class_acc[i]))

    total_acc = (class_acc[0]+class_acc[1])/2
    print('Accuracy of %5s : %.2f %%' % ('total', total_acc))

    # calculate valid best
    if total_acc > valid_best['total_acc']:
        valid_best['total_acc'] = total_acc
        valid_best['single_acc'] = class_acc[0]
        valid_best['double_acc'] = class_acc[1]
        valid_best['epoch'] = epoch+1

        #save model
        os.makedirs('./model', exist_ok=True)
        torch.save(net.state_dict(), './model/valid-best.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size.')
    parser.add_argument('--data_path', type=str, default='./jpeg_data/', help='path of jpeg dataset.')
    parser.add_argument('--epoch', type=int, default=10, help='number of training epoch.')
    parser.add_argument('--loss_interval', type=int, default=500)
    parser.add_argument('--split_training', default=False, action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_path = args.data_path
    train_dataset = SingleDoubleDataset(data_path)
    valid_dataset = SingleDoubleDatasetValid(data_path)

    net = Djpegnet(device)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    valid_best = dict(epoch=0, single_acc=0, double_Acc=0, total_acc=0)

    for epoch in range(0, args.epoch):
        net.train()
        train(train_dataloader, epoch)
        net.eval()
        valid(valid_dataloader, epoch)

    #print valid best
    print('[Best epoch: %d]' % valid_best['epoch'])
    print('Accuracy of %5s: %.2f %%' % ('single', valid_best['single_acc']))
    print('Accuracy of %5s: %.2f %%' % ('double', valid_best['double_acc']))
    print('Accuracy of %5s: %.2f %%' % ('total', valid_best['total_acc']))
    print('Done')

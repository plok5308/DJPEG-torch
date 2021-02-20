import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torchvision.models as models

#from model import Net
from djpegnet import Djpegnet
from data import SingleDoubleDataset, SingleDoubleDatasetValid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
classes = ('single','double')
valid_best = dict(epoch=0, single_acc=0, double_Acc=0, total_acc=0)
batch_size = 64

def train(dataloader, epoch):
    print('[epoch : %d]' % (epoch+1))
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    for batch_idx, samples in enumerate(dataloader):
        #inputs, labels = samples
        inputs, labels = samples[0].to(device), samples[1].to(device)

        inputs = inputs.float()
        #labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.6f' %
                (epoch + 1, batch_idx + 1, running_loss / 50))
            running_loss = 0.0

    torch.save(net.state_dict(), './trained_model/test.pth')

def valid(dataloader, epoch):
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_acc = list(0. for i in range(2))
    with torch.no_grad():
        for samples in dataloader:
            inputs, labels = samples[0].to(device), samples[1].to(device)
            inputs = inputs.float()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(2):
        class_acc[i] = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %.2f %%' % (
            classes[i], class_acc[i]))

    total_acc = (class_acc[0]+class_acc[1])/2
    print('Accuracy of %5s : %.2f %%' % ('Total', total_acc))

    # calculate valid best
    if total_acc > valid_best['total_acc']:
        valid_best['total_acc'] = total_acc
        valid_best['single_acc'] = class_acc[0]
        valid_best['double_acc'] = class_acc[1]
        valid_best['epoch'] = epoch+1




print('hello world.')
train_dataset = SingleDoubleDataset()
valid_dataset = SingleDoubleDatasetValid()

net = Djpegnet()
net.to(device)
optimizer = torch.optim.Adam(net.parameters())

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

for epoch in range(0,100):
    net.train()
    train(train_dataloader, epoch)
    net.eval()
    valid(valid_dataloader, epoch)

#print valid best
print('[Best epoch : %d]' % valid_best['epoch'])
print('Accuracy of %5s : %.2f %%' % ('single', valid_best['single_acc']))
print('Accuracy of %5s : %.2f %%' % ('double', valid_best['double_acc']))
print('Accuracy of %5s : %.2f %%' % ('Total', valid_best['total_acc']))
print('done')

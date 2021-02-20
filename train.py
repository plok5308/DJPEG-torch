import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from djpegnet import Djpegnet
from data import SingleDoubleDataset, SingleDoubleDatasetValid

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
classes = ('single','double')
valid_best = dict(epoch=0, single_acc=0, double_Acc=0, total_acc=0)
batch_size = 32

def train(dataloader, epoch):
    print('[epoch : %d]' % (epoch+1))
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    for batch_idx, samples in enumerate(dataloader):
        Ys, qvectors, labels = samples[0].to(device), samples[1].to(device), samples[2].to(device)

        Ys = Ys.float()
        Ys = torch.unsqueeze(Ys, axis=1)
        qvectors = qvectors.float()
        #labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(Ys, qvectors)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx % 200 == 199:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.6f' %
                (epoch + 1, batch_idx + 1, running_loss / 199))
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
data_path = '/home/jspark/Project/data_custom/jpeg_data/'
train_dataset = SingleDoubleDataset(data_path)
valid_dataset = SingleDoubleDatasetValid(data_path)

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

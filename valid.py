import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm

from djpegnet import Djpegnet
from data import SingleDoubleDataset, SingleDoubleDatasetValid

def valid(dataloader, epoch):
    classes = ('single', 'double')
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_acc = list(0. for i in range(2))
    with torch.no_grad():
        for samples in tqdm(dataloader):
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
    print('Accuracy of %5s : %.2f %%' % ('Total', total_acc))

    # calculate valid best
    if total_acc > valid_best['total_acc']:
        valid_best['total_acc'] = total_acc
        valid_best['single_acc'] = class_acc[0]
        valid_best['double_acc'] = class_acc[1]
        valid_best['epoch'] = epoch+1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size.')
    parser.add_argument('--data_path', type=str, default='./djpeg_dataset/jpeg_data/', help='path of jpeg dataset.')
    parser.add_argument('--model_path', type=str, default='./djpegnet/djpegnet.pth', help='trained network path.')
    args = parser.parse_args()



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_path = args.data_path
    valid_dataset = SingleDoubleDatasetValid(data_path)

    net = Djpegnet(device)
    #load weights
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters())

    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    valid_best = dict(epoch=0, single_acc=0, double_Acc=0, total_acc=0)
    net.eval()
    valid(valid_dataloader, epoch=0)
    print('Done')

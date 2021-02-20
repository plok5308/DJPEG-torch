import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = models.resnet18(pretrained=False, num_classes=64)
        self.net.load_state_dict(torch.load('./resnet18_200923.pth'))
        self.mha = nn.MultiheadAttention(1, 1)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.net(x)
        #import pdb; pdb.set_trace()
        x = torch.unsqueeze(x, axis=2)
        x = self.mha(x, x, x)[0]
        x = torch.squeeze(x, axis=2)
        x = self.fc2(x)
        return x


class DJPEG_Transformer(nn.Module):
    def __init__(self):
        super(DJPEG_Transformer, self).__init__()
        self.encoder = nn.Transformer(64, 4).encoder #vector dims - 192, multi head number -8
        #output of encoder [sequence, N, vector dims] - [1024, N, 192]
        self.fc1 = nn.Linear(1024*64, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        #NSD -> SND
        x = x.permute(1,0,2)
        x = self.encoder(x)
        #x = x.permute(1,0,2) #SND -> NSD
        #import pdb; pdb.set_trace()
        #x = x.view(-1, 1024*192)
        #x = torch.flatten(x, start_dim=1)
        x = torch.flatten(x, start_dim=0)
        #import pdb; pdb.set_trace()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.unsqueeze(x,axis=0)
        #import pdb; pdb.set_trace()
        return x




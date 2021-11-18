import os
import torch
import torch.nn.functional as F
import math
from PIL import JpegImagePlugin
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
from djpegnet import Djpegnet
from data import read_q_table

def _extract_patches(Y, patch_size, stride):
    patches=list()
    h, w = Y.shape[0:2]
    H = (h - patch_size) // stride
    W = (w - patch_size) // stride
    for i in range(0,H*stride, stride):
        for j in range(0,W*stride,stride):
            patch = Y[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    return patches, H, W

def localizing_double_JPEG(Y, qvectors):
    net.eval()
    result=0
    PATCH_SIZE = 256

    qvectors = torch.from_numpy(qvectors).float()
    qvectors = qvectors.to(device)
    qvectors = torch.unsqueeze(qvectors, axis=0)

    #result = np.zeros_like(Y)

    patches, H, W = _extract_patches(Y, patch_size=PATCH_SIZE, stride=args.stride)
    result = np.zeros((H, W))

    #import pdb; pdb.set_trace()
    num_batches = math.ceil(len(patches) / args.batch_size)

    result_flatten = np.zeros((H*W))
    for i in range(num_batches):
        print('[{} / {}] Detecting...'.format(i, num_batches))
        if i==(num_batches-1): #last batch
            batch_Y = patches[i*args.batch_size:]
        else:
            batch_Y = patches[i*args.batch_size:(i+1)*args.batch_size]

        batch_size = len(batch_Y)
        batch_Y = np.array(batch_Y)
        batch_Y = torch.unsqueeze(torch.from_numpy(batch_Y).float().to(device), axis=1)
        batch_qvectors = torch.repeat_interleave(qvectors, batch_size, dim=0)
        batch_output = net(batch_Y, batch_qvectors)
        batch_output = F.softmax(batch_output, dim=1)

        result_flatten[(i*args.batch_size):(i*args.batch_size)+batch_size] = \
                batch_output.detach().cpu().numpy()[:,0]

    result = np.reshape(result_flatten, (H, W))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--target', type=str, default='copy_move.jpg')
    args = parser.parse_args()

    dir_name = './images'
    file_name = args.target
    result_name = file_name.split('.')[0] + '_result.jpg'
    file_path = os.path.join(dir_name, file_name)
    result_path = os.path.join(dir_name, result_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #read an image
    #img = np.asarray(Image.open(file_path))
    im = Image.open(file_path)
    im = im.convert('YCbCr')
    Y = np.array(im)[:,:,0]

    #read quantization table of Y channel from jpeg images
    qvector = read_q_table(file_path).flatten()

    #load pre-trained weights
    net = Djpegnet(device)
    net.load_state_dict(torch.load('./model/djpegnet.pth', map_location=device))
    net.to(device)

    result = localizing_double_JPEG(Y, qvector) #localizaing using trained detecting double JPEG network.

    #plot and save the result
    fig = plt.figure()
    columns = 2
    rows = 1
    fig.add_subplot(rows, columns, 1)
    plt.imshow(Image.open(file_path))
    plt.title('input')

    fig.add_subplot(rows, columns, 2)
    result = result*255
    result = result.astype('uint8')
    img_result = Image.fromarray(result)
    img_result.convert("L")
    plt.imshow(img_result, cmap='gray', vmin=0, vmax=255)
    plt.title('result')
    plt.savefig(result_path)
    plt.show()


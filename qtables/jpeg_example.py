"""This code shows that how to make the training data with collected quantization tables. 
This code is an example, you can use another approach to make jpeg images."""

import scipy.io as sio
from PIL import Image
from PIL import JpegImagePlugin
import numpy as np

def load_qtable(idx):
    '''
    load quantization table from the mat file.

    args:
        idx - quantization table index ( [0, 1169] ).
    return:
        qtable - 8x8 numpy.ndarray that has uint8 type.

    '''
    q_tables = sio.loadmat('q_table_list.mat')['q_table'][:,0]
    return q_tables[idx]

def custom_jpeg(jpgfile, qtable):
    '''
    custom jpeg compression with the qtable.
    '''

    zigzag_index = np.array(
    [[0,  1,  5,  6, 14, 15, 27, 28],
    [2,  4,  7, 13, 16, 26, 29, 42],
    [3,  8, 12, 17, 25, 30, 41, 43],
    [9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]])

    qvector = np.zeros(64)
    for h in range(8):
        for w in range(8):
            qvector[zigzag_index[h,w]] = qtable[h,w]

    #TODO: Dequantize Y values from jpgfile's Y quantized data and quantization table.
    #TODO: Quantize Y values using the qtable.
    #TODO: replace Y quantization table of jpgfile to qvector.
    #TODO: save jpg file.

if __name__ == "__main__":
    jpg = JpegImagePlugin.JpegImageFile('dog.jpg')
    qtable1 = load_qtable(50)
    custom_jpeg(jpg, qtable1)
    
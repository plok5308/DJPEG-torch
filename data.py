import torch
import glob
import numpy as np
from PIL import Image
from PIL import JpegImagePlugin

def read_q_table(file_name):
    jpg = JpegImagePlugin.JpegImageFile(file_name)
    qtable = JpegImagePlugin.convert_dict_qtables(jpg.quantization)
    Y_qtable = qtable[0]
    Y_qtable_2d = np.zeros((8, 8))

    qtable_idx = 0
    for i in range(0, 8):
        for j in range(0, 8):
            Y_qtable_2d[i, j] = Y_qtable[qtable_idx]
            qtable_idx = qtable_idx + 1

    return Y_qtable_2d

class SingleDoubleDataset(torch.utils.data.Dataset): 
    def __init__(self):
        train_dir_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']

        file_list_double = list()
        for dir_idx in train_dir_list:
            file_list_double_ = glob.glob('/home/jspark/Project/data_custom/jpeg/double/'+dir_idx+'/*.jpg')
            file_list_double.extend(file_list_double_)

        double_label = [1]*len(file_list_double)
        
        file_list_single = list()
        for dir_idx in train_dir_list:
            file_list_single_ = glob.glob('/home/jspark/Project/data_custom/jpeg/single/'+dir_idx+'/*.jpg')
            file_list_single.extend(file_list_single_)            

        single_label = [0]*len(file_list_single)
        self.file_list = file_list_double + file_list_single
        self.label_list = double_label + single_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        im = Image.open(self.file_list[idx])
        #im = im.convert('RGB')
        #im = im.resize((256,256))
        img = np.array(im)
        #img = img/255.0
        #img = np.reshape(img, (3, 256,256))
        img = np.transpose(img,(2,0,1))

        q_table = read_q_table(self.file_list[idx])
        q_vector = q_table.flatten()
        label = self.label_list[idx]

        item = (img, q_vector, label)
        return item

class SingleDoubleDatasetValid(SingleDoubleDataset):
    def __init__(self):
        valid_dir_list = ['19','20']

        file_list_double = list()
        for dir_idx in valid_dir_list:
            file_list_double_ = glob.glob('/home/jspark/Project/data_custom/jpeg/double/'+dir_idx+'/*.jpg')
            file_list_double.extend(file_list_double_)

        double_label = [1]*len(file_list_double)
        
        file_list_single = list()
        for dir_idx in valid_dir_list:
            file_list_single_ = glob.glob('/home/jspark/Project/data_custom/jpeg/single/'+dir_idx+'/*.jpg')
            file_list_single.extend(file_list_single_)            

        single_label = [0]*len(file_list_single)
        self.file_list = file_list_double + file_list_single
        self.label_list = double_label + single_label

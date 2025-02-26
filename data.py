import os
import torch
import glob
import numpy as np
from PIL import Image

def read_q_table(image_path):
    jpg = Image.open(image_path)
    if not jpg.mode == 'RGB':
        jpg = jpg.convert('RGB')
    
    # Get the quantization tables directly
    qtables = jpg.quantization
    if qtables:
        # Convert quantization table format
        # Usually we want the luminance table (index 0)
        if isinstance(qtables, dict):
            qtable = qtables[0]  # Get luminance table
            qtable = np.array(qtable, dtype=np.float32)
        else:
            qtable = np.array(qtables[0], dtype=np.float32)  # Get luminance table
            
        return qtable.reshape((8, 8))
    else:
        raise ValueError("No quantization tables found in JPEG image")

class SingleDoubleDataset(torch.utils.data.Dataset): 
    def __init__(self, data_path):
        train_dir_list = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18']

        file_list_double = list()
        for dir_idx in train_dir_list:
            double_path = os.path.join(data_path, 'double')
            file_list_double_ = glob.glob(double_path+'/'+dir_idx+'/*.jpg')
            file_list_double.extend(file_list_double_)

        double_label = [1]*len(file_list_double)
        
        file_list_single = list()
        for dir_idx in train_dir_list:
            single_path = os.path.join(data_path, 'single')
            file_list_single_ = glob.glob(single_path+'/'+dir_idx+'/*.jpg')
            file_list_single.extend(file_list_single_)            

        single_label = [0]*len(file_list_single)
        self.file_list = file_list_double + file_list_single
        self.label_list = double_label + single_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        im = Image.open(self.file_list[idx])
        im = im.convert('YCbCr')
        #im = im.resize((256,256))
        Y = np.array(im)[:,:,0]
        #img = img/255.0
        #img = np.reshape(img, (3, 256,256))
        #img = np.transpose(img,(2,0,1))

        q_table = read_q_table(self.file_list[idx])
        q_vector = q_table.flatten()
        label = self.label_list[idx]

        item = (Y, q_vector, label)
        return item

class SingleDoubleDatasetValid(SingleDoubleDataset):
    def __init__(self, data_path):
        valid_dir_list = ['19','20']

        file_list_double = list()
        for dir_idx in valid_dir_list:
            double_path = os.path.join(data_path, 'double')
            file_list_double_ = glob.glob(double_path+'/'+dir_idx+'/*.jpg')
            file_list_double.extend(file_list_double_)

        double_label = [1]*len(file_list_double)
        
        file_list_single = list()
        for dir_idx in valid_dir_list:
            single_path = os.path.join(data_path, 'single')
            file_list_single_ = glob.glob(single_path+'/'+dir_idx+'/*.jpg')
            file_list_single.extend(file_list_single_)            

        single_label = [0]*len(file_list_single)
        self.file_list = file_list_double + file_list_single
        self.label_list = double_label + single_label

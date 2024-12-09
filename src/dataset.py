import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from src.utils import *

class DatasetForImageReader(Dataset):
    def __init__(
        self, 
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        data_type:str = "train"
    ):
        self.data_type = data_type
        if data_type == "train":
            self.use_image_folder = train_image_folder
        elif data_type == "test":
            self.use_image_folder = test_image_folder
        elif data_type == "valid":
            self.use_image_folder = valid_image_folder

        image_data = []
        files = os.listdir(self.use_image_folder)
        files = [f for f in files if os.path.isfile(os.path.join(self.use_image_folder, f))]
        files = sorted(files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

        for filename in files:
            img_file_path = os.path.join(self.use_image_folder, filename)
            image_data.append(cv2.imread(img_file_path))
        
        a = np.array(image_data)
        self.dataset = np.transpose(np.array(image_data), (0, 3, 1, 2))
        self.dataset = self.dataset / 255.0
        self.image_height = self.dataset.shape[2]
        self.image_weight = self.dataset.shape[3]
        self.data_type = data_type
        self.total_samples = len(self.dataset)

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        output = {}
        output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        return output
    
    def collate_fn(self,batch):
        return recursive_collate_fn(batch)

class DatasetForObjectDetect(DatasetForImageReader):
    def __init__()
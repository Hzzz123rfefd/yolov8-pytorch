import json
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
        target_width: int,
        target_height: int,
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        data_type:str = "train"
    ):
        self.data_type = data_type
        self.target_width = target_width
        self.target_height = target_height
        
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
            image_data.append(self.read_image(img_file_path))
        
        self.dataset = np.transpose(np.array(image_data), (0, 3, 1, 2))
        self.dataset = self.dataset / 255.0
        self.total_samples = len(self.dataset)

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        output = {}
        output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        return output
    
    def collate_fn(self,batch):
        return recursive_collate_fn(batch)
    
    def read_image(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        return resized_image

class DatasetForObjectDetect(DatasetForImageReader):
    def __init__( 
        self, 
        target_width:int,
        target_height:int,
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        train_annotation_folder:str = None,
        test_annotation_folder:str = None,
        valid_annotation_folder:str = None,
        data_type:str = "train"
    ):
        super().__init__(target_width, target_height, train_image_folder, test_image_folder, valid_image_folder, data_type)
        if self.data_type == "train":
            self.use_annotation_folder = train_annotation_folder
        elif self.data_type == "test":
            self.use_annotation_folder = test_annotation_folder
        elif self.data_type == "valid":
            self.use_annotation_folder = valid_annotation_folder
            
        self.annotations_data = []
        files = os.listdir(self.use_annotation_folder)
        files = [f for f in files if os.path.isfile(os.path.join(self.use_annotation_folder, f))]
        files = sorted(files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
        for filename in files:
            annotation_file_path = os.path.join(self.use_annotation_folder, filename)
            self.annotations_data.append(self.read_objects(annotation_file_path))

    def read_objects(self, annotation_path):
        target_width = 640
        target_height = 640
        
        with open(annotation_path, 'r', encoding='utf-8') as file:
            annotation_json = json.load(file)
            
        # calculate scale rate
        img_width = annotation_json["image_size"][0]
        img_height = annotation_json["image_size"][1]
        scale_x = target_width / img_width
        scale_y = target_height / img_height
        
        objects = []
        for object in annotation_json["objects"]:
            annotation = np.zeros(5)
            bbox = object["bbox"]
            category = object["category_id"]
            xmin = (bbox[0] * scale_x) 
            ymin = (bbox[1] * scale_y) 
            xmax = ((bbox[0] + bbox[2]) * scale_x) 
            ymax = ((bbox[1] + bbox[3]) * scale_y) 
            annotation[0:4] = [xmin, ymin, xmax, ymax]
            annotation[4] = category
            objects.append(annotation)
        objects = np.vstack(objects)
        return objects
    
    def __getitem__(self, idx):
        output = {}
        output["image"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        output["annotation"] = self.annotations_data[idx]
        return output
    
    def collate_fn(self,batch):
        max_object_num = max([output["annotation"].shape[0] for output in batch])
        
        for item in batch:
            gt_bboxes = torch.zeros((max_object_num, 4), dtype = torch.float32)
            gt_labels = torch.zeros((max_object_num, 1), dtype = torch.int64)
            gt_mask = torch.zeros((max_object_num, 1), dtype = torch.float32)
            
            object_num = item['annotation'].shape[0]
            gt_bboxes[:object_num, :] = torch.tensor(item['annotation'])[:, 0:4]
            gt_labels[:object_num, 0] = torch.tensor(item['annotation'])[:,4]
            gt_mask[:object_num,:] = 1
            item["gt_bboxes"] = gt_bboxes
            item["gt_labels"] = gt_labels
            item["gt_mask"] = gt_mask
            del item["annotation"]
        return recursive_collate_fn(batch)
        
import argparse
import sys
import os
import cv2
sys.path.append(os.getcwd())

from src.utils import *
from src import models
from src.model import *



def main(args):
    """ get model """
    config = load_config(args.model_cof)
    model = models[config["model_type"]](**config["model"])
    model.load_pretrained(config["logging"]["save_dir"])  
    
    """ inference """
    detect = model.inference(image_path = args.image_path, confidence = args.confidence)
    # cv2.imwrite(args.output_path, detect)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path',type=str, default= "data/images/train/00001.jpg")
    parser.add_argument('--model_cof',type=str, default="config/yolov8.yml")
    parser.add_argument('--confidence',type=float, default=0.9)
    parser.add_argument('--output_path',type=str, default="ret.png")
    args = parser.parse_args()
    main(args)
from src.model import ObjectDetect
from src.dataset import DatasetForObjectDetect


datasets = {
   "object_detect":DatasetForObjectDetect
}

models = {
    "yolov8":ObjectDetect,
}
# yolov8-pytorch
Reproduce YOLOV8 PyTorch and run it on your own dataset

## Installation
Install the packages required for development.
```bash
conda create -n yolov8 python=3.10
conda activate yolov8
git clone https://github.com/Hzzz123rfefd/yolov8-pytorch.git
cd yolov8-pytorch
pip install -r requirements.txt
```


## Usage
### Dataset
Firstly, you can download the coco2014 dataset  
[coco2014](https://cocodataset.org/#download)
download train2014、val2014 and their annotations,put them into `datasets/coco2014`
your directory structure should be:
- yolov8-pytorch/
  - datasets/
    - coco2014/
      - annotations
        - instances_train2014.json
        - instances_val2014.json
      - train2014
        - imgs
      - val2014
        - imgs

Then, you can process coco2014 data with following script:
```bash
python datasets/coco2014/process.py --output_dir coco2014_train/
```

No matter what dataset you use, please convert it to the required dataset format for this project, as follows (you can also view it in `data/annotations/train/00001.json` and `data/images/train/00001.jpg`)
```json
{"image_path": "/path","image_size": [454,640],"objects": [{"id": 1,"category_id": 18,"bbox": [100.0,150.0,50.0,80.0]}]}
```

### Trainning
An examplary training script is provided in `train.py`.
You can adjust the model parameters in `config/yolov8.yml`
```bash
python train.py --model_config_path config/yolov8.yml
```

### Inference
Once you have trained your model, you can use the following script to detect:
```bash
python example/inference.py --image_path {yout image path} --model_config_path config/density.yml --confidence 0.95 --output_path {density map path}
```

## Related links
 * yolo: https://github.com/ultralytics/ultralytics
 * coco Dataset: https://cocodataset.org/


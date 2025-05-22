## 利用官网代码进行模型训练推理
### 模型训练

#### 数据准备
需要训练集图片和测试集图片，训练集标签和测试集标签，分别按照下面的形式组织
- dataset_train/
  - images/
    - train/    
      - 1.jpg
      - 2.jpg
      ......
    - test/
      - 1.jpg
      - 2.jpg
      ......
  - labels/
    - train/
      - 1.txt
      - 2.txt
      ......
    - test/
      - 1.txt
      - 2.txt
      ......

*.txt 格式：
类别 x y w h
0 0.24158653846153846 0.7109375 0.13701923076923078 0.234375
#### 模型配置
dataset.yaml:

```yaml
train: path/dataset_train/images/train  # train images
val: path/dataset_train/images/test  # val images
# test: C:/Users/10198/Desktop/project/datasets/food_train/images/train  # test images （可选）

# number of classes
nc: 1

# Classes，填写类别名
names:
  0: food 
```

#### 模型训练 + 推理
```bash
python train.py
```

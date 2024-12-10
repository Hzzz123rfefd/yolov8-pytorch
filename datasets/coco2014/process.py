import argparse
import json
from collections import defaultdict
from shutil import copyfile
import sys
import os
sys.path.append(os.getcwd())


# # 输入文件路径和输出文件路径
# annotation_file = "datasets/coco2014/annotations/instances_val2014.json"
# output_annotation_dir = "outdir/annotations/train"
# image_dir = "datasets/coco2014/train2014"
# output_image_dir = "outdir/images/train"


# # 读取 COCO 数据集的 JSON 文件
# def load_json(file_path):
#     with open(file_path, "r") as f:
#         return json.load(f)

# # 保存单张图片的 JSON 文件
# def save_json(data, file_path):
#     with open(file_path, "w") as f:
#         json.dump(data, f, indent=4)

# # 主处理函数
# def process_coco_annotations(input_file, output_dir):
#     data = load_json(input_file)

#     # 构建 image_id 到图像信息的映射
#     images = {img["id"]: img for img in data["images"]}

#     # 按 image_id 分组 annotations
#     annotations_by_image = defaultdict(list)
#     for ann in data["annotations"]:
#         annotations_by_image[ann["image_id"]].append(ann)

#     # 为每张图片生成 JSON 文件
#     for image_id, image_info in images.items():
#         image_path = image_info["file_name"]
#         image_size = [image_info["height"], image_info["width"]]

#         objects = []
#         id = 0
#         for ann in annotations_by_image.get(image_id, []):
#             obj = {
#                 "id": id,
#                 "category_id": ann["category_id"],
#                 "bbox": ann["bbox"]
#             }
#             objects.append(obj)
#             id = id + 1

#         # 生成单张图片的 JSON 数据
#         image_data = {
#             "image_path": image_path,
#             "image_size": image_size,
#             "objects": objects
#         }


        
#         if os.path.exists(source_image_path):
#             output_file = f"{output_dir}{image_path.replace('.jpg', '.json')}"
#             save_json(image_data, output_file)
#             source_image_path = os.path.join(image_dir, image_path)
#             target_image_path = os.path.join(output_image_dir, image_path)
#         else:
#             print(f"Warning: Image file {source_image_path} not found.")

# # 运行
# process_coco_annotations(annotation_file, output_annotations_dir, output_images_dir)

def process_coco(annotation_file, images_dir, output_annotations_dir, output_images_dir, max_image_num = -1):
    os.makedirs(output_annotations_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)
    # get annotation information
    with open(annotation_file, "r") as f:
        data = json.load(f)

    # build mapping of image_id to image
    images = {img["id"]: img for img in data["images"]}

    # group by image_id
    annotations_by_image = defaultdict(list)
    for ann in data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    # generate json file for each image
    image_num = 0
    for image_id, image_info in images.items():
        image_path = image_info["file_name"]
        image_size = [image_info["width"], image_info["height"]]

        objects = []
        id = 0
        for ann in annotations_by_image.get(image_id, []):
            obj = {
                "id": id,
                "category_id": ann["category_id"],
                "bbox": ann["bbox"]
            }
            objects.append(obj)
            id = id + 1

        image_data = {
            "image_path": image_path,
            "image_size": image_size,
            "objects": objects
        }
        
        source_image_path = os.path.join(images_dir, image_path)
        if os.path.exists(source_image_path):
            output_file = f"{output_annotations_dir}{image_path.replace('.jpg', '.json')}"
            with open(output_file, "w") as f:
                json.dump(image_data, f, indent=4)
            target_image_path = os.path.join(output_images_dir, image_path)
            copyfile(source_image_path, target_image_path)
            image_num = image_num + 1
        else:
            print(f"Warning: Image file {source_image_path} not found.")
        if image_num == max_image_num:
            break

def main(args):
    train_annotation_file = args.dataset_dir + "annotations/instances_train2014.json"
    val_annotation_file = args.dataset_dir + "annotations/instances_val2014.json"
    process_coco(
        annotation_file = train_annotation_file, 
        images_dir = args.dataset_dir + "train2014", 
        output_annotations_dir = args.output_dir + "annotations/train/",
        output_images_dir = args.output_dir + "images/train/",
        max_image_num = 1024
    )
    process_coco(
        annotation_file = val_annotation_file, 
        images_dir = args.dataset_dir + "val2014", 
        output_annotations_dir = args.output_dir + "annotations/test/",
        output_images_dir = args.output_dir + "images/test/",
        max_image_num = 256
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",type=str,default = "datasets/coco2014/")
    parser.add_argument("--output_dir",type=str,default = "coco2014_train/")
    args = parser.parse_args()
    main(args)
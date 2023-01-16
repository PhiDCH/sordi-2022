import json 
import os 
from copy import deepcopy


path = "data/stage2/datasets/Hackathon_Stage2/Evaluation_set/dataset/objectclasses.json"
with open(path, mode='rb') as f:
    category = json.load(f)

annot = {}
annot['info'] = {'description': 'stage2'}
cate = []
for cls in category:
    cate.append({"id": cls['Id'], "name": cls['Name']})
annot["categories"] = cate

annot["categories"] = [{
            "id": 1008,
            "name": "klt_box_empty"
        }]


def get_annot(gt_folder: str, img_folder:str, save_json:str, img_id_add=0):
    sample_img = {
        "id": 0,
        "width": 1280,
        "height": 720,
        "filename": "data/stage2/datasets/Hackathon_Stage2/Evaluation_set/dataset/images/0.jpg"
    }
    images = []
    annotations = []
    def convert_ann(ann: dict):
        ann_ = {"iscrowd": 0}
        ann_['id'] = ann["Id"]
        ann_['category_id'] = ann["ObjectClassId"]
        ann_["bbox"] = [ann["Left"], ann["Top"], ann["Right"], ann["Bottom"]]
        ann_["area"] = (ann["Right"] - ann["Left"]) * (ann["Bottom"] - ann["Top"])
        return ann_

    for json_file in os.listdir(gt_folder)[:100]:
        i = int(json_file.split('.')[0])
        json_file = gt_folder +'/'+ json_file
        img_file = img_folder + f'/{i}.jpg'
        img = deepcopy(sample_img)
        img['id'] = i
        img['filename'] = img_file
        images.append(img)

        with open(json_file, 'rb') as f:
            ann = json.load(f)
        
        for bb in ann:
            # filter 
            if bb["ObjectClassId"] == 1008:
                ann_ = convert_ann(bb)
                ann_['image_id'] = i + img_id_add
                annotations.append(ann_)


    annot['images'] = images
    annot['annotations'] = annotations
    print(f"num of images {save_json} {len(annot['images'])}")

    with open(save_json, 'w') as f:
        json.dump(annot, f, indent=4)


if __name__=="__main__":
    gt_folder = "data/stage2/datasets/Hackathon_Stage2/Evaluation_set/dataset/labels/json"
    img_folder = "data/stage2/datasets/Hackathon_Stage2/Evaluation_set/dataset/images"
    save_json = "data/stage2/annotations/val-1008.json"

    get_annot(gt_folder, img_folder, save_json)

    gt_folder = "data/stage2/datasets/Hackathon_Stage2/Training_Dataset/Regensburg_Plant/labels/json"
    img_folder = "data/stage2/datasets/Hackathon_Stage2/Training_Dataset/Regensburg_Plant/images"
    save_json = "data/stage2/annotations/train-plant-1008.json"

    get_annot(gt_folder=gt_folder, img_folder=img_folder, save_json=save_json, img_id_add=10000)
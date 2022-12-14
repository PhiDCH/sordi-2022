#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np

import seaborn as sns

# json_file = "data/SORDI/annotations/sordi-non-single-asserts-train500.json"
json_file = "data/SORDI/annotations/sordi-non-single-asserts-train5000.json"
coco = COCO(json_file)
ids = coco.getImgIds()


with open(json_file, 'r') as f:
    datasets = json.load(f)
stat = {}
for cat in datasets['categories']:
    stat[cat['id']] = {'name': cat['name'], 'h/w': [], 'area/imgSize': []}
stat

from pycocotools.coco import COCO
from tqdm import tqdm

area720 = 1280*720
area360 = 640*360

for id in tqdm(ids):
    img = coco.loadImgs(id)
    if img[0]['width'] == 640:
        img_area = 640*360
    else: 
        img_area = 1280*720

    annots = coco.loadAnns(coco.getAnnIds(id))
    for annot in annots:
        _,_,w,h = annot['bbox']
        # stat[annot['category_id']]['h/w'].append(h/w)
        area = annot['area']
        stat[annot['category_id']]['area/imgSize'].append(area/img_area)


def save_plot(stat: dict, folder: str = 'data/stat/area'):
    x = stat['area/imgSize']
    sns.histplot(x, stat='probability', bins=30)
    plt.axvline(x=np.quantile(x,0.05), color='red') 
    plt.axvline(x=np.quantile(x, 0.95), color='green')
    name = stat['name']
    plt.savefig(f'{folder}/{name}.png')

save_plot(stat[5010])

# x = np.array(stat[5010]['h/w'])
# x = np.array(stat[5010]['area/imgSize'])
# # print(np.quantile(x,0.75))
# plt.subplot()
# sns.histplot(x, stat='probability', bins=30)
# plt.axvline(x=np.quantile(x,0.05), color='red')
# plt.axvline(x=np.quantile(x, 0.95), color='green')
# # plt.axvline(x=np.mean(x), color='red')
# # plt.axvline(x=np.median(x), color='green')
# # plt.savefig('a.png')
# plt.show()

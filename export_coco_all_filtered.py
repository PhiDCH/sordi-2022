import sqlite3
from sqlite3 import Error
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from copy import deepcopy

def create_sub_mask_annotation(image_id, category_id, annotation_id, bbox, is_crowd, area):
    annotation = {
        'iscrowd': is_crowd,  # dont need
        'image_id': image_id,  # pointing to an image entry (see below)
        # that is the label of the object, see id field from "category" below
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,    # the bbox as 4-tuple
        'area': area
    }
    return annotation


def create_image_entry(image_id, file_name, width, height):
    image_entry = {
        'id': image_id,   # unique, annotations "image_id" field are pointing to this one
        'width': width,   # framesize, annotations are in pixelcoords
        'height': height,
        'file_name': file_name
    }
    return image_entry


def create_category_entry(obj_id, obj_name):
    category_entry = {
        'id': obj_id,
        'name': obj_name
    }
    return category_entry


def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn


def plot_bar(stat, title: str = 'template'):
    name = list(stat.keys())
    price = list(stat.values())
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(name, price)

    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Add Plot Title
    ax.set_title(title,
                 loc='left', )

    # Add Text watermark
    fig.text(0.9, 0.15, 'phidch', fontsize=12,
             color='grey', ha='right', va='bottom',
             alpha=0.7)

    # Show Plot
    plt.show()


def main():
    conn = create_connection(
        '../SORDI-Data-Pipeline-Reader/SORDI.sqlite')

    sql = '''select rowid, * from frame'''
    cur = conn.cursor()
    cur.execute(sql)

    annotation_id = 0
    annotations_train = []
    images_train = []
    category = {}

    stat = {}
    stat_json = {'num_img': 0}
    with open('data/eval/objectclasses.json', 'r') as f:
        meta_data = json.load(f)
        for meta in meta_data:
            stat[meta['Id']] = {'name': meta['Name'], 'area/imgSize': []}
    tmp_stat = deepcopy(stat)
    # stat_train = stat.copy()

    max_num_img = 1000
    i = 0

    filter0 = None
    with open('data/stat/industrial-filtered/stat.json', 'r') as f:
        filter0 = json.load(f)
    filter1 = None
    with open('data/stat/plant-filtered/stat.json', 'r') as f:
        filter1 = json.load(f)
    metadata = {}
    with open('data/SORDI/config.json', 'r') as f:
        datasets = json.load(f)
    for dataset in datasets:
        datasetID = int(dataset['dataset_id'][-2:])
        metadata[datasetID] = {}
        if datasetID in [22,23]:
            metadata[datasetID]['volume'] = filter0['num_img']
        elif datasetID in [24,25]:
            metadata[datasetID]['volume'] = filter1['num_img']
        else:
            metadata[datasetID]['volume'] = dataset['dataset_size']

    with open('data/stat/config.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    train_volume = 40000
    train_single_volume = 0.1*train_volume
    train_plant_volume = 0.6*train_volume
    train_industrial_volume = 0.3*train_volume
    val_volume = 10000
    val_plant_volume = 0.7*val_volume
    val_industrial_volume = 0.3*val_volume
    print(f'train_single_volume {train_single_volume}')
    print(f'train_plant_volume {train_plant_volume}')
    print(f'train_industrial_volume {train_industrial_volume}')
    print(f'val_plant_volume {val_plant_volume}')
    print(f'val_industrial_volume {val_industrial_volume}')

    
    # for r in cur:
    #     # i += 1
    #     # if i > max_num_img:
    #     #     break

    #     (image_id, fname, label_json, W, H, uncertainty) = r
    #     fname = '/'.join([r for r in fname.split('/')[-4:]])

    #     split  = fname.split('/')[-3].split('_')[2]
    #     if split in ['h4024','h4025']:
    #         images_train.append(create_image_entry(image_id, fname, W, H))
    #         img_area = 921600 if W == 1280 else 230400
        
    #         get_img = True
    #         tmp = deepcopy(tmp_stat)
    #         for lj in json.loads(label_json):
    #             id_cls = lj['ObjectClassId']
    #             (l, t, r, b) = (lj['Left'], lj['Top'], lj['Right'], lj['Bottom'])
    #             w, h = r-l, b-t
    #             area = w*h

    #             if w > 3 and h > 3:
    #                 ratio = area/img_area
    #                 if filter:
    #                     min_ratio, max_ratio = filter['data'][str(id_cls)]['minmax']
    #                     min_ratio = min_ratio if min_ratio else 100/img_area
    #                     max_ratio = max_ratio if max_ratio else filter1['data'][str(id_cls)]['qual95'][1]
    #                     if ratio > min_ratio:
    #                         tmp[id_cls]['area/imgSize'].append(ratio)
    #                     if ratio > max_ratio:
    #                         get_img = False
    #                         break
    #                 else:
    #                     tmp[id_cls]['area/imgSize'].append(ratio)


    #             # bbox = (l, t, r-l, b-t)
    #             # if area > 0 and w > 3 and h > 3:
    #             #     annotations_train.append(create_sub_mask_annotation(
    #             #         image_id, lj['ObjectClassId'], annotation_id, bbox, 0, area))
    #             #     annotation_id += 1
    #             #     if lj['ObjectClassId'] in category:  # category.keys()
    #             #         pass
    #             #     else:
    #             #         category[lj['ObjectClassId']] = lj['ObjectClassName']
    #             #     stat_train[lj['ObjectClassName']] += 1

    #         if get_img: 
    #             stat_json['num_img'] += 1
    #             for k in stat:
    #                 stat[k]['area/imgSize'].extend(tmp[k]['area/imgSize'])
        
    #     # else: 
    #     #     print(split)
    #     #     break

    # sum_stat = {}
    # for k in stat.keys():
    #     name = stat[k]['name']
    #     data = stat[k]['area/imgSize']
    #     num_bbox = len(data)
    #     if data:
    #         qual95 = (np.quantile(data, 0.05), np.quantile(data, 0.95))
    #         mean = np.mean(data)
    #         minmax = (np.min(data), np.max(data))
    #         print(f'{name} qual95 {qual95} minmax {minmax}')
    #     else: 
    #         print(f'no instance of {name}')
    #         qual95 = (0,0)
    #         mean = 0
    #         minmax = (0,0)

    #     sum_stat[k] = {}
    #     sum_stat[k]['name'] = name
    #     sum_stat[k]['num_bbox'] = num_bbox
    #     sum_stat[k]['qual95'] = qual95
    #     sum_stat[k]['mean'] = mean
    #     sum_stat[k]['minmax'] = minmax 


    # stat_json['data'] = sum_stat

    # save_folder = ''
    # if filter:
    #     save_folder = 'data/stat/plant-filtered'
    # else:
    #     save_folder = 'data/stat/plant'



    # save_json(stat_json, save_folder)

    # # for k in category:
    # #     categories.append(create_category_entry(k, category[k]))

    # # save_dict_train = {'info': {"description": "SORDI 2022"}, 'categories': categories, 'annotations': annotations_train, 'images': images_train}
    # # with open('data/SORDI/annotations/sordi-non-single-asserts-train5000.json', 'w') as f:
    # #     json.dump(save_dict_train, f, indent=4)

    # for k in stat.keys():
    #     save_plot(stat[k], save_folder)


def save_json(stat: dict, folder: str = 'data/stat/area'):
    with open(f'{folder}/stat.json', 'w') as f:
        json.dump(stat, f, indent=4)

def save_plot(stat: dict, folder: str = 'data/stat/area'):
    x = stat['area/imgSize']
    if x:
        sns.histplot(x, stat='probability', bins=100)
        qual95 = (np.quantile(x, 0.05), np.quantile(x, 0.95))
        plt.axvline(qual95[0], color='red') 
        plt.axvline(qual95[1], color='green')
        name = stat['name']
        plt.savefig(f'{folder}/{name}.png')
        plt.close()


if __name__ == '__main__':
    main()

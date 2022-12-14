import sqlite3
from sqlite3 import Error
import json
import numpy as np
from matplotlib import pyplot as plt 


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

def plot_bar(stat, title:str='template'):
    name = list(stat.keys())
    price = list(stat.values())
    # Figure Size
    fig, ax = plt.subplots(figsize =(16, 9))
    
    # Horizontal Bar Plot
    ax.barh(name, price)
    
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad = 5)
    ax.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    ax.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.2)
    
    # Show top values
    ax.invert_yaxis()
    
    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width()+0.2, i.get_y()+0.5,
                str(round((i.get_width()), 2)),
                fontsize = 10, fontweight ='bold',
                color ='grey')
    
    # Add Plot Title
    ax.set_title(title,
                loc ='left', )
    
    # Add Text watermark
    fig.text(0.9, 0.15, 'phidch', fontsize = 12,
            color ='grey', ha ='right', va ='bottom',
            alpha = 0.7)
    
    # Show Plot
    plt.show()

def main():
    conn = create_connection('/home/phidch/Downloads/phi_ws/src/sordi-2022/src/SORDI-Data-Pipeline-Reader/SORDI-non-single-asserts.sqlite')
    # conn = create_connection('/home/robotic/Downloads/phidch_ws/src/bmw-lab/scripts/SORDI-Data-Pipeline-Reader/SORDI-single-asserts.sqlite')

    sql = '''select rowid, * from frame'''
    cur = conn.cursor()
    cur.execute(sql)
    
    annotation_id = 0
    annotations_train = []
    images_train = []
    annotations_val = []
    images_val = []
    categories = []
    category = {}
    
    stat = {}
    with open('/home/phidch/Downloads/phi_ws/src/sordi-2022/src/data/eval/objectclasses.json', 'r') as f:
        meta_data = json.load(f)
        for meta in meta_data:
            stat[meta['Name']] = 0
    stat_train = stat.copy()
    stat_val = stat.copy()

    max_num_img_each_split_train = 5000
    max_num_img_each_split_val = 1000
    num_img_echh_split = {}
    for r in cur:
        (image_id, fname, label_json, w, h, uncertainty) = r  

        split = fname.split('/')[-3]
        if split not in num_img_echh_split.keys():
            num_img_echh_split[split] = 0
        num_img_echh_split[split] += 1

        fname = '/'.join([r for r in fname.split('/')[-4:]])

        if num_img_echh_split[split] <= max_num_img_each_split_train:
            images_train.append(create_image_entry(image_id, fname, w, h))
            for lj in json.loads(label_json):
                (l,t,r,b) = (lj['Left'], lj['Top'], lj['Right'], lj['Bottom'])
                w,h = r-l, b-t
                bbox = (l, t, r-l, b-t)
                area = (r-l)*(b-t)
                if area>0 and w>3 and h>3:
                    annotations_train.append(create_sub_mask_annotation(image_id, lj['ObjectClassId'], annotation_id , bbox, 0, area))
                    annotation_id += 1
                    if lj['ObjectClassId'] in category: # category.keys()
                        pass
                    else:
                        category[lj['ObjectClassId']] = lj['ObjectClassName']
                    stat_train[lj['ObjectClassName']] += 1

        elif num_img_echh_split[split] <= max_num_img_each_split_train+max_num_img_each_split_val:
            images_val.append(create_image_entry(image_id, fname, w, h))
            for lj in json.loads(label_json):
                (l,t,r,b) = (lj['Left'], lj['Top'], lj['Right'], lj['Bottom'])
                w,h = r-l, b-t
                bbox = (l, t, r-l, b-t)
                area = (r-l)*(b-t)
                if area>0 and w>3 and h>3:
                    annotations_val.append(create_sub_mask_annotation(image_id, lj['ObjectClassId'], annotation_id , bbox, 0, area))
                    annotation_id += 1
                    if lj['ObjectClassId'] in category: # category.keys()
                        pass
                    else:
                        category[lj['ObjectClassId']] = lj['ObjectClassName']
                    stat_val[lj['ObjectClassName']] += 1
        else: 
            pass
           

    for k in category:
        categories.append(create_category_entry(k, category[k]))

    save_dict_train = {'info': {"description": "SORDI 2022"}, 'categories': categories, 'annotations': annotations_train, 'images': images_train}
    save_dict_val = {'info': {"description": "SORDI 2022"}, 'categories': categories, 'annotations': annotations_val, 'images': images_val}
    with open('data/SORDI/annotations/sordi-non-single-asserts-train5000.json', 'w') as f:
        json.dump(save_dict_train, f, indent=4)
    with open('data/SORDI/annotations/sordi-non-single-asserts-val5000.json', 'w') as f:
        json.dump(save_dict_val, f, indent=4)

    # with open('data/SORDI/annotations/sordi-single-asserts-train100.json', 'w') as f:
    #     json.dump(save_dict_train, f, indent=4)
    # with open('data/SORDI/annotations/sordi-single-asserts-val100.json', 'w') as f:
    #     json.dump(save_dict_val, f, indent=4)

    plot_bar(stat_train, 'stat_train')
    plot_bar(stat_val, 'stat_val')

if __name__=='__main__':
    main()
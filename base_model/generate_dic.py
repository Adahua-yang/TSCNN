'''
generate dictionary of train and valid dataset:

train & valid table columns:
'person_id', 'cam_id', 'img_id', 'img_path', 'flow_path'
 int,         str,      int,      str,        str
'''
import os
import json
import pandas as pd
import numpy as np


def generate_dic(config):
    dataset_name = config['use_dataset']
    rgb_root = config[dataset_name + '_rgb']
    flow_root = config[dataset_name + '_flow']
    
    table = []
    cam_list = os.listdir(rgb_root)
    cam1_root = os.path.join(rgb_root, cam_list[0])
    cam2_root = os.path.join(rgb_root, cam_list[1])
    cam1_person_list = os.listdir(cam1_root)
    cam2_person_list = os.listdir(cam2_root)
    person_list = list(set(cam1_person_list) & set(cam2_person_list))

    # use only 200 person in PRID2011 dataset

    if dataset_name == 'PRID2011':
        person_list = sorted(person_list, key=lambda x: int(x[-3:]))[:200]

    for cam_id in cam_list:
        # /cam1/
        rgb_cam_root = os.path.join(rgb_root, cam_id)
        flow_cam_root = os.path.join(flow_root, cam_id)

        for person_id in person_list:
            # /cam1/person001
            rgb_person_root = os.path.join(rgb_cam_root, person_id)
            flow_person_root = os.path.join(flow_cam_root, person_id)

            img_list = list(set(os.listdir(rgb_person_root)) & set(os.listdir(flow_person_root)))
            for img_id in img_list:
                rgb_path = os.path.join(rgb_person_root, img_id)
                flow_path = os.path.join(flow_person_root, img_id)
                if os.path.isfile(flow_path) is True:
                    table.append([cam_id, person_id, int(img_id[-9:-4]), rgb_path, flow_path])
    
    dataset_table = pd.DataFrame(table, columns=['cam_id', 'person_id', 'img_id', 'rgb_path', 'flow_path'])
    # split dataset to train and valid
    person_list = dataset_table.loc[:, 'person_id'].unique()
    train_person = person_list[:int(0.5 * len(person_list))]
    valid_person = person_list[int(0.5 * len(person_list)):]
    train_table = dataset_table.loc[[i in train_person for i in dataset_table['person_id'].values], :]
    valid_table = dataset_table.loc[[i in valid_person for i in dataset_table['person_id'].values], :]
    print('train_table shape=', train_table.shape)
    print('valid_table shape=', valid_table.shape)
    train_table.to_csv(config['train_dataset'], sep='#', index=False)
    valid_table.to_csv(config['valid_dataset'], sep='#', index=False)


def main():
    with open('setting.json', 'r') as fin:
        config = json.load(fin)
    config = config['generate_dic']
    if config['is_running'] is False:
        return

    # main function
    print('############generate table###########')
    generate_dic(config)


if __name__ == '__main__':
    main()
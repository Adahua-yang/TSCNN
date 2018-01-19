import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch


class datasetutil():
    def __init__(self, config):
        '''
        train/valid_dataset format: person_num * [[cam1_imgs], [cam2_imgs]]
        '''

        def preprocessRGB(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (48, 64))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            img = np.transpose(img, (2, 0, 1)).astype(np.float64)

            for i in range(3):
                v = np.sqrt(torch.var(torch.Tensor(img[i])))
                if v < 1e-6:
                    v = max(1, np.max(img[i]))
                img[i] = (img[i] - np.mean(img[i])) / np.sqrt(v)

            return img

        # note we just use part of channels of the opticalflow image.
        def preprocessFlow(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (48, 64))
            img = np.transpose(img, (2, 0, 1)).astype(np.float64)

            for i in range(0, 3):
                v = np.sqrt(torch.var(torch.Tensor(img[i])))
                if v < 1e-6:
                    v = max(1, np.max(img[i]))
                img[i] = (img[i] - np.mean(img[i])) / np.sqrt(v)

            return img

        def get_dataset(df):
            full_dataset = []
            person_list = df['person_id'].unique()
            cam_list = df['cam_id'].unique()
            # print 'person_list=', person_list
            # print 'cam_list=', cam_list
            with tqdm(total=len(person_list)) as pbar:
                for person_id in person_list:
                    # person_imgs.shape = (rgb_cam1, flow_cam1, rgb_cam2, flow_cam2) * (len1, ...) * channel * height * width
                    person_imgs = []
                    for cam_id in cam_list:
                        nowdf = df.loc[(df['person_id'] == person_id) & (df['cam_id'] == cam_id), :].sort_values('img_id')
                        person_imgs.append(np.array([preprocessRGB(line[3]) for line in nowdf.values]))
                        person_imgs.append(np.array([preprocessFlow(line[4]) for line in nowdf.values]))
                    # print('person_imgs.shape=', [mat.shape for mat in person_imgs])
                    full_dataset.append(person_imgs)
                    del person_imgs
                    pbar.update(1)
            return full_dataset

        self.config = config

        # the number of train and valid persons can also be seen in tqdm bar.
        # dataset.shape: person_num * person_imgs.shape
        df_train = pd.read_csv(config['train_dataset'], sep='#')
        self.train_dataset = get_dataset(df_train)
        df_valid = pd.read_csv(config['valid_dataset'], sep='#')
        self.valid_dataset = get_dataset(df_valid)

    @property
    def train_person_num(self):
        return len(self.train_dataset)

    @property
    def valid_person_num(self):
        return len(self.valid_dataset)

    def train_seq_len(self, idx, cam_id, reversed=False):
        cam_idx = 0 if cam_id == 0 else 2
        return len(self.train_dataset[idx][cam_idx]) if reversed is False else len(self.train_dataset[idx][cam_idx]) - 1

    def valid_seq_len(self, idx, cam_id, reversed=False):
        cam_idx = 0 if cam_id == 0 else 2
        return len(self.valid_dataset[idx][cam_idx]) if reversed is False else len(self.valid_dataset[idx][cam_idx]) - 1

    def train_input_id(self, idx, cam_id, seqlen, reversed=False):
        rgb_id, flow_id = (0, 1) if cam_id == 0 else (2, 3)

        if reversed is True:
            rgb_begin_index = max(1, int(np.random.random() * (len(self.train_dataset[idx][rgb_id]) - seqlen)))
            flow_begin_index = rgb_begin_index - 1

            rgb_select_index = list(range(rgb_begin_index, rgb_begin_index + seqlen))
            rgb_select_index.reverse()
            flow_select_index = list(range(flow_begin_index, flow_begin_index + seqlen))
            flow_select_index.reverse()

            dtrain_rgb = np.array(self.train_dataset[idx][rgb_id][rgb_select_index])
            dtrain_flow = np.array(self.train_dataset[idx][flow_id][flow_select_index])
        else:
            begin_index = max(0, int(np.random.random() * (len(self.train_dataset[idx][rgb_id]) - seqlen)))
            select_index = range(begin_index, begin_index + seqlen)

            dtrain_rgb = np.array(self.train_dataset[idx][rgb_id][select_index])
            dtrain_flow = np.array(self.train_dataset[idx][flow_id][select_index])
        # print('select rgb_idx=', rgb_select_index)
        # print('select flow_idx=', flow_select_index)
        # print('dtrain_rgb shape=', dtrain_rgb.shape)
        return dtrain_rgb, dtrain_flow

    def valid_input_id(self, idx, cam_id, seqlen, reversed=False):
        rgb_id, flow_id = (0, 1) if cam_id == 0 else (2, 3)

        if reversed is True:
            rgb_begin_index = max(1, int(np.random.random() * (len(self.valid_dataset[idx][rgb_id]) - seqlen)))
            flow_begin_index = rgb_begin_index - 1

            rgb_select_index = list(range(rgb_begin_index, rgb_begin_index + seqlen))
            rgb_select_index.reverse()
            flow_select_index = list(range(flow_begin_index, flow_begin_index + seqlen))
            flow_select_index.reverse()

            dvalid_rgb = np.array(self.valid_dataset[idx][rgb_id][rgb_select_index])
            dvalid_flow = np.array(self.valid_dataset[idx][flow_id][flow_select_index])
        else:
            begin_index = max(0, int(np.random.random() * (len(self.valid_dataset[idx][rgb_id]) - seqlen)))
            select_index = range(begin_index, begin_index + seqlen)

            dvalid_rgb = np.array(self.valid_dataset[idx][rgb_id][select_index])
            dvalid_flow = np.array(self.valid_dataset[idx][flow_id][select_index])
        # print 'dtrain shape=', dtrain.shape
        return dvalid_rgb, dvalid_flow
    
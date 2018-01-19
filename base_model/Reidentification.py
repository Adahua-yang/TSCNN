import os
import numpy as np
from tqdm import tqdm
import json
import time
import copy
import random
from model import modelUtil
from datasetutil import datasetutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reidentification():

    def __init__(self, config):
        self.config = config

        print('parameters:')
        for key in self.config:
            print('{} = {}'.format(key, self.config[key]))

        print('loading datasets...')
        self.datasetutil = datasetutil(config)
        self.train_person_num = self.datasetutil.train_person_num
        self.valid_person_num = self.datasetutil.valid_person_num
        print('train_person_num=', self.train_person_num)
        print('valid_person_num=', self.valid_person_num)
        self.config['person_num'] = self.train_person_num

        print('loading models...')
        self.modelUtil = modelUtil(self.config)

        self.bst_score = [0 for _ in range(self.valid_person_num)]


    def img_augment(self, img_rgb, img_flow, dx=None, dy=None, flip=None):
            '''
            img_ary.shape = 16(seq_len) * 6(channel) * 64(height) * 48(width)
            return.shape = 16 * 6 * 56 * 40
            '''
            dx = int(np.random.random() * 8) if dx is None else dx
            dy = int(np.random.random() * 8) if dy is None else dy
            flip = int(np.random.random() * 2) if flip is None else flip

            rgb_num, rgb_channel = len(img_rgb), len(img_rgb[0])
            flow_num, flow_channel = len(img_flow), len(img_flow[0])

            aug_rgb = np.zeros((rgb_num, rgb_channel, 56, 40))
            aug_flow = np.zeros((flow_num, flow_channel, 56, 40))

            for i in range(rgb_num):
                for j in range(rgb_channel):
                    tmp = img_rgb[i][j][dx:dx + 56, dy:dy + 40]
                    tmp = tmp - np.mean(tmp)
                    if flip is 1:
                        tmp = tmp[:, ::-1]
                    aug_rgb[i][j] = tmp
                    del tmp

            for i in range(flow_num):
                for j in range(flow_channel):
                    tmp = img_flow[i][j][dx:dx + 56, dy:dy + 40]
                    tmp = tmp - np.mean(tmp)
                    if flip is 1:
                        tmp = tmp[:, ::-1]
                    aug_flow[i][j] = tmp
                    del tmp

            return aug_rgb, aug_flow

    def save_model(self):
        self.modelUtil.save_model()

    def train_base(self):

        def fea_distance(fea_x, fea_y):
            # return 1. - np.sum(fea_x * fea_y) / (np.linalg.norm(fea_x) * np.linalg.norm(fea_y))
            return np.sqrt(np.sum((fea_x - fea_y) ** 2))

        def valid_score(dismat):
            person_num = dismat.shape[0]
            score = np.zeros(person_num)
            for i in range(person_num):
                disary = dismat[i]
                # print 'disary=', list(disary[:10])
                idx = len(np.where(disary < disary[i])[0])
                score[idx:] += 1.
            return map(int, 100. * score / self.valid_person_num)

        loss_log = []

        print('training...')
        for epoch in range(1, 1 + self.config['epoch']):

            self.modelUtil.exp_lr_scheduler(epoch)

            train_person_list = range(self.train_person_num)
            train_random_list1 = range(self.train_person_num)
            train_random_list2 = range(self.train_person_num)
            np.random.shuffle(train_person_list)
            np.random.shuffle(train_random_list1)
            np.random.shuffle(train_random_list2)

            loss_log.append(np.array([0., 0., 0., 0., 0.]))

            for i in range(self.train_person_num * 2):

                if i % 2 == 1:
                    person1_id = person2_id = train_person_list.pop()
                else:
                    person1_id = train_random_list1.pop()
                    person2_id = train_random_list2.pop()

                cam1_id = int(np.random.random() * 2)
                cam2_id = int(np.random.random() * 2)

                seqlen = min([self.datasetutil.train_seq_len(person1_id, cam1_id, reversed=self.config['reversed']), \
                    self.datasetutil.train_seq_len(person2_id, cam2_id, reversed=self.config['reversed']), self.config['max_seqlen']])

                # we decide process the sequences which length is small than max_seqlen.
                if seqlen < self.config['max_seqlen']:
                    continue
                # print 'seqlen=', seqlen
                person1_rgb, person1_flow = self.datasetutil.train_input_id(person1_id, cam1_id, seqlen, reversed=self.config['reversed'])
                person2_rgb, person2_flow = self.datasetutil.train_input_id(person2_id, cam2_id, seqlen, reversed=self.config['reversed'])
                person1_rgb, person1_flow = self.img_augment(person1_rgb, person1_flow)
                person2_rgb, person2_flow = self.img_augment(person2_rgb, person2_flow)

                loss = self.modelUtil.train_model(person1_rgb, person1_flow, person1_id, person2_rgb, person2_flow, person2_id)
                loss_log[-1] += np.array(loss)

            loss_log[-1] /= (self.train_person_num * 2)
            print('{}: epoch={}, loss={:.3f}, cls={:.3f}, hinge={:.3f}, pull={:.3f}, push={:.3f}'\
                .format(self.config['run_tag'], epoch, loss_log[-1][0], loss_log[-1][1], loss_log[-1][2], loss_log[-1][3], loss_log[-1][4]))

            if epoch % self.config['valid_epoch'] == 0:

                dismat = np.zeros((self.valid_person_num, self.valid_person_num))
                feature_ary = [[np.zeros((16, 128)), np.zeros((16, 128))] for _ in range(self.valid_person_num)]

                for shiftx in range(8):
                    shifty = shiftx
                    for flip in range(2):
                        valid_feature = []
                        for i in range(self.valid_person_num):
                            seqlen = min([self.datasetutil.valid_seq_len(i, 0, reversed=self.config['reversed']), \
                                self.datasetutil.valid_seq_len(i, 1, reversed=self.config['reversed']), self.config['max_seqlen']])
                            if seqlen < self.config['max_seqlen']:
                                valid_feature.append([np.zeros(128), np.zeros(128)])
                                continue
                            person1_rgb, person1_flow = self.datasetutil.valid_input_id(i, 0, seqlen, reversed=self.config['reversed'])
                            person2_rgb, person2_flow = self.datasetutil.valid_input_id(i, 1, seqlen, reversed=self.config['reversed'])
                            person1_rgb, person1_flow = self.img_augment(person1_rgb, person1_flow, shiftx, shifty, flip)
                            person2_rgb, person2_flow = self.img_augment(person2_rgb, person2_flow, shiftx, shifty, flip)

                            person1_feature, person2_feature = self.modelUtil.abstract_feature(person1_rgb, \
                                person1_flow, person2_rgb, person2_flow)

                            valid_feature.append([np.mean(person1_feature, 0), np.mean(person2_feature, 0)])
                            feature_ary[i][0] += person1_feature
                            feature_ary[i][1] += person2_feature

                        for i in range(self.valid_person_num):
                            for j in range(self.valid_person_num):
                                dismat[i, j] += fea_distance(valid_feature[i][0], valid_feature[j][1]) + \
                                    fea_distance(valid_feature[i][1], valid_feature[j][0])

                print('dismat=\n', dismat)
                base_score = valid_score(dismat)
                self.bst_score = [max(self.bst_score[i], base_score[i]) for i in range(self.valid_person_num)]
                print('base valid score={}'.format(' '.join(map(str, base_score[:20]))))
                print('bstb valid score={}'.format(' '.join(map(str, self.bst_score[:20]))))
                if base_score[0] == self.bst_score[0]:
                    print('save model...')
                    self.modelUtil.save_model()
                    print('save feature matrix...')
                    if os.path.isdir(self.config['run_tag']) is False:
                        os.mkdir(self.config['run_tag'])
                    for file_name in range(self.valid_person_num):
                        feature_ary[file_name][0] /= 16
                        feature_ary[file_name][1] /= 16
                        with open('{}/{}'.format(self.config['run_tag'], file_name), 'w') as fout:
                            for cam in range(2):
                                for i in range(16):
                                    for j in range(128):
                                        fout.write(str(feature_ary[file_name][cam][i][j]) + ' ')
                                fout.write('\n')
                    print('save distance matrix...')
                    dismat /= np.max(dismat)
                    with open('{}/{}'.format(self.config['run_tag'], 'dismat'), 'w') as fout:
                        for i in range(self.valid_person_num):
                            for j in range(self.valid_person_num):
                                fout.write(str(dismat[i][j]) + ' ')
                            fout.write('\n')
                    print('finished!')
        print(self.bst_score)


def main():
    with open('setting.json', 'r') as fin:
        config = json.load(fin)
    config = config['model']
    if config['is_running'] is False:
        return

    # main function
    print('############running model############')
    model = Reidentification(config)
    model.train_base()


if __name__ == '__main__':
    main()

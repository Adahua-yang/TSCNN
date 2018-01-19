import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torchvision import models, transforms
from collections import namedtuple
from model import model


class backup(Function):

    @staticmethod
    def forward(self, input1, weight, input2):
        '''
        input.shape = batch_size * rows * columns
        Y = input1 * weight * input2
        '''
        self.save_for_backward(input1, weight, input2)
        # print('fusion-input1.shape=', input1.shape)
        # print('fusion-input2.shape=', input2.shape)
        batch_size, output_dim1, output_dim2 = input1.shape[0], input1.shape[1], input2.shape[2]

        output = np.zeros((batch_size, output_dim1, output_dim2))

        for i in range(batch_size):
            output[i] = torch.mm(torch.mm(input1[i], weight), input2[i]).cpu().numpy()
        output = torch.cuda.FloatTensor(output)
        output.cuda()
        return output

    @staticmethod
    def backward(self, grad_output):
        '''
        Y = AWB, than what is the answer of dY/dW ?
        answer: A.t * B.t
        '''
        input1, weight, input2 = self.saved_variables
        batch_size = input1.shape[0]

        grad1_output = np.zeros(input1.shape)
        grad2_output = np.zeros(input2.shape)
        grad_weight = np.zeros(weight.shape)

        # print('grad_output.shape=', grad_output.shape)
        for i in range(batch_size):
            grad1_output[i] = torch.mm(grad_output[i], torch.mm(weight, input2[i]).t()).data.cpu().numpy()
            grad_weight += torch.mm(torch.mm(input1[i].t(), grad_output[i]), input2[i].t()).data.cpu().numpy()
            grad2_output[i] = torch.mm(torch.mm(input1[i], weight).t(), grad_output[i]).data.cpu().numpy()

        grad_weight /= batch_size

        grad1_output = Variable(torch.cuda.FloatTensor(grad1_output))
        grad1_output.cuda()
        grad_weight = Variable(torch.cuda.FloatTensor(grad_weight))
        grad_weight.cuda()
        grad2_output = Variable(torch.cuda.FloatTensor(grad2_output))
        grad2_output.cuda()

        # print('grad1_output.shape=', grad1_output.shape)
        # print('grad_weight.shape=', grad_weight.shape)
        # print('grad2_output.shape=', grad2_output.shape)
        return grad1_output, grad_weight, grad2_output


class matrixLinear(Function):

    @staticmethod
    def forward(self, input1, weight, input2):
        '''
        input.shape= (16, 128)
        Y = input1 * weight * input2
        '''
        self.save_for_backward(input1, weight, input2)
        output = torch.mm(torch.mm(input1, weight), input2)
        return output


    @staticmethod
    def backward(self, grad_output):
        '''
        Y = AWB, than what is the answer of dY/dW ?
        answer: A.t * B.t
        '''
        input1, weight, input2 = self.saved_variables
        return torch.mm(grad_output, torch.mm(weight, input2).t()), \
            torch.mm(torch.mm(input1.t(), grad_output), input2.t()), torch.mm(torch.mm(input1, weight).t(), grad_output)

class net(nn.Module):

    def __init__(self, **args):
        '''
        input_channel= 5
        person_num= 150
        '''

        super(net, self).__init__()

        self.config = args

        self.conv1 = nn.Conv2d(self.config['input_channel'], 6, (5, 5), (1, 1))
        self.conv2 = nn.Conv2d(6, 16, (5, 5), (1, 1))
        self.fc1 = nn.Linear(3840, 128)

    def forward(self, x):
        '''
        output.shape = (1, 128), (1, person_num)
        '''

        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(F.tanh(self.conv2(x)), 2)
        # x = F.tanh(self.conv3(x))
        # print('x.shape=', x.shape)

        x = x.view(x.shape[0], -1)
        # print('linear.shape=', x.shape)

        x = F.dropout(x, 0.6)
        # print('x.shape=', x.shape)
        net_fea = self.fc1(x)

        return net_fea


class fusion(nn.Module):

    def __init__(self, **args):
        '''
        matrix_shape = (16, 128)
        '''
        super(fusion, self).__init__()
        self.config = args

        hidden_size = self.config['matrix_shape'][1]
        self.weight = nn.Parameter((torch.rand(hidden_size, hidden_size) - 0.5) * 2 / hidden_size)

        self.matrixLinear = matrixLinear.apply

    def forward(self, input1, input2):
        '''
        input.shape = seqlen * featurelen, seqlen * featurelen
        return shape = 16 * 16
        '''
        fusion_mat = self.matrixLinear(input1, self.weight, input2.t())

        # fusion_mat = self.bn(self.conv1(fusion_mat))

        return fusion_mat


class full_model(nn.Module):

    def __init__(self, config):

        super(full_model, self).__init__()

        self.config = config

        self.model_normal = model(seqlen=self.config['max_seqlen'], person_num=self.config['person_num'], rnn_type=self.config['rnn_type'])
        self.model_normal.load_state_dict(torch.load(self.config['normal_path']))
        self.model_normal.train(True)

        self.model_reversed = model(seqlen=self.config['max_seqlen'], person_num=self.config['person_num'], rnn_type=self.config['rnn_type'])
        self.model_reversed.load_state_dict(torch.load(self.config['reversed_path']))
        self.model_reversed.train(True)

        self.bn1_normal = nn.BatchNorm2d(1)
        self.bn1_reversed = nn.BatchNorm2d(1)
        self.bn2_normal = nn.BatchNorm2d(1)
        self.bn2_reversed = nn.BatchNorm2d(1)

        # self.fusion = fusion(matrix_shape=(16, 128))
        self.net = net(input_channel=2)

        self.classify = nn.Linear(128, self.config['person_num'])

    def get_hidden(self):
        weight = next(self.parameters()).data
        if self.config['rnn_type'] is 'LSTM':
            hidden = (Variable(weight.new(1, 1, 128).zero_()), Variable(weight.new(1, 1, 128).zero_()))
        else:
            hidden = Variable(weight.new(1, 1, 128).zero_())

        return hidden

    def forward(self, input1_rgb, input1_flow, hidden1, input2_rgb, input2_flow, hidden2):

        def abstract_feature(person1_rgb, person1_flow, h1, person2_rgb, person2_flow, h2, reversed=False):
            idx = 0 if reversed is False else 1
            use_model = self.model_normal if reversed is False else self.model_reversed
            output_model = use_model(person1_rgb[idx], person1_flow[idx], h1[idx], person2_rgb[idx], person2_flow[idx], h2[idx])
            return output_model

        def sendinto_net(person_fea1, person_fea2):
            '''
            person_fea.shape = (16, 128)
            '''
            # reverse person_fea2
            idx = list(range(person_fea2.shape[0]))
            idx.reverse()
            idx = Variable(torch.cuda.LongTensor(idx))
            idx.cuda()
            # print('idx=\n', idx)
            person_fea2 = torch.index_select(person_fea2, 0, idx)

            person_fea1 = person_fea1.unsqueeze(0)
            person_fea2 = person_fea2.unsqueeze(0)

            person_fea = torch.cat((person_fea1, person_fea2), 0).unsqueeze(0)
            person_fea = self.net(person_fea)

            # fusion_mat = self.fusion(person_fea1, person_fea2)
            # fusion_mat = fusion_mat.unsqueeze(0).unsqueeze(0)
            # person_fea = self.net(fusion_mat)

            # person_fea = (torch.mean(person_fea1, 0, True) + torch.mean(person_fea2, 0, True)) / 2
            return person_fea

        # abstract feature from loaded model
        person1_list_normal, person2_list_normal = abstract_feature(input1_rgb, input1_flow, hidden1, input2_rgb, input2_flow, hidden2)
        person1_fea_normal, _ = person1_list_normal
        person2_fea_normal, _ = person2_list_normal

        person1_list_reversed, person2_list_reversed = abstract_feature(input1_rgb, input1_flow, hidden1, input2_rgb, input2_flow, hidden2, reversed=True)
        person1_fea_reversed, _ = person1_list_reversed
        person2_fea_reversed, _ = person2_list_reversed

        # process with our net model
        # person1_fea_normal = self.bn1_normal(person1_fea_normal.unsqueeze(0).unsqueeze(0)).squeeze()
        # person1_fea_reversed = self.bn1_reversed(person1_fea_reversed.unsqueeze(0).unsqueeze(0)).squeeze()
        person1_fea = sendinto_net(person1_fea_normal, person1_fea_reversed)
        person1_cls = F.log_softmax(self.classify(person1_fea))

        # person2_fea_normal = self.bn2_normal(person2_fea_normal.unsqueeze(0).unsqueeze(0)).squeeze()
        # person2_fea_reversed = self.bn2_reversed(person2_fea_reversed.unsqueeze(0).unsqueeze(0)).squeeze()
        person2_fea = sendinto_net(person2_fea_normal, person2_fea_reversed)
        person2_cls = F.log_softmax(self.classify(person2_fea))

        return (person1_fea, person1_cls), (person2_fea, person2_cls)

class modelUtil():

    def __init__(self, config):
        '''
        seqlen = 16
        person_num = 150
        rnn_type = 'RNN'
        learning_rate = 0.001
        lr_decay_epoch = 300
        cuda = True
        '''

        self.config = config
        self.config['cuda'] = torch.cuda.is_available() and self.config['cuda']

        self.classify_loss = nn.NLLLoss()
        self.hinge_loss = nn.HingeEmbeddingLoss(self.config['margin'])
        self.cos_loss = nn.CosineEmbeddingLoss(0.1)

        self.model = full_model(self.config)
        if self.config['cuda'] is True:
            self.model.cuda()

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

        self.FloatTensor = torch.cuda.FloatTensor if self.config['cuda'] else torch.Tensor
        self.LongTensor = torch.cuda.LongTensor if self.config['cuda'] else torch.LongTensor

    def exp_lr_scheduler(self, epoch):
        if epoch % self.config['lr_decay_epoch'] == 0:
            lr = self.config['learning_rate'] * (0.1 ** (epoch / self.config['lr_decay_epoch']))
            print('######################change learning rate={}'.format(lr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def toVariable(self, tensor_ary, tensor_type):
        tensor = Variable(tensor_type(tensor_ary))
        if (self.config['cuda']) is True:
            tensor.cuda()
        return tensor

    def repackage_hidden(self, hidden):
        if type(hidden) == Variable:
            return Variable(hidden.data)
        else:
            return tuple(self.repackage_hidden(v) for v in hidden)


    def train_model(self, input1_rgb, input1_flow, input1_label, input2_rgb, input2_flow, input2_label):
        '''
        numpy.array: input1_rgb, input1_flow, input1_label, input2_rgb, input2_flow, input2_label
        '''
        self.model.train(True)

        # print('input1_rgb.shape=', input1_rgb.shape)
        # print('input1_flow.shape=', input1_flow.shape)

        person1_rgb_normal, person1_rgb_reversed = input1_rgb
        person1_flow_normal, person1_flow_reversed = input1_flow
        person1_rgb_normal = self.toVariable(person1_rgb_normal, self.FloatTensor)
        person1_flow_normal = self.toVariable(person1_flow_normal, self.FloatTensor)
        person1_rgb_reversed = self.toVariable(person1_rgb_reversed, self.FloatTensor)
        person1_flow_reversed = self.toVariable(person1_flow_reversed, self.FloatTensor)
        person1_label = self.toVariable([input1_label], self.LongTensor)

        person2_rgb_normal, person2_rgb_reversed = input2_rgb
        person2_flow_normal, person2_flow_reversed = input2_flow
        person2_rgb_normal = self.toVariable(person2_rgb_normal, self.FloatTensor)
        person2_flow_normal = self.toVariable(person2_flow_normal, self.FloatTensor)
        person2_rgb_reversed = self.toVariable(person2_rgb_reversed, self.FloatTensor)
        person2_flow_reversed = self.toVariable(person2_flow_reversed, self.FloatTensor)
        person2_label = self.toVariable([input2_label], self.LongTensor)

        hinge_label = self.toVariable([1. if input1_label == input2_label else -1.], self.FloatTensor)
        cos_label = self.toVariable([1 if input1_label == input2_label else -1], self.LongTensor)
        
        hidden1 = (self.repackage_hidden(self.model.get_hidden()), self.repackage_hidden(self.model.get_hidden()))
        hidden2 = (self.repackage_hidden(self.model.get_hidden()), self.repackage_hidden(self.model.get_hidden()))

        self.optimizer.zero_grad()

        person1_list, person2_list = self.model((person1_rgb_normal, person1_rgb_reversed), (person1_flow_normal, person1_flow_reversed), \
            hidden1, (person2_rgb_normal, person2_flow_reversed), (person2_flow_normal, person2_flow_reversed), hidden2)

        # about loss of the network
        loss_cls = self.classify_loss(person1_list[1], person1_label) + self.classify_loss(person2_list[1], person2_label)
        # loss_center = torch.sum(torch.norm(person1_list[0] - person1_list[1], 2, 1)) + \
        #     torch.sum(torch.norm(person1_list[0] - person1_list[1], 2, 1))
        distance = torch.norm(person1_list[0] - person2_list[0], 2)
        loss_hinge = self.hinge_loss(distance, hinge_label)
        # loss_cos = self.cos_loss(person1_list[0], person2_list[0], cos_label)

        # print('classification loss=', loss_cls.data[0])
        # print('center loss=', loss_center.data[0])
        # print('hinge loss=', loss_hinge.data[0])

        # svm generate loss
        # waited to update...

        loss = loss_cls + loss_hinge
        loss.backward()
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-.1, 1.)
        self.optimizer.step()
        return loss.data[0], loss_cls.data[0], loss_hinge.data[0], 0., 0.

    def abstract_feature(self, input1_rgb, input1_flow, input2_rgb, input2_flow):
        '''
        numpy.array: input1_rgb, input1_flow input2_rgb, input2_flow
        '''
        self.model.train(False)

        person1_rgb_normal, person1_rgb_reversed = input1_rgb
        person1_flow_normal, person1_flow_reversed = input1_flow
        person1_rgb_normal = self.toVariable(person1_rgb_normal, self.FloatTensor)
        person1_flow_normal = self.toVariable(person1_flow_normal, self.FloatTensor)
        person1_rgb_reversed = self.toVariable(person1_rgb_reversed, self.FloatTensor)
        person1_flow_reversed = self.toVariable(person1_flow_reversed, self.FloatTensor)

        person2_rgb_normal, person2_rgb_reversed = input2_rgb
        person2_flow_normal, person2_flow_reversed = input2_flow
        person2_rgb_normal = self.toVariable(person2_rgb_normal, self.FloatTensor)
        person2_flow_normal = self.toVariable(person2_flow_normal, self.FloatTensor)
        person2_rgb_reversed = self.toVariable(person2_rgb_reversed, self.FloatTensor)
        person2_flow_reversed = self.toVariable(person2_flow_reversed, self.FloatTensor)

        hidden1 = (self.repackage_hidden(self.model.get_hidden()), self.repackage_hidden(self.model.get_hidden()))
        hidden2 = (self.repackage_hidden(self.model.get_hidden()), self.repackage_hidden(self.model.get_hidden()))

        person1_list, person2_list = self.model((person1_rgb_normal, person1_rgb_reversed), (person1_flow_normal, person1_flow_reversed), \
            hidden1, (person2_rgb_normal, person2_flow_reversed), (person2_flow_normal, person2_flow_reversed), hidden2)

        person1_feature = person1_list[0].squeeze().data.cpu().numpy()
        person2_feature = person2_list[0].squeeze().data.cpu().numpy()
        
        return person1_feature, person2_feature

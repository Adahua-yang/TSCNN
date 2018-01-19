import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torchvision import models, transforms
from collections import namedtuple

class cnn(nn.Module):

    def __init__(self, **args):
        '''
        input_channel=2
        '''
        super(cnn, self).__init__()

        self.config = args

        self.zero_padding1 = nn.ZeroPad2d(4)
        self.zero_padding2 = nn.ZeroPad2d(4)
        self.zero_padding3 = nn.ZeroPad2d(4)

        self.conv1 = nn.Conv2d(self.config['input_channel'], 16, (5, 5), (1, 1))
        self.conv2 = nn.Conv2d(16, 32, (5, 5), (1, 1))
        self.conv3 = nn.Conv2d(32, 32, (5, 5), (1, 1))

        self.linear = nn.Linear(32 * 10 * 8, 128)

    def forward(self, input):
        '''
        output.shape = (16, 128)
        '''
        batch_size = input.shape[0]

        input = self.zero_padding1(input)
        input = self.conv1(input)
        input = F.tanh(input)
        input = F.max_pool2d(input, (2, 2), (2, 2))

        # print 'conv1 shape=', input.shape

        input = self.zero_padding2(input)
        input = self.conv2(input)
        input = F.tanh(input)
        input = F.max_pool2d(input, (2, 2), (2, 2))

        # print 'conv2 shape=', input.shape

        input = self.zero_padding3(input)
        input = self.conv3(input)
        input = F.tanh(input)
        input = F.max_pool2d(input, (2, 2), (2, 2))

        # print 'conv3 shape=', input.shape

        # -> (16, 32 * 10 * 8)
        input = input.view(batch_size, -1)
        # print 'input shape=', input.shape

        input = F.dropout(input, 0.6)
        output = self.linear(input)

        return output

class rnn(nn.Module):

    def __init__(self, **args):
        '''
        rnn_type='RNN', 'GRU', 'LSTM'
        '''
        super(rnn, self).__init__()

        self.config = args
        self.rnn_layer = getattr(nn, self.config['rnn_type'])(input_size=256, hidden_size=128, batch_first=True, dropout=0.6)

    def forward(self, input, hidden):
        '''
        expect input.shape = seqlen * featurelen
        expect output.shape = seqlen * featurelen
        '''
        input = input.unsqueeze(0)
        output, _ = self.rnn_layer(input, hidden)
        output = output.squeeze()

        return output


class model(nn.Module):

    def __init__(self, **args):
        '''
        seqlen = 16
        person_num = 150
        rnn_type = 'RNN'
        '''
        super(model, self).__init__()

        self.config = args

        self.rgb_cnn = cnn(input_channel=3)
        self.flow_cnn = cnn(input_channel=3)

        # self.cnn = cnn(input_channel=5)

        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)

        self.rnn = rnn(rnn_type=self.config['rnn_type'])
        # self.flow_rnn = rnn(rnn_type=self.config['rnn_type'])

        # self.fusion = fusion(seqlen=self.config['seqlen'])

        # self.net = net(input_channel=2, person_num=self.config['person_num'])

        self.classify = nn.Linear(128, self.config['person_num'])

    def get_hidden(self):
        weight = next(self.parameters()).data
        if self.config['rnn_type'] is 'LSTM':
            hidden = (Variable(weight.new(1, 1, 128).zero_()), Variable(weight.new(1, 1, 128).zero_()))
        else:
            hidden = Variable(weight.new(1, 1, 128).zero_())

        return hidden

    def forward(self, input1_rgb, input1_flow, hidden1, input2_rgb, input2_flow, hidden2):

        def abstract_feature(input_rgb, input_flow, hidden, bn):
            output_rgb = self.rgb_cnn(input_rgb)
            output_flow = self.flow_cnn(input_flow)

            output_cnn = torch.cat((output_rgb, output_flow), dim=1)
            # output_cnn = bn(output_cnn.unsqueeze(0).unsqueeze(0)).squeeze()
            output_rnn = self.rnn(output_cnn, hidden)

            person_fea = output_rnn
            person_vec = torch.mean(person_fea, 0, True)
            person_cls = F.log_softmax(self.classify(person_vec))

            # fusion_output = self.fusion(output_rgb, output_flow)
            # person_fea, person_cls = self.net(fusion_output)

            return person_fea, person_cls

        person1_fea, person1_cls = abstract_feature(input1_rgb, input1_flow, hidden1, self.bn1)
        person2_fea, person2_cls = abstract_feature(input2_rgb, input2_flow, hidden2, self.bn2)

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
        self.cos_loss = nn.CosineEmbeddingLoss()

        self.model = model(seqlen=self.config['max_seqlen'], person_num=self.config['person_num'], rnn_type=self.config['rnn_type'])
        if self.config['load_cnn']:
            self.model.load_state_dict(torch.load(self.config['save_cnn_path']))
            self.model.train(True)
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

        person1_rgb = self.toVariable(input1_rgb, self.FloatTensor)
        person1_flow = self.toVariable(input1_flow, self.FloatTensor)
        person1_label = self.toVariable([input1_label], self.LongTensor)
        person2_rgb = self.toVariable(input2_rgb, self.FloatTensor)
        person2_flow = self.toVariable(input2_flow, self.FloatTensor)
        person2_label = self.toVariable([input2_label], self.LongTensor)

        hinge_label = self.toVariable([1. if input1_label == input2_label else -1.], self.FloatTensor)
        cos_label = self.toVariable([1 if input1_label == input2_label else -1], self.LongTensor)
        
        hidden1 = self.repackage_hidden(self.model.get_hidden())
        hidden2 = self.repackage_hidden(self.model.get_hidden())

        self.optimizer.zero_grad()
        person1_list, person2_list = self.model(person1_rgb, person1_flow, hidden1, person2_rgb, person2_flow, hidden2)

        # about loss of the network
        loss_cls = self.classify_loss(person1_list[1], person1_label) + self.classify_loss(person2_list[1], person2_label)
        # loss_center = torch.sum(torch.norm(person1_list[0] - person1_list[1], 2, 1)) + \
        #     torch.sum(torch.norm(person1_list[0] - person1_list[1], 2, 1))
        person1_vec = torch.mean(person1_list[0], 0, True)
        person2_vec = torch.mean(person2_list[0], 0, True)
        distance = torch.norm(person1_vec - person2_vec, 2)
        loss_hinge = self.hinge_loss(distance, hinge_label)
        # loss_cos = self.cos_loss(person1_vec, person2_vec, cos_label)

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

        person1_rgb = self.toVariable(input1_rgb, self.FloatTensor)
        person1_flow = self.toVariable(input1_flow, self.FloatTensor)
        person2_rgb = self.toVariable(input2_rgb, self.FloatTensor)
        person2_flow = self.toVariable(input2_flow, self.FloatTensor)

        hidden1 = self.repackage_hidden(self.model.get_hidden())
        hidden2 = self.repackage_hidden(self.model.get_hidden())

        person1_list, person2_list = self.model(person1_rgb, person1_flow, hidden1, person2_rgb, person2_flow, hidden2)

        person1_feature = person1_list[0].squeeze().data.cpu().numpy()
        person2_feature = person2_list[0].squeeze().data.cpu().numpy()
        
        return person1_feature, person2_feature


    def save_model(self):
        torch.save(self.model.state_dict(), self.config['save_cnn_path'])

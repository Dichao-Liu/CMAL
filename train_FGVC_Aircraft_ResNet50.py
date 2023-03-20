from __future__ import print_function
import os
from PIL import Image
import torchvision.models
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from utils import *


class Features(nn.Module):
    def __init__(self, net_layers):
        super(Features, self).__init__()
        self.net_layer_0 = nn.Sequential(net_layers[0])
        self.net_layer_1 = nn.Sequential(net_layers[1])
        self.net_layer_2 = nn.Sequential(net_layers[2])
        self.net_layer_3 = nn.Sequential(net_layers[3])
        self.net_layer_4 = nn.Sequential(*net_layers[4])
        self.net_layer_5 = nn.Sequential(*net_layers[5])
        self.net_layer_6 = nn.Sequential(*net_layers[6])
        self.net_layer_7 = nn.Sequential(*net_layers[7])


    def forward(self, x):
        x = self.net_layer_0(x)
        x = self.net_layer_1(x)
        x = self.net_layer_2(x)
        x = self.net_layer_3(x)
        x = self.net_layer_4(x)
        x1 = self.net_layer_5(x)
        x2 = self.net_layer_6(x1)
        x3 = self.net_layer_7(x2)
        return x1, x2, x3


class Network_Wrapper(nn.Module):
    def __init__(self, net_layers, num_class):
        super().__init__()
        self.Features = Features(net_layers)

        self.max_pool1 = nn.MaxPool2d(kernel_size=56, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=28, stride=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=14, stride=1)

        self.conv_block1 = nn.Sequential(
            BasicConv(512, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class)
        )

        self.conv_block2 = nn.Sequential(
            BasicConv(1024, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.conv_block3 = nn.Sequential(
            BasicConv(2048, 512, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(512, 1024, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(1024 * 3),
            nn.Linear(1024 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x1, x2, x3 = self.Features(x)

        x1_ = self.conv_block1(x1)
        map1 = x1_.detach()
        x1_ = self.max_pool1(x1_)
        x1_f = x1_.view(x1_.size(0), -1)

        x1_c = self.classifier1(x1_f)

        x2_ = self.conv_block2(x2)
        map2 = x2_.detach()
        x2_ = self.max_pool2(x2_)
        x2_f = x2_.view(x2_.size(0), -1)
        x2_c = self.classifier2(x2_f)

        x3_ = self.conv_block3(x3)
        map3 = x3_.detach()
        x3_ = self.max_pool3(x3_)
        x3_f = x3_.view(x3_.size(0), -1)
        x3_c = self.classifier3(x3_f)

        x_c_all = torch.cat((x1_f, x2_f, x3_f), -1)
        x_c_all = self.classifier_concat(x_c_all)

        return x1_c, x2_c, x3_c, x_c_all, map1, map2, map3


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None, data_path = ''):

    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)


    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=data_path+'/train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)


    net = torchvision.models.resnet50()
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
    net.load_state_dict(state_dict)

    net_layers = list(net.children())
    net_layers = net_layers[0:8]
    net = Network_Wrapper(net_layers, 100)

    netp = torch.nn.DataParallel(net, device_ids=[0])


    device = torch.device("cuda")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.Features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        train_loss5 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)


            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])


            # Train the experts from deep to shallow with data augmentation by multiple steps
            # e3
            optimizer.zero_grad()
            inputs3 = inputs
            output_1, output_2, output_3, _, map1, map2, map3 = netp(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()



            p1 = net.state_dict()['classifier3.1.weight']
            p2 = net.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output_3, p1, p2)
            inputs3_att = attention_im(inputs, att_map_3)

            p1 = net.state_dict()['classifier2.1.weight']
            p2 = net.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output_2, p1, p2)
            inputs2_att = attention_im(inputs, att_map_2)

            p1 = net.state_dict()['classifier1.1.weight']
            p2 = net.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output_1, p1, p2)
            inputs1_att = attention_im(inputs, att_map_1)
            inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)

            # e2
            optimizer.zero_grad()
            flag = torch.rand(1)
            if flag < (1 / 3):
                inputs2 = inputs3_att
            elif (1 / 3) <= flag < (2 / 3):
                inputs2 = inputs1_att
            elif flag >= (2 / 3):
                inputs2 = inputs

            _, output_2, _, _, _, map2, _ = netp(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()

            # e1
            optimizer.zero_grad()
            flag = torch.rand(1)
            if flag < (1 / 3):
                inputs1 = inputs3_att
            elif (1 / 3) <= flag < (2 / 3):
                inputs1 = inputs2_att
            elif flag >= (2 / 3):
                inputs1 = inputs

            output_1, _, _, _, map1, _, _ = netp(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()


            # Train the experts and their concatenation with the overall attention region in one go
            optimizer.zero_grad()
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = netp(inputs_ATT)
            concat_loss_ATT = CELoss(output_1_ATT, targets)+\
                            CELoss(output_2_ATT, targets)+\
                            CELoss(output_3_ATT, targets)+\
                            CELoss(output_concat_ATT, targets) * 2
            concat_loss_ATT.backward()
            optimizer.step()


            # Train the concatenation of the experts with the raw input
            optimizer.zero_grad()
            _, _, _, output_concat, _, _, _ = netp(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()


            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss_ATT.item()
            train_loss5 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_ATT: %.5f |Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1),  train_loss5/ (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_ATT: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1), train_loss5 / (idx + 1)))

        if epoch < 5 or epoch >= 100:
            val_acc_com, val_loss = test(net, CELoss, 3, data_path+'/test')
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc_com, val_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)

if __name__ == '__main__':

    data_path = '<the-path-to>/FGVC_Aircraft'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train(nb_epoch=200,             # number of epoch
             batch_size=16,         # batch size
             store_name='Results_FGVC_Aircraft_ResNet50',     # folder for output
             resume=False,          # resume training from checkpoint
             start_epoch=0,         # the start epoch number when you resume the training
             model_path='',
             data_path = data_path)         # the saved model where you want to resume the training

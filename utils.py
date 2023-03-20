import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from basic_conv import *


def show_image(inputs):
    inputs = inputs.squeeze()
    to_pil_image = transforms.ToPILImage()
    img = to_pil_image(inputs.cpu())
    img.show()



def map_generate(attention_map, pred, p1, p2):
    batches, feaC, feaH, feaW = attention_map.size()

    out_map=torch.zeros_like(attention_map.mean(1))

    for batch_index in range(batches):
        map_tpm = attention_map[batch_index]
        map_tpm = map_tpm.reshape(feaC, feaH*feaW)
        map_tpm = map_tpm.permute([1, 0])
        p1_tmp = p1.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p1_tmp)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

        pred_tmp = pred[batch_index]
        pred_ind = pred_tmp.argmax()
        p2_tmp = p2[pred_ind].unsqueeze(1)

        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p2_tmp)
        out_map[batch_index] = map_tpm.reshape(feaH, feaW)

    return out_map

def attention_im(images, attention_map, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta
        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images



def highlight_im(images, attention_map, attention_map2, attention_map3, theta=0.5, padding_ratio=0.1):
    images = images.clone()
    attention_map = attention_map.clone().detach()
    attention_map2 = attention_map2.clone().detach()
    attention_map3 = attention_map3.clone().detach()

    batches, _, imgH, imgW = images.size()

    for batch_index in range(batches):
        image_tmp = images[batch_index]
        map_tpm = attention_map[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm = torch.nn.functional.upsample_bilinear(map_tpm, size=(imgH, imgW)).squeeze()
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)


        map_tpm2 = attention_map2[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm2 = torch.nn.functional.upsample_bilinear(map_tpm2, size=(imgH, imgW)).squeeze()
        map_tpm2 = (map_tpm2 - map_tpm2.min()) / (map_tpm2.max() - map_tpm2.min() + 1e-6)

        map_tpm3 = attention_map3[batch_index].unsqueeze(0).unsqueeze(0)
        map_tpm3 = torch.nn.functional.upsample_bilinear(map_tpm3, size=(imgH, imgW)).squeeze()
        map_tpm3 = (map_tpm3 - map_tpm3.min()) / (map_tpm3.max() - map_tpm3.min() + 1e-6)

        map_tpm = (map_tpm + map_tpm2 + map_tpm3)
        map_tpm = (map_tpm - map_tpm.min()) / (map_tpm.max() - map_tpm.min() + 1e-6)
        map_tpm = map_tpm >= theta

        nonzero_indices = torch.nonzero(map_tpm, as_tuple=False)
        height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
        height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
        width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
        width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

        image_tmp = image_tmp[:, height_min:height_max, width_min:width_max].unsqueeze(0)
        image_tmp = torch.nn.functional.upsample_bilinear(image_tmp, size=(imgH, imgW)).squeeze()

        images[batch_index] = image_tmp

    return images



def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)



def test(net, criterion, batch_size, test_path):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    correct_com2 = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path,
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

        p1 = net.state_dict()['classifier3.1.weight']
        p2 = net.state_dict()['classifier3.4.weight']
        att_map_3 = map_generate(map3, output_3, p1, p2)

        p1 = net.state_dict()['classifier2.1.weight']
        p2 = net.state_dict()['classifier2.4.weight']
        att_map_2 = map_generate(map2, output_2, p1, p2)

        p1 = net.state_dict()['classifier1.1.weight']
        p2 = net.state_dict()['classifier1.4.weight']
        att_map_1 = map_generate(map1, output_1, p1, p2)

        inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
        output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

        outputs_com2 = output_1 + output_2 + output_3 + output_concat
        outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        _, predicted_com2 = torch.max(outputs_com2.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()
        correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1),
            100. * float(correct_com) / total, correct_com, total))

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss


def test_tresnetl(net, criterion, batch_size, test_path):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    correct_com2 = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Scale((421, 421)),
        transforms.CenterCrop(368),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path,
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

        p1 = net.state_dict()['classifier3.1.weight']
        p2 = net.state_dict()['classifier3.4.weight']
        att_map_3 = map_generate(map3, output_3, p1, p2)

        p1 = net.state_dict()['classifier2.1.weight']
        p2 = net.state_dict()['classifier2.4.weight']
        att_map_2 = map_generate(map2, output_2, p1, p2)

        p1 = net.state_dict()['classifier1.1.weight']
        p2 = net.state_dict()['classifier1.4.weight']
        att_map_1 = map_generate(map1, output_1, p1, p2)

        inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
        output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

        outputs_com2 = output_1 + output_2 + output_3 + output_concat
        outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        _, predicted = torch.max(output_concat.data, 1)
        _, predicted_com = torch.max(outputs_com.data, 1)
        _, predicted_com2 = torch.max(outputs_com2.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct_com += predicted_com.eq(targets.data).cpu().sum()
        correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

        print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1),
            100. * float(correct_com) / total, correct_com, total))

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss



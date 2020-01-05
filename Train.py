from __future__ import print_function
from __future__ import division

import os
import time
import argparse
import numpy as np
from dataloader import Image_dataset,default_loader
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import transforms
from torch.utils.data import DataLoader
from torchFewShot.models.net import Model
# sys.path.append('./torchFewShot')
# import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder

label_onehot = OneHotEncoder()
label_onehot.fit([[0], [1], [2], [3], [4]])

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/Datasets/miniImageNet--ravi', help='/miniImageNet')
parser.add_argument('--data_name', default='miniImageNet', help='miniImageNet|StanfordDog|StanfordCar|CubBird')
parser.add_argument('--mode', default='train', help='train|val|test')
parser.add_argument('--outf', default='./results/')
parser.add_argument('--resume', default='', type=str, help='path to the lastest checkpoint (default: none)')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--scale_class',type=int ,default=7)
parser.add_argument('--num_class',type=int, default=64)
parser.add_argument('--basemodel', default='CAM', help='Conv64F|ResNet256F')
#  Few-shot parameters  #
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--batchsize',type=int, default=4)
parser.add_argument('--episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--testepisodeSize', type=int, default=1, help='one episode is taken as a mini-batch')
parser.add_argument('--epochs', type=int, default=100, help='the total number of training epoch')
parser.add_argument('--episode_train_num', type=int, default=1200, help='the total number of training episodes')
parser.add_argument('--episode_val_num', type=int, default=2000, help='the total number of evaluation episodes')
parser.add_argument('--episode_test_num', type=int, default=2000, help='the total number of testing episodes')
parser.add_argument('--way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--shot_num', type=int, default=5, help='the number of shot')
parser.add_argument('--query_num', type=int, default=6, help='the number of queries')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='the number of gpus')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 100)')
opt = parser.parse_args()
opt.cuda = True


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        loss = (- targets * log_probs).mean(0).sum()
        return loss / inputs.size(2)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
	torch.save(state, filename)

# def adjust_learning_rate(optimizer, epoch_num):
# 	lr = opt.lr * (0.05 ** (epoch_num // 5))
# 	for param_group in optimizer.param_groups:
# 		param_group['lr'] = lr

def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(cls_score, query_targets,batch_size,num_example_query):
    with torch.no_grad():
        _, preds = torch.max(cls_score.detach().cpu(), 1)

        acc = (torch.sum(preds == query_targets.detach().cpu()).float()) / query_targets.size(0)

        gt = (preds == query_targets.detach().cpu()).float()
        gt = gt.view(batch_size, num_example_query).numpy()  # [b, n]
        acc = np.sum(gt, 1) / num_example_query
        acc = np.sum(acc)/batch_size
    #
    return acc

def train(train_loader, model, criterion, optimizer, epoch_index, F_txt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    model.train()
    for episode_index, (support_images,query_images,support_targets,query_targets,original_targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        support_images,query_images,original_targets = support_images.cuda(),query_images.cuda(),\
                                                                     original_targets.cuda()
        batch_size = support_images.size(0)
        num_example_train = support_images.size(1)
        num_example_query = query_images.size(1)

        new_y_train = label_onehot.transform(support_targets.reshape(-1, 1)).toarray().reshape(batch_size, num_example_train, -1)

        new_y_test = label_onehot.transform(query_targets.reshape(-1, 1)).toarray().reshape(batch_size, num_example_query, -1)

        y_train = torch.from_numpy(new_y_train).float().cuda()
        y_test = torch.from_numpy(new_y_test).float().cuda()

        K_shot = y_train.size(2)
        ytest,cls_score = model(support_images,query_images,y_train,y_test)

        new_cls_score = torch.sum(cls_score.view(batch_size * num_example_query, K_shot,-1),dim=2)
        new_cls_score = new_cls_score.view(batch_size * num_example_query, K_shot)
        query_targets = query_targets.view(batch_size * num_example_query)

        loss1 = 0.5*criterion(cls_score,query_targets.cuda())
        loss2 = criterion(ytest,original_targets.view(-1))
        loss = loss1+loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec = accuracy(new_cls_score, query_targets,batch_size,num_example_query)

        losses.update(loss.item(),batch_size)
        top1.update(prec.item())
        batch_time.update(time.time() - end)
        end = time.time()

        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print('Eposide-({0}): [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1))

            print('Eposide-({0}): [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                top1=top1), file=F_txt)

def validate(val_loader, model, criterion, epoch_index, best_prec1, F_txt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    accuracies = []

    end = time.time()
    with torch.no_grad():
        for episode_index, (support_images,query_images,support_targets,query_targets) in enumerate(val_loader):
            batch_time.update(time.time() - end)

            # Convert query and support images
            support_images,query_images = support_images.cuda(),query_images.cuda()
            batch_size = support_images.size(0)
            num_example_train = support_images.size(1)
            num_example_query = query_images.size(1)

            new_y_train = label_onehot.transform(support_targets.reshape(-1, 1)).toarray().reshape(batch_size, num_example_train, -1)

            new_y_test = label_onehot.transform(query_targets.reshape(-1, 1)).toarray().reshape(batch_size, num_example_query, -1)

            y_train = torch.from_numpy(new_y_train).float().cuda()
            y_test = torch.from_numpy(new_y_test).float().cuda()
            cls_score = model(support_images,query_images,y_train,y_test)
            # print(cls_score.shape)
            cls_score = cls_score.view(batch_size * num_example_query, -1)
            K_shot = y_train.size(2)
            # cls_score = torch.sum(cls_score.view(batch_size * num_example_query, K_shot, -1), dim=2)
            query_targets = query_targets.view(batch_size * num_example_query).cuda()
            loss = criterion(cls_score, query_targets)
            acc = accuracy(cls_score,query_targets,batch_size,num_example_query)

            losses.update(loss.item() / batch_size)
            top1.update(acc.item())
            accuracies.append(acc)

            batch_time.update(time.time() - end)
            end = time.time()

            if episode_index % opt.print_freq == 0 and episode_index != 0:
                print('Test-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

                print('Test-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1),
                    file=F_txt)

        print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))
        print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1),
              file=F_txt)
        # print('lr:{}'.format(learning_rate),file=F_txt)

        return top1.avg, accuracies

def test(val_loader, model, criterion, epoch_index, best_prec1, F_txt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    accuracies = []

    end = time.time()
    with torch.no_grad():
        for episode_index, (support_images,query_images,support_targets,query_targets) in enumerate(val_loader):
            batch_time.update(time.time() - end)

            # Convert query and support images
            support_images,query_images = support_images.cuda(),query_images.cuda()
            batch_size = support_images.size(0)
            num_example_train = support_images.size(1)
            num_example_query = query_images.size(1)

            new_y_train = label_onehot.transform(support_targets.reshape(-1, 1)).toarray().reshape(batch_size, num_example_train, -1)

            new_y_test = label_onehot.transform(query_targets.reshape(-1, 1)).toarray().reshape(batch_size, num_example_query, -1)

            y_train = torch.from_numpy(new_y_train).float().cuda()
            y_test = torch.from_numpy(new_y_test).float().cuda()
            cls_score = model(support_images,query_images,y_train,y_test)
            # print(cls_score.shape)
            cls_score = cls_score.view(batch_size * num_example_query, -1)
            K_shot = y_train.size(2)
            # cls_score = torch.sum(cls_score.view(batch_size * num_example_query, K_shot, -1), dim=2)
            query_targets = query_targets.view(batch_size * num_example_query).cuda()
            loss = criterion(cls_score, query_targets)

            _, preds = torch.max(cls_score.detach().cpu(), 1)
            acc = (torch.sum(preds == query_targets.detach().cpu()).float()) / query_targets.size(0)
            top1.update(acc.item(),query_targets.size(0))
            losses.update(loss.item(),query_targets.size(0))
            gt = (preds == query_targets.detach().cpu()).float()
            gt = gt.view(batch_size, num_example_query).numpy()  # [b, n]
            acc = np.sum(gt, 1) / num_example_query
            acc = np.reshape(acc, (batch_size))
            accuracies.append(acc)

            if episode_index % opt.print_freq == 0 and episode_index != 0:
                print('Test-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

                print('Test-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1),
                    file=F_txt)

        print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1))
        print(' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1),
              file=F_txt)

    accuracy = top1.avg
    test_accuracies = np.array(accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(2000)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))
    return top1.avg, accuracies


outf = opt.outf + 'CAN_' + 'ResNet_'+ str(opt.way_num) + '_'+ str(opt.shot_num) +'/'+'temp_test_adam'

if not os.path.exists(outf):
    os.makedirs(outf)

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

cudnn.benchmark = True
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

txt_save_path = os.path.join(outf, 'opt_resutls.txt')
F_txt = open(txt_save_path, 'a+')
print(opt)
print(opt, file=F_txt)
ngpu = int(opt.ngpu)
global best_prec1, epoch_index
best_prec1 = 0
epoch_index = 0

model = Model(opt.scale_class,opt.num_class).cuda()
criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

if opt.resume:
	if os.path.isfile(opt.resume):
		print("=> loading checkpoint '{}'".format(opt.resume))
		checkpoint = torch.load(opt.resume)
		epoch_index = checkpoint['epoch_index']
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch_index']))
		print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch_index']), file=F_txt)
	else:
		print("=> no checkpoint found at '{}'".format(opt.resume))
		print("=> no checkpoint found at '{}'".format(opt.resume), file=F_txt)

if opt.ngpu > 1:
	model = nn.DataParallel(model, range(opt.ngpu))

# print the architecture of the network
print(model)
print(model, file=F_txt)

# ======================================== Training phase ===============================================
print('\n............Start training............\n')
start_time = time.time()

LUT_lr = [(60, 0.1), (70, 0.006), (80, 0.0012), (90, 0.00024)]
for epoch_item in range(opt.epochs):
    print('===================================== Epoch %d =====================================' % epoch_item)
    print('===================================== Epoch %d =====================================' % epoch_item,
          file=F_txt)
    # adjust_learning_rate(optimizer, epoch_item)
    learning_rate = adjust_learning_rate(optimizer, epoch_item, LUT_lr)
    # ======================================= Folder of Datasets =======================================
    # image transform & normalization
    # Test_ImgTransform = transforms.Compose([
    #     transforms.Resize((opt.imageSize, opt.imageSize)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    #
    # # traintransformer
    # Train_ImgTransform = transforms.Compose([
    #     transforms.Resize((opt.imageSize, opt.imageSize)),
    #     transforms.RandomCrop(opt.imageSize),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

    transform_train = transforms.Compose([
        transforms.Resize((84, 84), interpolation=3),
        transforms.RandomCrop(84, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(0.5)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((84, 84), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    trainset = Image_dataset(data_dir= '/home/lemon/few-shot/DN4/dataset/miniImageNet/mini-imagenet', mode = 'train', image_size = opt.imageSize,
                            transform = transform_train,episode_num = opt.episode_train_num, way_num = opt.way_num, shot_num = opt.shot_num, query_num = opt.query_num,
                             num_class = opt.num_class,loader=default_loader,batch_size=opt.batchsize)
    valset = Image_dataset(data_dir= '/home/lemon/few-shot/DN4/dataset/miniImageNet/mini-imagenet', mode = 'val', image_size = opt.imageSize,
                            transform = transform_test,episode_num = opt.episode_val_num, way_num = opt.way_num, shot_num = opt.shot_num, query_num = opt.query_num,
                             num_class = opt.num_class,loader=default_loader,batch_size=opt.batchsize)
    testset = Image_dataset(data_dir= '/home/lemon/few-shot/DN4/dataset/miniImageNet/mini-imagenet', mode = 'test', image_size = opt.imageSize,
                            transform = transform_test,episode_num = opt.episode_val_num, way_num = opt.way_num, shot_num = opt.shot_num, query_num = opt.query_num,
                             num_class = opt.num_class,loader=default_loader,batch_size=opt.batchsize)

    train_dataloder = DataLoader(trainset,batch_size=4,num_workers=4,shuffle=True)
    val_dataloader = DataLoader(valset,batch_size=4,num_workers=4,shuffle=True)
    test_dataloader = DataLoader(testset,batch_size=4,num_workers=4,shuffle=True)

    print('Trainset: %d' % len(trainset))
    print('Valset: %d' % len(valset))
    print('Testset: %d' % len(testset))
    print('Trainset: %d' % len(trainset), file=F_txt)
    print('Valset: %d' % len(valset), file=F_txt)
    print('Testset: %d' % len(testset), file=F_txt)

    if epoch_item < 1:
        model.train()
    else:
        model.eval()

    # train(train_dataloder,model,criterion,optimizer,epoch_item,F_txt,learning_rate)
    train(train_dataloder, model, criterion, optimizer, epoch_item, F_txt)
    print('============ Validation on the val set ============')
    print('============ validation on the val set ============', file=F_txt)
    if epoch_item%10==0 or epoch_item>60:
        prec1, _ = test(val_dataloader, model, criterion, epoch_item, best_prec1, F_txt)

        # record the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save the checkpoint
        if is_best:
            save_checkpoint(
                {
                    'epoch_index': epoch_item,
                    'arch': opt.basemodel,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(outf, 'model_best.pth.tar'))

        if epoch_item % 10 == 0:
            filename = os.path.join(outf, 'epoch_%d.pth.tar' % epoch_item)
            save_checkpoint(
                {
                    'epoch_index': epoch_item,
                    'arch': opt.basemodel,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, filename)

        # Testing Prase
        print('============ Testing on the test set ============')
        print('============ Testing on the test set ============', file=F_txt)
        prec1, _ = test(test_dataloader, model, criterion, epoch_item, best_prec1, F_txt)

F_txt.close()
print('............Training is end............')

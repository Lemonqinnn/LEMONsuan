import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchFewShot.models.resnet12 import resnet12
from torchFewShot.models.cam import CAM
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Model(nn.Module):
    def __init__(self, scale_cls, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls

        self.base = resnet12()
        self.cam = CAM()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(num_classes,num_classes)

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest*ftrain, dim=-1)

        # ....
        # ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        # ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        # ftrain_norm = ftrain_norm.unsqueeze(4)
        # ftrain_norm = ftrain_norm.unsqueeze(5)
        #
        # ftrain_norm = ftrain_norm.repeat(1,1,1,1,6,6)
        # new_f = torch.pow((ftest_norm-ftrain_norm),2)
        # cls_scores = self.scale_cls * torch.sum(new_f,dim=3)
        # scores = cls_scores.view(ftrain_norm.size(0) * ftrain_norm.size(1), *cls_scores.size()[2:])
        # scores = torch.sum(scores.view(ftrain_norm.size(0) * ftrain_norm.size(1), ftrain_norm.size(2), -1), dim=2)

        #...
        # ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        # ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        # ftrain_norm = ftrain_norm.unsqueeze(4)
        # ftrain_norm = ftrain_norm.unsqueeze(5)
        # cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        # cls_scores = cls_scores.view(cls_scores.size(0) * cls_scores.size(1), *cls_scores.size()[2:])

        return scores



    def forward(self, xtrain, xtest, ytrain, ytest=None):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1)
        ftrain = torch.bmm(ytrain, ftrain)

        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])
        ftrain, ftest = self.cam(ftrain, ftest)

        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)
        if not self.training:
            return self.test(ftrain, ftest)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)

        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])

        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3) 
        ytest = ytest.unsqueeze(3)
        ftest = torch.matmul(ftest, ytest) 
        ftest = ftest.view(batch_size * num_test, -1, 6, 6)
        ytest = self.clasifier(ftest)
        # ytest = self.avgpool(ytest)
        # ytest = ytest.view(batch_size*num_test,-1)
        # ytest = self.fc(ytest)

        return ytest, cls_scores



# if __name__ == '__main__':
#     model = Model(scale_cls=7, num_classes=64).cuda()
#     # print(list(model.clasifier.parameters())[-2].shape)
#
#     x_train = torch.randn(2,25,3,84,84).cuda()
#     x_test = torch.randn(2,10,3,84,84).cuda()
#
#     y_train = np.random.randint(0,4,(2,25))
#     y_test = np.random.randint(0,4,(2,10))
#     labels_test = torch.from_numpy(y_test)
#     label_onehot = OneHotEncoder()
#     label_onehot.fit([[0],[1],[2],[3],[4]])
#     new_y_train = label_onehot.transform(y_train.reshape(-1,1)).toarray().reshape(2,25,-1)
#
#     new_y_test = label_onehot.transform(y_test.reshape(-1,1)).toarray().reshape(2,10,-1)
#     y_train = torch.from_numpy(new_y_train).float().cuda()
#     y_test = torch.from_numpy(new_y_test).float().cuda()
#
#     # model.eval()
#     # print(x_train.shape)
#     # print(x_test.shape)
#     # print(y_train.shape)
#     # print(y_test.shape)
#     ytest,cls_scores = model(x_train,x_test,y_train,y_test)
#     batch_size = 2
#     num_test_examples = 10
#     cls_scores = torch.mean(cls_scores.view(batch_size *num_test_examples, 5, -1), dim=2)
#     cls_scores = cls_scores.view(batch_size * num_test_examples, 5)
#     labels_test = labels_test.view(batch_size * num_test_examples)
#     _, preds = torch.max(cls_scores.detach().cpu(), 1)
#
#     acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
#     print(acc)
#     gt = (preds == labels_test.detach().cpu()).float()
#     gt = gt.view(batch_size, num_test_examples).numpy()  # [b, n]
#     acc = np.sum(gt, 1) / num_test_examples
#     acc = np.reshape(acc, (batch_size))
#     acc = np.sum(acc)/batch_size
#     print(acc)

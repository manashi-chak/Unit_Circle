import torch
from torch import nn
import torch.nn.functional as F
import pdb
import math


class unit_circle(nn.Module):
    def __init__(self, pad_h, pad_w, kernel_h, kernel_w):
        self.inplanes = 1  # grayscale image
        super(unit_circle, self).__init__()

        self.pad_h, self.pad_w = pad_h, pad_w  # pad_h : horizontal padding, pad_w: vertical padding
        self.kernel_h, self.kernel_w = kernel_h, kernel_w  # kernel_h : spatial extent of horizontal kernel, pad_w: spatial extent of vertical kernel

        self.conv_uc = nn.Conv2d(1, 2, kernel_size=(self.kernel_w, self.kernel_h), stride=(11, 2),
                                 padding=0, bias=True)  # this conv is called 5 times for each image in the pair

        # this loop is for initialization of  weights and bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def circular_cat(self, x, filter_size=15):
        size = (filter_size - 1) // 2
        return torch.cat((x[:, :, :, -size:], x, x[:, :, :, :size]), dim=-1)

    # ======== Normalising the output of UC Layer==========#
    def norm(self, x):
        x_norm = torch.norm(x, dim=1, keepdim=True) + (10 ** -12)
        x = x / x_norm
        x = x + 1
        x = x / 2
        return x

    def forward(self, x):
        x = self.conv_uc(self.circular_cat(x, self.conv_uc.kernel_size[1]))
        out = self.norm(x)

        return out


class Radim(nn.Module):

    def __init__(self, num_classes, include_top=True):
        self.inplanes = 1  # grayscale image
        super(Radim, self).__init__()
        self.include_top = include_top
        self.num_classes = num_classes

        # ==========Unit Circle Layer Processing Starts =================#

        self.uc_x1 = unit_circle(pad_h=4, pad_w=0, kernel_h=3, kernel_w=9)
        self.uc_x2 = unit_circle(pad_h=4, pad_w=0, kernel_h=15, kernel_w=9)
        self.uc_x3 = unit_circle(pad_h=4, pad_w=0, kernel_h=27, kernel_w=9)
        self.uc_x4 = unit_circle(pad_h=4, pad_w=0, kernel_h=27, kernel_w=9)
        self.uc_x5 = unit_circle(pad_h=4, pad_w=0, kernel_h=51, kernel_w=9)

        # =========== Matcher Network Starts ====================#
        self.bn1 = nn.BatchNorm2d(20)  # Batch normalision on the activation maps which is o/p from UC layer and i/p to matcher
        self.drop1 = nn.Dropout(p=0.2)  # Dropout on the activation maps which is o/p from UC layer and i/p to matcher

        self.conv1 = nn.Conv2d(20, 32, kernel_size=3, stride=(1, 2), padding=(1, 0))

        self.bn2 = nn.BatchNorm2d(32)  # Batch normalision after 1st conv of the Matcher
        self.drop2 = nn.Dropout(p=0.2)  # Dropout after 1st conv of the Matcher

        self.conv2 = nn.Conv2d(32, 40, kernel_size=4, stride=(1, 2), padding=(2, 0))

        self.bn3 = nn.BatchNorm2d(40)  # Batch normalision after 2nd conv of the Matcher
        self.drop3 = nn.Dropout(p=0.2)  # Dropout after 2nd conv of the Matcher

        self.conv3 = nn.Conv2d(40, 50, kernel_size=5, padding=(2, 0))

        self.bn4 = nn.BatchNorm2d(50)  # Batch normalision after 3rd conv of the Matcher
        self.drop4 = nn.Dropout(p=0.2)  # Dropout after 3rd conv of the Matcher

        self.conv4 = nn.Conv2d(50, 50, kernel_size=5, stride=(1, 2), padding=(2, 0))

        self.bn5 = nn.BatchNorm2d(50)  # Batch normalization after 4th conv of the Matcher
        self.drop5 = nn.Dropout(p=0.3)  # Dropout after 2nd conv of the Matcher

        self.conv5 = nn.Conv2d(50, 60, kernel_size=(4, 7), stride=(1, 2))

        self.bn6 = nn.BatchNorm2d(60)  # Batch normalization after 5th conv of the Matcher
        self.drop6 = nn.Dropout(p=0.3)  # Dropout after 5th conv of the Matcher

        self.conv6 = nn.Conv2d(60, 60, kernel_size=5, stride=1)

        self.bn7 = nn.BatchNorm2d(60)  # Batch normalization after 6th conv of the Matcher
        self.drop7 = nn.Dropout(p=0.4)  # Dropout after 6th conv of the Matcher

        self.conv7 = nn.Conv2d(60, 60, kernel_size=(4, 7), stride=1)

        self.bn8 = nn.BatchNorm2d(60)  # Batch normalization after 7th conv of the Matcher
        self.drop8 = nn.Dropout(p=0.4)  # Dropout after 7th conv of the Matcher

        self.conv_last = nn.Conv2d(60, 1, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():  # initialize weights
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ================== Forward =================#
    def forward(self, x, y):  # x,y forms the image pair
        # ========First image (x) goes to the UC layers=======#
        mysize=x.shape
        x=x.reshape(mysize[0],1,mysize[1],mysize[2])
        x1 = self.uc_x1(x)
        x2 = self.uc_x2(x)
        x3 = self.uc_x3(x)
        x4 = self.uc_x4(x)
        x5 = self.uc_x5(x)

        # ========Second image (y) goes to the UC layers=======#
        mysize=y.shape
        y=y.reshape(mysize[0],1,mysize[1],mysize[2])
        y1 = self.uc_x1(y)
        y2 = self.uc_x2(y)
        y3 = self.uc_x3(y)
        y4 = self.uc_x4(y)
        y5 = self.uc_x5(y)

        # ======Concatination of the activation maps along the depth =====#
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        y = torch.cat((y1, y2, y3, y4, y5), 1)

        ######################MATCHER#############################
        x = torch.cat((x, y), 1)  # concatinating the activation maps from the UC layer of the x,y pair along the depth

        ######################MATCHER#############################

        nonlin = F.elu
        x = self.bn1(x)

        x = nonlin(self.bn2(self.conv1(self.drop1(x))))
        x = nonlin(self.bn3(self.conv2(self.drop2(x))))
        x = nonlin(self.bn4(self.conv3(self.drop3(x))))
        x = nonlin(self.bn5(self.conv4(self.drop4(x))))
        x = nonlin(self.bn6(self.conv5(self.drop5(x))))
        x = nonlin(self.bn7(self.conv6(self.drop6(x))))
        x = nonlin(self.bn8(self.conv7(self.drop7(x))))

        x = self.conv_last(self.drop8(x))

        x = torch.sigmoid(x)

        if not self.include_top:
            return x

        return x

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Author: Ke Xian
Email: kexian@hust.edu.cn
Date: 2019/04/09
'''

import torch
import torch.nn as nn
import torch.nn.init as init
import sys
#sys.path.append('models/syncbn')
#import modules.nn as NN

# ==============================================================================================================

class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels

        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_branch = nn.Sequential(nn.ReLU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True),\
                                         nn.BatchNorm2d(num_features=self.mid),\
                                         nn.ReLU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels= self.mid, kernel_size=3, padding=1, stride=1, bias=True))
        self.relu = nn.ReLU(inplace=True)

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): 
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class ATA(nn.Module):
    def __init__(self, inchannels, reduction = 8):
        super(ATA, self).__init__()
        self.inchannels = inchannels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.inchannels*2, self.inchannels // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.inchannels // reduction, self.inchannels),
                                nn.Sigmoid())
        self.init_params()

    def forward(self, low_x, high_x):
        n, c, _, _ = low_x.size()
        x = torch.cat([low_x, high_x], 1)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.fc(x).view(n,c,1,1)
        x = low_x * x + high_x

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)
        
        self.init_params()

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

                    
class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor
        
        self.adapt_conv = nn.Sequential(nn.Conv2d(in_channels=int(self.inchannels), out_channels=int(self.inchannels/2), kernel_size=3, padding=1, stride=1, bias=True),\
                                  nn.BatchNorm2d(num_features=int(self.inchannels/2)),\
                                  nn.ReLU(inplace=True),\
                                  nn.Conv2d(in_channels=int(self.inchannels/2), out_channels=self.outchannels, kernel_size=3, padding=1, stride=1, bias=True),\
                                       nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))
                
        self.init_params()
        
    def forward(self, x):
        x = self.adapt_conv(x)
        return x
        
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)        

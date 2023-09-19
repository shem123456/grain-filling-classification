#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import timm
import torchvision

# model = timm.create_model('ghostnet_100', pretrained=False, num_classes=5).to("cuda:0")
# data = torch.randn(16, 3, 224, 224).to("cuda:0")
# summary(model, data)

# model = torchvision.models.vgg16(pretrained=True)  # 加载torch原本的vgg16模型，设置pretrained=True，即使用预训练模型
# num_fc = model.classifier[6].in_features  # 获取最后一层的输入维度
# model.classifier[6] = torch.nn.Linear(num_fc, 12)
# print(model)
#print(model)
print(torch.cuda.is_available())


 
#!/usr/bin/env python

from __future__ import print_function


import torch
import torch.nn as nn
import definitions
from torchvision import datasets, transforms
from torchvision.utils import save_image

batch_size = 1
test_batch_size = 1

trans = transforms.Compose([
    transforms.Resize((20, 20)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(definitions.DATA_PATH, train=True, download=True,
                   transform=trans),
    batch_size=batch_size, shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(definitions.DATA_PATH, train=False,
                   transform=trans),
    batch_size=test_batch_size, shuffle=False)


for i, data in enumerate(train_loader):
    inputs, labels = data
    sample = inputs

    save_image(sample, "/mnt/storage/teo/npaq/ccs-submission/trans-data/img" + str(i) + ".png")

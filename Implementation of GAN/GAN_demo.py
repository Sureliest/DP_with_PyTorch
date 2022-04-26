import os

import numpy as np
import torch
import numpy

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F

os.makedirs('images_01', exist_ok=True)
cuda = True if torch.cuda.is_available() else False
img_shape = (1, 28, 28)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.model(img)
        return validity


loss_funciton = torch.nn.BCELoss()
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    loss_funciton.cuda()

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    )
)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.002, betas=(0.5, 0.999))

for epoch in range(20):
    for i, (imgs, _) in enumerate(dataloader):
        valid = torch.Tensor(imgs.size(0), 1).fill_(1.0).cuda()
        fake = torch.Tensor(imgs.size(0), 1).fill_(0.0).cuda()
        real_imgs = imgs.type(torch.Tensor).cuda()

        # 训练生成器
        optimizer_G.zero_grad()
        z = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], 100))).cuda()
        gen_imgs = generator(z)
        g_loss = loss_funciton(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = loss_funciton(discriminator(real_imgs), valid)
        fake_loss = loss_funciton(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % 100 == 0:
            save_image(gen_imgs.data[:25], "images_01/%d.png" % batches_done, nrow=5, normalize=True)

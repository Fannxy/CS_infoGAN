import numpy as np
import itertools
import pickle as pkl
import pandas as pd

# for network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# for data loading
from torchvision import datasets
import torchvision.transforms as transforms

import sys
sys.path.append('./')
import infoGAN_mnist as im
import LP_Compressed_Sensing as lcs

BATCH_SIZE = 64
CATEGORY = 10
LATENT_NUM = 2
cat_num = np.array([10])


def data_loading():
    '''
    used for MNIST data loading
    '''
    num_workers = 0
    batch_size = 64
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, train_data


def cat_generation(label, cate_num):
    cat_arr = np.zeros(cat_num)
    cat_arr[label] = 1

    return cat_arr


def con_generation(
        con_num,
        style=None,
):
    if style is None:
        con_arr = np.random.standard_normal(size=(con_num))
        return con_arr
    else:
        lat_arr = np.random.uniform(-1, 1, size=(style))
        con_arr = np.random.standard_normal(size=(con_num))
        return np.hstack([lat_arr, con_arr])


def z_batch_generation(labels_batch):
    labels_batch = labels_batch.numpy()
    batch_size = len(labels_batch)

    cat_batch = np.vstack(
        [cat_generation(item, CATEGORY) for item in labels_batch])
    con_batch = np.vstack([con_generation(64) for i in range(batch_size)])

    z_batch = np.hstack([cat_batch, con_batch])

    return z_batch


class Proj_Net(nn.Module):
    '''
    maps the output x to the input noise z
    '''

    def __init__(self, input_size, cat_num, con_num, noi_num):
        super(Proj_Net, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(1 * 28 * 28, 16 * 16),)
        self.l2 = nn.Sequential(nn.Linear(16 * 16, 128))
        self.cat_layer = nn.Sequential(nn.Linear(128, cat_num), nn.Softmax())
        self.con_layer = nn.Sequential(nn.Linear(128, con_num))
        self.noi_layer = nn.Sequential(nn.Linear(128, noi_num))

    def forward(self, x):
        x = x.view(64, 1, 28 * 28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        cat = F.relu(self.cat_layer(x))
        con = F.relu(self.con_layer(x))
        noi = self.noi_layer(x)

        return cat, con, noi


if __name__ == '__main__':

    # hyper parameters
    num_epoch = 50
    lamda = 1
    lamda_con = 0.1

    # generator
    input_size = 74
    output_size = 28 * 28
    Gen = im.Generator(input_size, output_size)

    # projector network
    Pro = Proj_Net(28 * 28, CATEGORY, LATENT_NUM, 62)
    im.weights_init_normal(Pro)

    # loss for Proj_Net
    con_loss = torch.nn.MSELoss()
    cat_loss = torch.nn.CrossEntropyLoss()
    noi_loss = torch.nn.MSELoss()

    # optim for Proj_Net
    lr = 0.001
    proj_optimizer = optim.Adam(Pro.parameters(), lr)
    
    # data load
    training_loader, training_data = data_loading()

    # result
    Res = {'loss_list': []}

    Pro.train()

    for epoch in range(num_epoch):

        for batch_i, (images, labels) in enumerate(training_loader):

            z = z_batch_generation(labels)
            cat_z = torch.Tensor(z[:, :10])
            con_z = torch.Tensor(z[:, 11:13])
            noi_z = torch.Tensor(z[:, 12:])
            z = torch.Tensor(z)
            x = Gen(z)

            proj_optimizer.zero_grad()
            cat_proj, con_proj, noi_proj = Pro(x)
            cat_proj = cat_proj.view((BATCH_SIZE, CATEGORY))
            con_proj = con_proj.view((BATCH_SIZE, LATENT_NUM))
            noi_proj = noi_proj.view((BATCH_SIZE, 62))
            cat_proj = torch.max(cat_proj, 1)[1]

            proj_loss = (lamda * con_loss(con_proj, con_z) + lamda * cat_loss(
                cat_z, cat_proj) + lamda_con * noi_loss(noi_z, noi_proj)) / (
                    2 * lamda + lamda_con)
            proj_loss.backward()
            proj_optimizer.step()

            if batch_i % 500 == 0:
                Res['loss_list'].append(proj_loss.item())
                print('Epoch [{:5d}/{:5d}] | porj_loss: {:6.4f}]'.format(
                    epoch + 1, num_epoch, proj_loss.item()))

    torch.save(Pro, './Projector.pkl')
    res_store = pd.DataFrame(Res)
    res_store.to_csv('./proj_loss.csv', index=False)

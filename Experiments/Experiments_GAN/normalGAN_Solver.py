import numpy as np
import torch
import sys
sys.path.append('./PyTorch-GAN-Mnist')
from model import generator

import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torchvision.utils as vutils

from utils import *
from dataloader import get_data
import pandas as pd

sys.path.append('./final_code')
import LP_Compressed_Sensing as lcs


model_path = './PyTorch-GAN-Mnist/Generator_epoch_200.pth'
train_image_file = './data/MNIST/raw/train-images-idx3-ubyte'
train_label_file = './data/MNIST/raw/train-labels-idx1-ubyte'

device = 'cpu'

params = {
    'batch_size' : 1,
    'num_epochs' : 10000,
    'lr' : 0.00016,
    'num_z' : 128,
    'N' : 28,
    'DATA_LEN' : 150,
}

def load_gan(model_path):
    
    G = torch.load(model_path)
    G.eval()

    return G


class CS_Solver(torch.nn.Module):
    '''
    used to solve ||AG(z) - y|| problem, where G is normal GAN's Generator.
    '''
    def __init__(self, in_features, out_features):
        super(CS_Solver, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.z_layer = nn.Linear(self.in_features, self.out_features, bias=False)
        
    def forward(self, X):
        out_z = self.z_layer(X)
        tmp_z = out_z.view((1, 128))
        return out_z

def generate_A_y(m, n, data):
    m = m
    n = n
    data = data.view((n*n)).detach().numpy()
    # observation metrix
    A = np.random.normal(size=(m, n*n))
    # measurement
    y = A.dot(data)


    return torch.Tensor(A), torch.Tensor(y)

def load_data():

    train_images, img_rows, img_cols = lcs.decode_images(train_image_file)
    train_labels = lcs.decode_labels(train_label_file)
    original_data = train_images[:params['DATA_LEN']]
    original_label = train_labels[:params['DATA_LEN']]
    #original_id = [info_map[str(int(original_label[i]))] for i in range(params['DATA_LEN'])]

    return original_data, original_label

def final_calculating_error(constra, original):

    original = np.array(original).reshape((28, 28))
    error = np.abs(constra - original)**2
    error = np.average(np.sqrt(error))

    return error


def recover(target, idx, m_p):
    print(" ---- The idx: ", idx)

    target = torch.Tensor(target)
    A, y = generate_A_y(int(m_p*params['N']), params['N'], target)

    cnn = CS_Solver(1, 128)

    #init_img = G(init_z.view((1, 74, 1, 1))).view((28, 28)).detach().numpy()
    #plt.imshow(init_img)
    cnn.train()

    solver_loss = torch.nn.MSELoss()
    solver_optim = torch.optim.Adam(cnn.parameters(), lr=params['lr'])
    loss_list = []
    loss_cond = []
    
    for epoch in range(params['num_epochs']):

        data = torch.ones(1, 1)
        out_z = cnn(data).view((1, 128))

        G_out = G(out_z)
        rescale_G = (G_out - torch.min(G_out) / (torch.max(G_out) - torch.min(G_out))) * 255
        cal_from_G = torch.mm(A, rescale_G.view((784, 1))).view(int(m_p*params['N']))
        loss_tar = solver_loss(cal_from_G, y)

        solver_optim.zero_grad()

        loss_tar.backward()
        loss_list.append(loss_tar.item())

        # stop condition
        if epoch % 1000 == 999:
            stop = np.sum(np.diff(loss_cond)>0) > 900
            if stop:
                print("stop!!!")
                break;
            loss_cond = []
        else:
            loss_cond.append(loss_tar.item())

        if loss_tar.item() <= 1e-3:
            print("RECOVER SUCCESS")
            break;

        if epoch % 500 == 0:
            print("in round %d, loss is: "%epoch, loss_tar.item())

        solver_optim.step()

    constra = G(cnn.z_layer.weight.data.view((1, 128)))
    #constra = constra.view((1, 74, 1, 1))

    return constra, loss_list[-1], len(loss_list)


if __name__ == '__main__':
    original_data, original_id = load_data()
    mp_list = [0.1]*50 + [0.125]*50 + [0.15]*50
    G = load_gan(model_path)

    constra_list = []
    ori_list = []
    me_list = []
    re_list = []
    epoch_list = []

    for i in range(params['DATA_LEN']):
        print(" ========  In round %d | %d ============" %(i, params['DATA_LEN']))
        constra, me_loss, epoch_len = recover(original_data[i], original_id[i], mp_list[i])
        constra = constra.detach().numpy().reshape((28, 28))
        re_loss = final_calculating_error(constra, original_data[i])

        re_list.append(re_loss)
        constra_list.append(constra.flatten())
        ori_list.append(original_data[i].flatten())
        me_list.append(me_loss)
        epoch_list.append(epoch_len)

    error_res = {
        'me' : me_list,
        're' : re_list,
        'nc' : epoch_list,
    }

    error_csv = pd.DataFrame(error_res)
    error_csv.to_csv('./normal_gan/error.csv')

    np.savetxt('./normal_gan/ori_img.txt', np.array(ori_list).flatten())
    np.savetxt('./normal_gan/constra_img.txt', np.array(constra_list).flatten())

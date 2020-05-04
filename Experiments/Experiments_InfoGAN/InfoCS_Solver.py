import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('./InfoGAN-Pytorch')
sys.path.append('./final_code')
import LP_Compressed_Sensing as lcs
from models.mnist_model import Generator
from utils import *
from dataloader import get_data

model_path = './InfoGAN-PyTorch/result_v1/model_final.pth'
train_image_file = './data/MNIST/raw/train-images-idx3-ubyte'
train_label_file = './data/MNIST/raw/train-labels-idx1-ubyte'

params = {
    'batch_size' : 1,
    'num_epochs' : 10,
    'lr' : 0.00016,
    'num_z' : 62,
    'num_dis_c' : 1,
    'num_c_dim' : 10,
    'num_con_c' : 2,
    'lay_len' : 74,
    'N' : 28,
    'DATA_LEN' : 3,
}

info_map = {
    '0': 1,
    '1': 8,
    '2': 9,
    '3': 0,
    '4': 3,
    '5': 4,
    '6': 2,
    '7': 7,
    '8': 5,
    '9': 6
}

class ANN_Solver(torch.nn.Module):
    '''
    used to solve the min||AG(z) - y|| problem
    where ``z`` is the weights of the one-layer fully-connected network
    '''
    def __init__(self, in_features, out_features, conti_num):
        super(ANN_Solver, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conti_num = conti_num
        self.z_layer = nn.Linear(self.in_features, self.out_features, bias=False)
        self.c_layer = nn.Linear(self.out_features, self.conti_num, bias=True)

    def forward(self, X):
        out_z = self.z_layer(X)
        tmp_z = out_z.view((1, 1, 62))
        out_c = self.c_layer(tmp_z)
        return out_z, out_c


def make_z_normalize(id_, con_, num_z, batch_size):
    z = torch.randn(batch_size, num_z, 1, 1)
    #print(torch.max(z))
    z = (z - torch.min(z)) / (torch.max(z) - torch.min(z))
    z = torch.sqrt(z)

    id_x = torch.zeros((10, batch_size))
    id_x[id_, :] = 1

    con_ = torch.Tensor(con_)

    return z, id_x, con_


def init_weight_z(model, id_, con_):
    #z, idx = make_z(params['num_dis_c'], params['num_c_dim'], params['num_con_c'], params['num_z'], 1, device=torch.device('cpu'), id_=id_)
    #z[0][-2:] = con_
    z, idx, con_ = make_z_normalize(id_, con_, 62, params['batch_size'])
    z_weight = z.view(model.z_layer.weight.shape)
    c_bias = con_.view(model.c_layer.bias.shape)
    model.z_layer.weight.data = z_weight
    model.c_layer.bias.data = c_bias

    print(id_)
    return z_weight, c_bias, idx


def generate_A_y(m, n, data):
    m = m
    n = n
    data = data.view((n*n)).detach().numpy()
    # observation metrix
    A = np.random.normal(size=(m, n*n))
    # measurement
    y = A.dot(data)


    return torch.Tensor(A), torch.Tensor(y)

def load_GAN():

    state_dict = torch.load('./InfoGAN-PyTorch/result_v1/model_final.pth', map_location=torch.device('cpu'))
    G = Generator()
    G.load_state_dict(state_dict['netG'])
    G.eval()

    return G

def load_data():

    train_images, img_rows, img_cols = lcs.decode_images(train_image_file)
    train_labels = lcs.decode_labels(train_label_file)
    original_data = train_images[:params['DATA_LEN']]
    original_label = train_labels[:params['DATA_LEN']]
    original_id = [info_map[str(int(original_label[i]))] for i in range(params['DATA_LEN'])]

    return original_data, original_id

def final_calculating_error(constra, original):

    original = np.array(original).reshape((28, 28))
    error = np.sum(np.abs(constra - original)**2)
    error = np.average(np.sqrt(error))

    return error


def recover(target, idx, m_p):
    print(" ---- The idx: ", idx)

    target = torch.Tensor(target)
    A, y = generate_A_y(int(m_p*params['N']), params['N'], target)

    ann = ANN_Solver(1, 62, 2)
    #print(ann)
    init_con = np.random.normal(0.005, 0.00025, 2).astype(float)
    init_con = torch.tensor(init_con).view((2, 1, 1)).float()
    z, c, latent_code = init_weight_z(ann, idx, init_con)
    c = c.view((1, 2))
    latent_code = latent_code.view((1, 10))

    init_z = torch.cat((z.view((1, 62)), latent_code, c), dim=1)
    init_img = G(init_z.view((1, 74, 1, 1))).view((28, 28)).detach().numpy()
    #plt.imshow(init_img)
    ann.train()

    solver_loss = torch.nn.MSELoss()
    solver_optim = torch.optim.Adam(ann.parameters(), lr=params['lr'])
    loss_list = []
    loss_cond = []
    for epoch in range(params['num_epochs']):

        data = torch.ones(1, 1)
        out_z, out_c = ann(data)

        out = torch.cat((out_z.view((1, 62)), latent_code, out_c.view((1, 2))), dim=1)
        G_out = G(out.view((1, 74, 1, 1)))
        rescale_G = (G_out - torch.min(G_out) / (torch.max(G_out) - torch.min(G_out))) * 255
        cal_from_G = torch.mm(A, rescale_G.view((784, 1))).view(int(m_p*params['N']))
        loss_tar = solver_loss(cal_from_G, y)

        solver_optim.zero_grad()

        loss_tar.backward()
        loss_list.append(loss_tar.item())

        # stop condition
        if epoch % 100 == 99:
            stop = np.sum(np.diff(loss_cond)>0) > 80
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

    constra = torch.cat((ann.z_layer.weight.data.view((1, 62)), latent_code, ann.c_layer.bias.data.view((1, 2))), dim=1)
    constra = constra.view((1, 74, 1, 1))

    return init_img, constra, loss_list[-1]


if __name__ == '__main__':

    # data loading
    original_data, original_id = load_data()
    #mp_lists = [[0.1+0.5*i]*params['DATA_LEN'] for i in range(5)]
    mp_list = [0.1]*params['DATA_LEN']
    G = load_GAN()
    init_list = []
    constra_list = []
    measurement_error = []
    reconstruction_error = []

    for i in range(params['DATA_LEN']):
        init_img, constra_z, final_me = recover(original_data[i], original_id[i], mp_list[i])
        constra_img = G(constra_z).detach().numpy().reshape((28, 28))
        final_re = final_calculating_error(constra_img, original_data[i])
        init_list.append(init_img)
        constra_list.append(constra_img)
        measurement_error.append(final_me)
        reconstruction_error.append(final_re)

    error_res = {
        'me': measurement_error,
        're': reconstruction_error
    }

    error_csv = pd.DataFrame(error_res)
    error_csv.to_csv('./recover_1/error.csv')

    np.savetxt('./recover_1/init_img.txt', np.array(init_list).flatten())
    np.savetxt('./recover_1/constra_img.txt', np.array(constra_list).flatten())










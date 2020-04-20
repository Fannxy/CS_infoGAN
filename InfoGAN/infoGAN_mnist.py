# for calculation
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

BATCH_SIZE = 64
CATEGORY = 10
LATENT_NUM = 2


def data_loading():
    '''
    used for MNIST data loading
    '''
    num_workers = 0
    batch_size = 64
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader


def create_continuous_noise(num_continuous, style_size, size):
    '''
    used to create continuous noise
    
    paramters:
    ----------
    num_continuous: how many continuous latent code
    style_size: size of pure noise
    size: batch size - hyper param
    '''
    
    contin = np.random.uniform(-1.0, 1.0, size=(size, num_continuous))
    style = np.random.standard_normal(size=(size, style_size))
    
    return np.hstack([contin, style])



def create_discrete_noise(category, size):
    '''
    used to create discrete noise
    
    parameters:
    -----------
    category: value nums of discrete variables
    size: batch_size
    
    '''
    noise = []
    for i in category:
        noise.append(np.random.randint(0, i, size=size))

    return noise

def make_one_hot(indices, size):
    
    as_one_hot = np.zeros((indices.shape[0], size))
    as_one_hot[np.arange(0, indices.shape[0]), indices] = 1.0
    
    return as_one_hot


def create_infogan_noise(category, num_continuous, style_size):
    '''
    genetare the final noise
    
    noise structure:
    np.array of shape (BATCH_SIZE, size)
    size = category(10) + num_continuius(2) + style_size(62) = 74
    structure{
        first 10 - one_hot_array representing one integer of 0-9
        next 2 - continuous latent code
        last 62 - pure noise
    }
    '''
    noise = []
    cat_num = np.array([CATEGORY])
    discrete_sample = create_discrete_noise(cat_num, BATCH_SIZE)
    continuous_sample = create_continuous_noise(num_continuous, style_size, BATCH_SIZE)
    
    for i, sample in zip(cat_num, discrete_sample):
        noise.append(make_one_hot(sample, size=i))
    noise.append(continuous_sample)
    
    return np.hstack(noise)


def weights_init_normal(m):
    '''
    used for param initialization
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Generator

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_size, 32*14*14),
        )
        self.l2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
        )
        self.l3 = nn.Sequential(
            #nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.Conv2d(16, 1, 3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], 32, 14, 14)
        x = F.leaky_relu(self.l2(x))
        x = self.l3(x)
        img = F.tanh(x)
        
        return x


# Discriminator

class Discriminator(nn.Module):
    '''
    f can take F.leaky_relu, with negative_slope=0.2
    '''
    
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        #self.map1 = nn.Linear(input_size, hidden_size)
        self.map1 = nn.Sequential(
                        nn.Conv2d(input_size, hidden_size, (16, 16), stride=2),
                    )
        self.map2 = nn.Sequential(
                        nn.Conv2d(hidden_size, 128, (4, 4), stride=2),
                    )
        self.map3 = nn.Sequential(
                        nn.BatchNorm2d(128),
                    )
        self.lin_layer = nn.Linear(128*2*2, 128)
        self.f = F.relu
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # output layers
        self.adv_layer = nn.Sequential(nn.Linear(128, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128, CATEGORY), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128, LATENT_NUM))
    
    def forward(self, x):
        #x (64, 1, 28, 28)
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        x = self.map3(x)
        x = x.view((64, 128*2*2))
        out = self.f(self.lin_layer(x))
        
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent = self.latent_layer(out)
        
        return validity, label, latent


if __name__ == '__main__':

    # device
    #cuda = True if torch.cuda.is_available() else False
    cuda = False
    # parameters
    z_size = 74               # input_size
    g_output_size = 28*28     # generator output size

    input_channel = 1         # input channel for discriminator
    d_output_size = 1         # validity output size
    d_hidden_size = 32        # hidden layer size for discriminator
    
    batch_size = BATCH_SIZE

    # initialization
    D = Discriminator(input_channel, d_hidden_size, d_output_size)
    G = Generator(z_size, g_output_size)

    # infoGAN loss

    # loss weights
    lambda_cat = 1
    lambda_con = 0.1

    # loss functions
    adv_loss = torch.nn.MSELoss()
    cat_loss = torch.nn.CrossEntropyLoss()
    con_loss = torch.nn.MSELoss()


    # cuda:
    if cuda:
        G.cuda()
        D.cuda()
        adv_loss.cuda()
        cat_loss.cuda()
        con_loss.cuda()

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # optimizater
    lr_g = 0.001
    lr_d = 0.0002
    d_optimizer = optim.Adam(D.parameters(), lr_d)
    g_optimizer = optim.Adam(G.parameters(), lr_g)
    info_optimizer = optim.Adam(itertools.chain(G.parameters(), D.parameters()), lr_d)

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)


    num_epochs = 100

    # results stored here
    Results = {
        'd_loss': [],
        'g_loss': [],
        'info_loss': [],
        #'samples': []
    }
    samples = []

    sample_size = BATCH_SIZE
    cat_num = np.array([CATEGORY])
    fixed_z = create_infogan_noise(cat_num, LATENT_NUM, 62)
    #fixed_z = torch.from_numpy(fixed_z).float()
    fixed_z = Variable(FloatTensor(torch.from_numpy(fixed_z).float()))
    real_labels = Variable(FloatTensor(np.ones(batch_size)))
    fake_labels = Variable(FloatTensor(np.zeros(batch_size)))



    D.train()
    G.train()
    
    train_loader = data_loading()

    for epoch in range(num_epochs):
        
        for batch_i, (real_images, _) in enumerate(train_loader):

            #===============================
            #      Desciminator
            #===============================
            batch_size = real_images.size(0)
            real_images = real_images*2 - 1
            
            d_optimizer.zero_grad()
            
            # 1. train with real image
            if real_images.size()[0]!=64:
                break;
            real_pred, _, _ = D(real_images)
            if real_pred.size()[0] != real_labels.size()[0]:
                break;        
            
            # 2. train with fake image
            cat_num = np.array([CATEGORY])
            z = create_infogan_noise(cat_num, LATENT_NUM, 62)
            z = Variable(FloatTensor(z))
            fake_images = G(z)

            # 3. compute the loss
            fake_pred, _, _ = D(fake_images)
            #d_fake_loss = adv_loss(fake_pred.squeeze(), fake_labels)
            
            # 4. backpop
            d_loss = adv_loss(real_pred.squeeze(), real_labels) + adv_loss(fake_pred.squeeze(), fake_labels)
            d_loss.backward()
            d_optimizer.step()
            
            
            #===============================
            #      Generator
            #===============================
            g_optimizer.zero_grad()
            fake_images = G(z)
            
            # compute the loss
            validity, _, _ = D(fake_images)
            g_loss = adv_loss(validity.squeeze(), real_labels)
            
            #backpop
            g_loss.backward()
            g_optimizer.step()
            
            #===============================
            #      information loss
            #===============================
            info_optimizer.zero_grad()
            
            cat_num = np.array([CATEGORY])
            z = create_infogan_noise(cat_num, LATENT_NUM, 62)
            label_input = z[:, :10]
            con_input = z[:, 11:13]
            #z = torch.from_numpy(z).float()
            z = Variable(FloatTensor(z))

            label_input = Variable(FloatTensor(label_input))
            label_indices = torch.max(label_input, 1)[1]
            con_input = Variable(FloatTensor(con_input))
            gen_images = G(z)
            
            _, pred_label, pred_code = D(gen_images)
            
            info_loss = lambda_cat*cat_loss(pred_label, label_indices) + lambda_con*con_loss(pred_code, con_input)
            
            info_loss.backward()
            info_optimizer.step()
            

            if batch_i % 500 == 0:
                print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f} | info_loss: {:6.4f}'.format(
                        epoch+1, num_epochs, d_loss.item(), g_loss.item(), info_loss.item()))
            
        #losses.append((d_loss.item(), g_loss.item(), info_loss.item()))
        Results['d_loss'].append(d_loss.item())
        Results['g_loss'].append(g_loss.item())
        Results['info_loss'].append(info_loss.item())

        G.eval()
        samples_z = G(fixed_z)
        samples.append(samples_z)
        G.train()
         
        with open('./result_v3/train_samples.pkl', 'wb') as f:
            pkl.dump(samples, f)

        torch.save(G, './result_v3/mnist_G.pkl')
        torch.save(D, './result_v3/mnist_D.pkl')
        res_store = pd.DataFrame(Results)
        res_store.to_csv('./result_v3/results.csv', index=False)


    with open('./result_v3/train_samples.pkl', 'wb') as f:
        pkl.dump(samples, f)

    torch.save(G, './result_v3/mnist_G.pkl')
    torch.save(D, './result_v3/mnist_D_v3.pkl')
    res_store = pd.DataFrame(Results)
    res_store.to_csv('./result_v3/results.csv', index=False)




            
            
            

            


import numpy as np
import pandas as pd
import torch
import torchvision

import struct
import matplotlib.pyplot as plt
from scipy.optimize import linprog

np.random.seed(1337)


train_image_file = './data/MNIST/raw/train-images-idx3-ubyte'


def decode_images(image_file):
    '''
    use to decode idx3 type file data
    '''
    bin_data = open(image_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii' # big endian
    mc_num, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    
    print("mc_num: ", mc_num)
    print("num_images: ", num_images)
    print("num_rows: ", num_rows)
    print("num_cols: ", num_cols)
    
    # decode dataset
    image_size = num_rows * num_cols
    print(image_size)
    offset += struct.calcsize(fmt_header)
    
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, image_size))
    
    for i in range(num_images):
        if (i+1) % 10000 == 0:
            print("dealing ... ", i)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((image_size))
        offset += struct.calcsize(fmt_image)
        
    return images, num_rows, num_cols


def decode_labels(label_file):
    '''
    use to decode idx1 type label data
    '''
    
    bin_data = open(label_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    mc_num, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print("dealing ... ", i)
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)

    return labels


def lp1_cs(m, n, ori_data):
    '''
    using lp1 minimization addressing CS

    Parameters
    ----------
    m: using m ponits
    n: the original size
    A: observing matrix
    ori_data: original data
    '''
    m = m
    n = n
    A = np.random.randint(low=0, high=2, size=(m, n))

    img = ori_data
    b = A.dot(img)

    I = np.identity(n)
    z = np.zeros(shape=(m, n))
    b_0 = np.zeros(shape=(n))
    c_x = np.zeros(shape=(n))
    c_t = np.ones(shape=(n))

    c = np.append(c_x, c_t)
    con_1 = np.hstack((I, -I))
    con_2 = np.hstack((-I, -I))
    con_3 = np.hstack((A, z))
    con_4 = np.hstack((-A, z))

    P = np.vstack((con_1, con_2, con_3, con_4))
    B = np.hstack((b_0, b_0, b, -b))

    res = linprog(c, A_ub=P, b_ub=B, method='interior-point')['x'][:n]

    return res




if __name__ == '__main__':

    # data loading
    training_image, image_row, image_col = decode_images(train_image_file)
    test_data = training_image[:10]

    m_p = [0.1+0.05*step for step in range(10)]
    n = image_row * image_col
    m_list = [int(n*per) for per in m_p]

    Results = {
       'diff_list': [],
        'res_list': []
    }

    for i in range(10):
        recon = lp1_cs(m_list[i], n, test_data[i])
        diff = np.linalg.norm(recon-training_image[0])/len(recon)
        Results['diff_list'].append(diff)
        Results['res_list'].append(recon)


    res_store = pd.DataFrame(Results)
    res_store.to_csv('./result_cs.csv', index=False)





    

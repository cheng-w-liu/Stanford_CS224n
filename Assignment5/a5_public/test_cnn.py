#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_cnn.py: sanity check for CNN module
"""

import numpy as np
import torch
from cnn import CNN

def main():

    seed = 1234
    torch.manual_seed(seed)

    batch_size = 3
    char_embed = 2
    m_word = 5
    X_reshaped = torch.randn(batch_size, char_embed, m_word)
    #print('X_reshaped:')
    #print(X_reshaped)

    word_embed = 4
    cnn = CNN(word_embed, char_embed, 2)
    X_conv_out = cnn(X_reshaped)

    #print('X_conv_out:')
    #print(X_conv_out)
    EXPECTED_OUTPUT = np.array(
        [[ 0.46302482, -0.        ,  0.39172465,  0.3098483 ],
         [ 0.61179116, -0.        ,  0.24779617,  0.30089255],
         [ 0.0726867 ,  0.54824059,  0.33242535,  0.53520018]]
    )
    assert np.all(np.isclose(X_conv_out.detach().numpy(), EXPECTED_OUTPUT, atol=0.0001))
    print('sanity check passed for CNN network')



if __name__ == '__main__':
    main()

    # import torch
    # import numpy
    # from scipy.signal import correlate2d

    # X_reshaped = np.array(
    #     [[[-0.1117, -0.4966, 0.1631, -0.8817, 0.0539],
    #       [0.6684, -0.0597, -0.4675, -0.2153, 0.8840]],
    #      [[-0.7584, -0.3689, -0.3424, -1.4020, 1.4255],
    #       [0.7987, -1.4949, 0.8810, -1.1786, -0.9340]],
    #      [[-0.5675, -0.2772, -0.4030, 0.4195, 0.9380],
    #       [0.0078, -0.3139, -1.1567, 1.8409, -1.0174]]]
    # )
    # batch_size = X_reshaped.shape[0]
    # m_word = X_reshaped.shape[2]
    #
    # kernels = np.array([[[-0.0811, -0.4345],
    #                      [0.3839, 0.3083]],
    #
    #                     [[0.2528, 0.3988],
    #                      [0.1839, 0.2658]],
    #
    #                     [[0.4149, -0.1007],
    #                      [-0.3900, -0.2459]],
    #
    #                     [[-0.0667, -0.0549],
    #                      [-0.0034, 0.2865]]])
    #
    # f = kernels.shape[0]
    # kernel_size = kernels.shape[2]
    #
    # X_conv = np.zeros((batch_size, f, m_word - kernel_size + 1))
    #
    # for b in range(batch_size):
    #     for i in range(f):
    #         X_conv[b, i, :] = correlate2d(X_reshaped[b, :, :], kernels[i, :, :], mode='valid')[0]
    #
    # X_conv = X_conv * (X_conv > 0)
    #
    # X_conv = np.amax(X_conv, axis=2)
    #
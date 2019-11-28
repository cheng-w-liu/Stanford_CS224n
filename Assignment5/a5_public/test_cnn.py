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
    cnn = CNN(char_embed, word_embed, 2)
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
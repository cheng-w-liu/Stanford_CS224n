#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_highway.py: sanity check for Highway module
"""

import numpy as np
import torch
from highway import Highway

def main():

    seed = 1234
    torch.manual_seed(seed)

    emb_dim = 3
    highway = Highway(emb_dim)
    N = 2
    X_conv_out = torch.randn(N, emb_dim)
    X_highway = highway(X_conv_out)
    EXPECTED_OUTPUT = np.array([[-0.07900741, -0.1736746 ,  0.00824207], [ 0.48591639,  0.11661039,  0.32287086]])
    assert np.all(np.isclose(X_highway.detach().numpy(), EXPECTED_OUTPUT, atol=0.001))
    print('sanity check passed for Highway network')


if __name__ == '__main__':
    main()
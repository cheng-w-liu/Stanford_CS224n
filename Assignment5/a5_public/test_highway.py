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

    # import torch
    # import numpy
    #
    # X_conv_out = np.array([[-0.1549, -1.3706, -0.1319], [0.8848, -0.2611, 0.6104]])
    #
    # w_proj_weight = np.array([[-0.5439, -0.1133, -0.2773], [-0.1540, -0.5100, 0.2317], [-0.5175, -0.0368, 0.2007]])
    # w_proj_bias = np.array([-0.1946, 0.3276, 0.0728]).reshape(-1, 1)
    #
    # X_proj = (np.matmul(w_proj_weight, X_conv_out.T) + w_proj_bias).T
    # X_proj = X_proj * (X_proj > 0)
    #
    # gate_proj_weight = np.array([[0.3174, 0.3704, -0.2549], [0.2098, -0.2498, 0.1810], [-0.3017, 0.2671, 0.1169]])
    # gate_proj_bias = np.array([-0.2259, -0.2832, 0.1494]).reshape(-1, 1)
    #
    # X_gate = (np.matmul(gate_proj_weight, X_conv_out.T) + gate_proj_bias).T
    # X_gate = 1.0 / (1.0 + np.exp(-X_gate))
    #
    # X_highway = np.multiply(X_gate, X_proj) + np.multiply(1.0 - X_gate, X_conv_out)